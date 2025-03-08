import os
import time
import configargparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

from autoforge.Helper.FilamentHelper import hex_to_rgb, load_materials
from autoforge.Helper.ImageHelper import resize_image
from autoforge.Helper.OptimizerHelper import (
    init_height_map,
    composite_image,
    discretize_solution,
    init_height_map_depth_color_adjusted,
)
from autoforge.Helper.OutputHelper import (
    generate_stl,
    generate_swap_instructions,
    generate_project_file,
)
from autoforge.Helper.PruningHelper import prune_redundant_layers
from autoforge.Loss.PerceptionLoss import MultiLayerVGGPerceptualLoss
from autoforge.Modules.Optimizer import FilamentOptimizer


def main():
    parser = configargparse.ArgParser()
    parser.add_argument("--config", is_config_file=True, help="Path to config file")

    parser.add_argument(
        "--input_image", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to CSV file with material data",
    )
    parser.add_argument(
        "--output_folder", type=str, default="output", help="Folder to write outputs"
    )

    parser.add_argument(
        "--iterations", type=int, default=5000, help="Number of optimization iterations"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for optimization",
    )
    parser.add_argument(
        "--layer_height", type=float, default=0.04, help="Layer thickness in mm"
    )
    parser.add_argument(
        "--max_layers", type=int, default=75, help="Maximum number of layers"
    )
    parser.add_argument(
        "--min_layers",
        type=int,
        default=0,
        help="Minimum number of layers. Used for pruning.",
    )
    parser.add_argument(
        "--background_height",
        type=float,
        default=0.4,
        help="Height of the background in mm",
    )
    parser.add_argument(
        "--background_color", type=str, default="#000000", help="Background color"
    )

    parser.add_argument(
        "--output_size",
        type=int,
        default=1024,
        help="Maximum dimension for target image",
    )
    parser.add_argument(
        "--solver_size",
        type=int,
        default=128,
        help="Maximum dimension for solver (fast) image",
    )
    parser.add_argument(
        "--init_tau", type=float, default=1.0, help="Initial tau value for Gumbel-Softmax"
    )
    parser.add_argument(
        "--final_tau", type=float, default=0.01, help="Final tau value for Gumbel-Softmax"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization during optimization",
    )

    parser.add_argument(
        "--stl_output_size",
        type=int,
        default=200,
        help="Size of the longest dimension of the output STL file",
    )

    parser.add_argument(
        "--perform_pruning",
        type=bool,
        default=True,
        help="Perform pruning after optimization",
    )
    parser.add_argument(
        "--pruning_max_colors",
        type=int,
        default=100,
        help="Max number of colors allowed after pruning",
    )
    parser.add_argument(
        "--pruning_max_swaps",
        type=int,
        default=100,
        help="Max number of swaps allowed after pruning",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Specify the random seed, or use 0 for automatic generation",
    )

    parser.add_argument(
        "--use_depth_anything_height_initialization",
        action="store_true",
        help="Use Depth Anything v2 for height image initialization",
    )
    parser.add_argument(
        "--depth_strength",
        type=float,
        default=0.25,
        help="Weight (0 to 1) for blending even spacing with the cluster's average depth",
    )
    parser.add_argument(
        "--depth_threshold",
        type=float,
        default=0.05,
        help="Threshold for splitting two distinct colors based on depth",
    )
    parser.add_argument(
        "--min_cluster_value",
        type=float,
        default=0.1,
        help="Minimum normalized value for the lowest cluster (to avoid pure black)",
    )
    parser.add_argument(
        "--w_depth",
        type=float,
        default=0.5,
        help="Weight for depth difference in ordering_depth",
    )
    parser.add_argument(
        "--w_lum",
        type=float,
        default=1.0,
        help="Weight for luminance difference in ordering_depth",
    )
    parser.add_argument(
        "--order_blend",
        type=float,
        default=0.1,
        help="Slider (0 to 1) blending original luminance ordering (0) and depth-informed ordering (1).",
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Use the Metal Performance Shaders (MPS) backend, if available.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name of the run used for TensorBoard logging"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging"
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    os.makedirs(args.output_folder, exist_ok=True)

    # Basic checks
    assert (args.background_height / args.layer_height).is_integer(), (
        "Background must be multiple of layer_height"
    )

    random_seed = args.random_seed
    if random_seed == 0:
        random_seed = int(time.time())
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Prepare background color
    bgr_tuple = hex_to_rgb(args.background_color)
    background = torch.tensor(bgr_tuple, dtype=torch.float32, device=device)

    # Load materials
    material_colors_np, material_TDs_np, material_names, _ = load_materials(
        args.csv_file
    )
    material_colors = torch.tensor(
        material_colors_np, dtype=torch.float32, device=device
    )
    material_TDs = torch.tensor(material_TDs_np, dtype=torch.float32, device=device)

    # Read input image
    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # This smaller 'target' is the solver resolution
    solver_img_np = resize_image(img, args.solver_size)
    target_solver = torch.tensor(solver_img_np, dtype=torch.float32, device=device)

    # For the final resolution
    output_img_np = resize_image(img, args.output_size)
    output_target = torch.tensor(output_img_np, dtype=torch.float32, device=device)

    # Initialize pixel_height_logits from the large (final) image
    if args.use_depth_anything_height_initialization:
        pixel_height_logits_init = init_height_map_depth_color_adjusted(
            output_img_np,
            args.max_layers,
            depth_strength=args.depth_strength,
            depth_threshold=args.depth_threshold,
            min_cluster_value=args.min_cluster_value,
            w_depth=args.w_depth,
            w_lum=args.w_lum,
            order_blend=args.order_blend,
        )
    else:
        # Default initialization using your existing method
        pixel_height_logits_init = init_height_map(
            output_img_np, args.max_layers, args.layer_height, random_seed=random_seed
        )

    # But we will solve at the smaller resolution => resize that init to solver size
    # Add a dummy channel so that cv2.resize can handle it
    tmp_3d = pixel_height_logits_init[..., None]  # shape (H, W, 1)
    phl_solver_np = resize_image(
        tmp_3d, args.solver_size
    )  # shape (H', W') cause of weird opencv quirk

    # VGG Perceptual Loss
    perception_loss_module = MultiLayerVGGPerceptualLoss().to(device).eval()

    # Create an optimizer instance
    optimizer = FilamentOptimizer(
        args=args,
        target=target_solver,
        pixel_height_logits_init=phl_solver_np,
        material_colors=material_colors,
        material_TDs=material_TDs,
        background=background,
        device=device,
        perception_loss_module=perception_loss_module,
    )

    # Main optimization loop
    print("Starting optimization...")
    tbar = tqdm(range(args.iterations))
    for i in tbar:
        loss_val = optimizer.step(record_best=i % 3 == 0)

        optimizer.visualize(interval=25)
        optimizer.log_to_tensorboard(interval=100)

        if (i + 1) % 100 == 0:
            tbar.set_description(
                f"Iteration {i + 1}, Loss = {loss_val:.4f}, best validation Loss = {optimizer.best_discrete_loss:.4f}"
            )

    post_opt_step = 0

    # After we finish, we can get the final discrete solution at solver resolution
    disc_global, disc_height_image = optimizer.get_discretized_solution(best=True)

    # Do a quick search for a better rng seed
    new_rng_seed, new_loss = optimizer.rng_seed_search(
        optimizer.best_discrete_loss, 100
    )

    if new_loss < optimizer.best_discrete_loss:
        optimizer.best_seed = new_rng_seed
        optimizer.best_discrete_loss = new_loss
        print(f"New best seed found: {new_rng_seed} with loss: {new_loss}")

    optimizer.log_to_tensorboard(interval=1, namespace="post_opt", step=(post_opt_step := post_opt_step + 1))

    # Optionally prune
    if args.perform_pruning:
        disc_global = optimizer.prune(
            max_colors_allowed=args.pruning_max_colors,
            max_swaps_allowed=args.pruning_max_swaps,
            disc_global=disc_global,
            disc_height_image=disc_height_image,
            tau_g=optimizer.best_tau,
        )

        optimizer.log_to_tensorboard(interval=1, namespace="post_opt", step=(post_opt_step := post_opt_step + 1))

        # Do a quick search for a better rng seed
        new_rng_seed, new_loss = optimizer.rng_seed_search(
            optimizer.best_discrete_loss, 500
        )

        optimizer.log_to_tensorboard(interval=1, namespace="post_opt", step=(post_opt_step := post_opt_step + 1))

        if new_loss < optimizer.best_discrete_loss:
            optimizer.best_seed = new_rng_seed
            optimizer.best_discrete_loss = new_loss
            print(f"New best seed found: {new_rng_seed} with loss: {new_loss}")
    else:
        print(
            f"No better seed found. Best seed loss: {optimizer.best_discrete_loss}. Best searched loss: {new_loss}"
        )
    # Now, we want to apply that discrete solution to the full resolution:
    print("Applying solved solution to full resolution...")
    params = {
        "global_logits": disc_global,
        "pixel_height_logits": torch.from_numpy(pixel_height_logits_init).to(device),
    }

    if args.perform_pruning:
        print("Removing redundant layers from the discrete solution...")
        params, pruned_loss, max_layers = prune_redundant_layers(
            params,
            final_tau=args.final_tau,
            h=args.layer_height,
            target=output_target,  # or whichever target image you use for final compositing
            material_colors=material_colors,
            material_TDs=material_TDs,
            background=background,
            rng_seed=optimizer.best_seed,  # using the best seed from your optimizer
            perception_loss_module=perception_loss_module,
            tolerance=1e-3,  # adjust as needed
            min_layers=args.min_layers,
        )
        args.max_layers = max_layers

    disc_global, disc_height_image = discretize_solution(
        params, args.final_tau, args.layer_height, args.max_layers
    )

    print("Done. Saving outputs...")
    # Save Image
    comp_disc = composite_image(
        params["pixel_height_logits"],
        params["global_logits"],
        args.final_tau,
        args.final_tau,
        args.layer_height,
        args.max_layers,
        material_colors,
        material_TDs,
        background,
        mode="discrete",  # or "discrete" if you prefer fully discrete
        rng_seed=optimizer.best_seed,
    )

    optimizer.log_to_tensorboard(interval=1, namespace="post_opt", step=(post_opt_step := post_opt_step + 1))


    comp_disc_np = comp_disc.cpu().numpy().astype(np.uint8)
    comp_disc_np = cv2.cvtColor(comp_disc_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(args.output_folder, "final_model.png"), comp_disc_np)

    stl_filename = os.path.join(args.output_folder, "final_model.stl")
    height_map_mm = (
        disc_height_image.cpu().numpy().astype(np.float32)
    ) * args.layer_height
    generate_stl(height_map_mm, stl_filename, args.background_height, maximum_x_y_size=args.stl_output_size)

    # Swap instructions
    background_layers = int(args.background_height // args.layer_height)
    swap_instructions = generate_swap_instructions(
        disc_global.cpu().numpy(),
        disc_height_image.cpu().numpy(),
        args.layer_height,
        background_layers,
        args.background_height,
        material_names,
    )
    with open(os.path.join(args.output_folder, "swap_instructions.txt"), "w") as f:
        for line in swap_instructions:
            f.write(line + "\n")

    # Project file
    project_filename = os.path.join(args.output_folder, "project_file.hfp")
    generate_project_file(
        project_filename,
        args,
        disc_global.cpu().numpy(),
        disc_height_image.cpu().numpy(),
        output_target.shape[1],
        output_target.shape[0],
        stl_filename,
        args.csv_file,
    )

    print("All done. Outputs in:", args.output_folder)
    print("Happy Printing!")


if __name__ == "__main__":
    main()
