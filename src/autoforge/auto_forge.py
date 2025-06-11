import argparse
import sys
import os
import time
import traceback

import configargparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

from autoforge.Helper import PruningHelper
from autoforge.Helper.FilamentHelper import hex_to_rgb, load_materials
from autoforge.Helper.Heightmaps.ChristofidesHeightMap import (
    run_init_threads,
)

from autoforge.Helper.ImageHelper import resize_image, imread
from autoforge.Helper.OutputHelper import (
    generate_stl,
    generate_swap_instructions,
    generate_project_file,
)
from autoforge.Modules.Optimizer import FilamentOptimizer

# check if we can use torch.set_float32_matmul_precision('high')
if torch.__version__ >= "2.0.0":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception as e:
        print("Warning: Could not set float32 matmul precision to high. Error:", e)
        pass


def parse_args():
    parser = configargparse.ArgParser()
    parser.add_argument("--config", is_config_file=True, help="Path to config file")

    parser.add_argument(
        "--input_image", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="",
        help="Path to CSV file with material data",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="",
        help="Path to json file with material data",
    )
    parser.add_argument(
        "--output_folder", type=str, default="output", help="Folder to write outputs"
    )

    parser.add_argument(
        "--iterations", type=int, default=6000, help="Number of optimization iterations"
    )

    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=1.0,
        help="Fraction of iterations for keeping the tau at the initial value",
    )

    parser.add_argument(
        "--learning_rate_warmup_fraction",
        type=float,
        default=0.01,
        help="Fraction of iterations that the learning rate is increasing (warmup)",
    )

    parser.add_argument(
        "--init_tau",
        type=float,
        default=1.0,
        help="Initial tau value for Gumbel-Softmax",
    )

    parser.add_argument(
        "--final_tau",
        type=float,
        default=0.01,
        help="Final tau value for Gumbel-Softmax",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.015,
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
        "--visualize",
        type=bool,
        default=True,
        help="Enable visualization during optimization",
        action=argparse.BooleanOptionalAction,
    )

    # Instead of an output_size parameter, we use stl_output_size and nozzle_diameter.
    parser.add_argument(
        "--stl_output_size",
        type=int,
        default=150,
        help="Size of the longest dimension of the output STL file in mm",
    )

    parser.add_argument(
        "--nozzle_diameter",
        type=float,
        default=0.4,
        help="Diameter of the printer nozzle in mm (details smaller than half this value will be ignored)",
    )

    parser.add_argument(
        "--early_stopping",
        type=int,
        default=2000,
        help="Number of steps without improvement before stopping",
    )

    parser.add_argument(
        "--perform_pruning",
        type=bool,
        default=True,
        help="Perform pruning after optimization",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--fast_pruning",
        type=bool,
        default=True,
        help="Use fast pruning method",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--fast_pruning_percent",
        type=float,
        default=0.5,
        help="Percentage of increment search for fast pruning",
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
        "--pruning_max_layer",
        type=int,
        default=75,
        help="Max number of layers allowed after pruning",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Specify the random seed, or use 0 for automatic generation",
    )

    parser.add_argument(
        "--mps",
        action="store_true",
        help="Use the Metal Performance Shaders (MPS) backend, if available.",
    )

    parser.add_argument(
        "--run_name", type=str, help="Name of the run used for TensorBoard logging"
    )

    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )

    parser.add_argument(
        "--num_init_rounds",
        type=int,
        default=8,
        help="Number of rounds to choose the starting height map from.",
    )

    parser.add_argument(
        "--num_init_cluster_layers",
        type=int,
        default=-1,
        help="Number of layers to cluster the image into.",
    )

    parser.add_argument(
        "--disable_visualization_for_gradio",
        type=int,
        default=0,
        help="Simple switch to disable the matplotlib render window for gradio rendering.",
    )

    parser.add_argument(
        "--best_of",
        type=int,
        default=1,
        help="Run the program multiple times and output the best result.",
    )

    args = parser.parse_args()
    return args


def start(args):
    if args.num_init_cluster_layers == -1:
        args.num_init_cluster_layers = args.max_layers // 2

    # check if csv or json is given
    if args.csv_file == "" and args.json_file == "":
        print("Error: No CSV or JSON file given. Please provide one of them.")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    os.makedirs(args.output_folder, exist_ok=True)

    # Basic checks
    if not (args.background_height / args.layer_height).is_integer():
        print(
            "Error: Background height must be a multiple of layer height.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found.", file=sys.stderr)
        sys.exit(1)

    if args.csv_file != "" and not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.", file=sys.stderr)
        sys.exit(1)
    if args.json_file != "" and not os.path.exists(args.json_file):
        print(f"Error: Json file '{args.json_file}' not found.", file=sys.stderr)
        sys.exit(1)

    random_seed = args.random_seed
    if random_seed == 0:
        random_seed = int(time.time() * 1000) % 1000000
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Prepare background color
    bgr_tuple = hex_to_rgb(args.background_color)
    background = torch.tensor(bgr_tuple, dtype=torch.float32, device=device)

    # Load materials
    material_colors_np, material_TDs_np, material_names, _ = load_materials(args)
    material_colors = torch.tensor(
        material_colors_np, dtype=torch.float32, device=device
    )
    material_TDs = torch.tensor(material_TDs_np, dtype=torch.float32, device=device)

    # Read input image
    img = imread(args.input_image, cv2.IMREAD_UNCHANGED)
    computed_output_size = int(round(args.stl_output_size * 2 / args.nozzle_diameter))
    print(f"Computed solving pixel size: {computed_output_size}")
    alpha = None
    if img.shape[2] == 4:
        # Extract alpha channel
        alpha = img[:, :, 3]
        alpha = alpha[..., None]
        # Compute output size based on stl_output_size and nozzle_diameter

        alpha = resize_image(alpha, computed_output_size)
        # Remove the alpha channel from the image
        img = img[:, :, :3]

    # Convert image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # For the final resolution
    output_img_np = resize_image(img, computed_output_size)
    output_target = torch.tensor(output_img_np, dtype=torch.float32, device=device)

    global_logits_init = None
    # Initialize pixel_height_logits from the large (final) image
    print("Initalizing height map. This can take a moment...")
    # Default initialization
    pixel_height_logits_init, global_logits_init, pixel_height_labels = (
        run_init_threads(
            output_img_np,
            args.max_layers,
            args.layer_height,
            bgr_tuple,
            random_seed=random_seed,
            num_threads=args.num_init_rounds,
            init_method="kmeans",
            cluster_layers=args.num_init_cluster_layers,
            material_colors=material_colors_np,
        )
    )

    # if we have an alpha mask we set the height for those pixels to -13.815512 (the lowest init sigmoid value)
    # Now with unlocked height map we probably need to think about changing this somehow. TODO: Think about this.
    if alpha is not None:
        pixel_height_logits_init[alpha < 128] = -13.815512

    # VGG Perceptual Loss
    # We currently disable this as it is not used in the optimization.
    perception_loss_module = None  # MultiLayerVGGPerceptualLoss().to(device).eval()

    # Create an optimizer instance
    optimizer = FilamentOptimizer(
        args=args,
        target=output_target,
        pixel_height_logits_init=pixel_height_logits_init,
        pixel_height_labels=pixel_height_labels,
        global_logits_init=global_logits_init,
        material_colors=material_colors,
        material_TDs=material_TDs,
        background=background,
        device=device,
        perception_loss_module=perception_loss_module,
    )

    # Main optimization loop
    print("Starting optimization...")
    tbar = tqdm(range(args.iterations))
    with torch.autocast(device.type, dtype=torch.bfloat16):
        for i in tbar:
            loss_val = optimizer.step(record_best=i % 8 == 0)

            optimizer.visualize(interval=50)
            optimizer.log_to_tensorboard(interval=100)

            if (i + 1) % 100 == 0:
                tbar.set_description(
                    f"Iteration {i + 1}, Loss = {loss_val:.4f}, best validation Loss = {optimizer.best_discrete_loss:.4f}, learning_rate= {optimizer.current_learning_rate:.6f}"
                )
            if (
                optimizer.best_step is not None
                and optimizer.num_steps_done - optimizer.best_step > args.early_stopping
            ):
                print(
                    "Early stopping after",
                    args.early_stopping,
                    "steps without improvement.",
                )
                break

    post_opt_step = 0

    optimizer.log_to_tensorboard(
        interval=1, namespace="post_opt", step=(post_opt_step := post_opt_step + 1)
    )

    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16):
            if args.perform_pruning:
                optimizer.prune(
                    max_colors_allowed=args.pruning_max_colors,
                    max_swaps_allowed=args.pruning_max_swaps,
                    min_layers_allowed=args.min_layers,
                    max_layers_allowed=args.pruning_max_layer,
                    search_seed=True,
                    fast_pruning=args.fast_pruning,
                    fast_pruning_percent=args.fast_pruning_percent,
                )
                optimizer.log_to_tensorboard(
                    interval=1,
                    namespace="post_opt",
                    step=(post_opt_step := post_opt_step + 1),
                )

            disc_global, disc_height_image = optimizer.get_discretized_solution(
                best=True
            )

            final_loss = PruningHelper.get_initial_loss(
                optimizer.best_params["global_logits"].shape[0], optimizer
            )
            # write to text file
            with open(os.path.join(args.output_folder, "final_loss.txt"), "w") as f:
                f.write(f"{final_loss}")

            print("Done. Saving outputs...")
            # Save Image
            comp_disc = optimizer.get_best_discretized_image()
            args.max_layers = optimizer.max_layers

            optimizer.log_to_tensorboard(
                interval=1,
                namespace="post_opt",
                step=(post_opt_step := post_opt_step + 1),
            )

            comp_disc_np = comp_disc.cpu().numpy().astype(np.uint8)
            comp_disc_np = cv2.cvtColor(comp_disc_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(args.output_folder, "final_model.png"), comp_disc_np
            )

            stl_filename = os.path.join(args.output_folder, "final_model.stl")
            height_map_mm = (
                disc_height_image.cpu().numpy().astype(np.float32)
            ) * args.layer_height
            generate_stl(
                height_map_mm,
                stl_filename,
                args.background_height,
                maximum_x_y_size=args.stl_output_size,
                alpha_mask=alpha,
            )

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
            with open(
                os.path.join(args.output_folder, "swap_instructions.txt"), "w"
            ) as f:
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
            return final_loss

def main():
    args = parse_args()
    final_output_folder = args.output_folder
    run_best_loss = 1000000000
    if args.best_of == 1:
        start(args)
    else:
        temp_output_folder = os.path.join(args.output_folder, "temp")
        ret = []
        for i in range(args.best_of):
            try:
                print(f"Run {i + 1}/{args.best_of}")
                run_folder = os.path.join(temp_output_folder, f"run_{i + 1}")
                args.output_folder = run_folder
                os.makedirs(args.output_folder, exist_ok=True)
                run_loss = start(args)
                print(f"Run {i + 1} finished with loss: {run_loss}")
                if run_loss < run_best_loss:
                    run_best_loss = run_loss
                    print(f"New best loss found: {run_best_loss} in run {i + 1}")
                ret.append((run_folder, run_loss))
                # garbage collection
                torch.cuda.empty_cache()
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                # close all matplotlib windows if there are any
                import matplotlib.pyplot as plt

                plt.close("all")
            except Exception:
                traceback.print_exc()
        # get run with best loss
        best_run = min(ret, key=lambda x: x[1])
        best_run_folder = best_run[0]
        best_loss = best_run[1]

        # move files from run folder to final output folder
        if not os.path.exists(final_output_folder):
            os.makedirs(final_output_folder)
        for file in os.listdir(best_run_folder):
            src_file = os.path.join(best_run_folder, file)
            dst_file = os.path.join(final_output_folder, file)
            if os.path.isfile(src_file):
                os.rename(src_file, dst_file)


if __name__ == "__main__":
    main()