import concurrent
import sys
import os
import time
import traceback
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
from random import shuffle

import cv2
import torch
import numpy as np
from tqdm import tqdm

from autoforge.Helper.FilamentHelper import hex_to_rgb, load_materials
from autoforge.Helper.Heightmaps.ChristofidesHeightMap import run_init_threads
from autoforge.Helper.ImageHelper import resize_image
from autoforge.Modules.Optimizer import FilamentOptimizer


class Config:
    # Update these file paths as needed!
    input_image = "default_input.png"  # Path to input image
    csv_file = "default_materials.csv"  # Path to CSV file with material data
    output_folder = "output"
    iterations = 2000
    learning_rate = 0.015

    warmup_fraction = 0.1 #[0.0, 1.0]
    height_logits_learning_start_fraction = 0.1 #[0.0, 1.0] but should be less than height_logits_learning_full_fraction
    height_logits_learning_full_fraction = 0.5 # [0.0, 1.0] but should be more than height_logits_learning_start_fraction
    init_tau = 1.0 # [0.0, 1.0] needs to be more than final_tau
    final_tau = 0.01 # [0.0, 1.0] needs to be less than init_tau

    layer_height = 0.04
    max_layers = 75
    min_layers = 0
    background_height = 0.4
    background_color = "#000000"
    output_size = 128
    visualize = False
    stl_output_size = 200
    perform_pruning = True
    pruning_max_colors = 100
    pruning_max_swaps = 100
    pruning_max_layer = 75
    random_seed = 0
    use_depth_anything_height_initialization = False
    depth_strength = 0.25
    depth_threshold = 0.05
    min_cluster_value = 0.1
    w_depth = 0.5
    w_lum = 1.0
    order_blend = 0.1
    mps = False
    run_name = None
    tensorboard = False


def main(input_image, csv_file, warmup, h_start, h_full, tau_init, tau_final):
    # Create config object using default values and override with given hyperparameters.
    args = Config()
    args.input_image = input_image
    args.csv_file = csv_file

    # Set the hyperparameters for the grid search.
    args.warmup_fraction = warmup
    args.height_logits_learning_start_fraction = h_start
    args.height_logits_learning_full_fraction = h_full
    args.init_tau = tau_init
    args.final_tau = tau_final

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

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found.", file=sys.stderr)
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
    material_colors_np, material_TDs_np, material_names, _ = load_materials(
        args.csv_file
    )
    material_colors = torch.tensor(
        material_colors_np, dtype=torch.float32, device=device
    )
    material_TDs = torch.tensor(material_TDs_np, dtype=torch.float32, device=device)

    # Read input image
    img = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)

    alpha = None
    # Check for alpha mask
    if img.shape[2] == 4:
        # Extract the alpha channel
        alpha = img[:, :, 3]
        alpha = alpha[..., None]
        alpha = resize_image(alpha, args.output_size)
        # Convert the image from BGRA to BGR
        img = img[:, :, :3]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create final resolution target image
    output_img_np = resize_image(img, args.output_size)
    output_target = torch.tensor(output_img_np, dtype=torch.float32, device=device)

    pixel_height_logits_init = run_init_threads(
        output_img_np,
        args.max_layers,
        args.layer_height,
        bgr_tuple,
        random_seed=random_seed,
        init_method="kmeans",
        cluster_layers=20,
        lab_space=False,
        num_threads=8,
    )

    # Set initial height for transparent areas if an alpha mask exists
    if alpha is not None:
        pixel_height_logits_init[alpha < 128] = -13.815512

    # VGG Perceptual Loss (disabled in this example)
    perception_loss_module = None

    # Create an optimizer instance
    optimizer = FilamentOptimizer(
        args=args,
        target=output_target,
        pixel_height_logits_init=pixel_height_logits_init,
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
        loss_val = optimizer.step(record_best=i % 10 == 0)

        optimizer.visualize(interval=25)
        optimizer.log_to_tensorboard(interval=100)

        if (i + 1) % 100 == 0:
            tbar.set_description(
                f"Iteration {i + 1}, Loss = {loss_val:.4f}, best validation Loss = {optimizer.best_discrete_loss:.4f}"
            )

    optimizer.prune(
        max_colors_allowed=args.pruning_max_colors,
        max_swaps_allowed=args.pruning_max_swaps,
        min_layers_allowed=args.min_layers,
        max_layers_allowed=args.pruning_max_layer,
    )

    print("Done. Saving outputs...")
    # Save Image
    comp_disc = optimizer.get_best_discretized_image()
    args.max_layers = optimizer.max_layers

    comp_disc = comp_disc.detach()

    # Compute and print the MSE loss between the target and final output
    mse_loss = torch.nn.functional.mse_loss(output_target, comp_disc)
    return mse_loss.item()


def main_suppressed(input_image, csv_file, warmup, h_start, h_full, tau_init, tau_final):
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            result = main(input_image, csv_file, warmup, h_start, h_full, tau_init, tau_final)
    return result


if __name__ == "__main__":
    folder = "../../../images/test_images/"
    csv_file = "../../../bambulab.csv"
    images = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(".jpg")]
    fixed_lr = 1e-1
    parallel_limit = 10

    # Define grid search parameters
    warmup_values = np.linspace(0, 1, 5)  # 5 values for warmup_fraction
    height_values = np.linspace(0, 1, 5)  # 5 values for height fractions
    tau_values = np.linspace(0, 1, 5)     # 5 values for tau

    # Build valid pairs for height fractions (start < full)
    height_pairs = [(hs, hf) for hs in height_values for hf in height_values if hs < hf]
    # Build valid pairs for tau (init_tau > final_tau)
    tau_pairs = [(ti, tf) for ti in tau_values for tf in tau_values if ti > tf]

    # Create grid: warmup x height_pairs x tau_pairs = 5 * 10 * 10 = 500 combinations
    from itertools import product
    hyperparam_grid = list(product(warmup_values, height_pairs, tau_pairs))
    print(f"Total hyperparameter combinations: {len(hyperparam_grid)}")

    out_dict = {}
    # Loop over each hyperparameter combination
    for idx, (warmup, (h_start, h_full), (tau_init, tau_final)) in enumerate(hyperparam_grid):
        try:
            out_dict_str = f"w={warmup}_height_start={h_start}_height_full={h_full}_tau_init={tau_init}_tau_final={tau_final}"
            print(f"Running {idx+1}/{len(hyperparam_grid)}: {out_dict_str}")
            exec = ProcessPoolExecutor(max_workers=parallel_limit)
            tlist = []
            for img in images:
                for i in range(1):
                    tlist.append(
                        exec.submit(
                            main_suppressed,
                            img,
                            csv_file,
                            warmup,
                            h_start,
                            h_full,
                            tau_init,
                            tau_final,
                        )
                    )
            for t in tqdm(concurrent.futures.as_completed(tlist), total=len(tlist)):
                result_list = out_dict.get(out_dict_str, [])
                result_list.append(t.result())
                out_dict[out_dict_str] = result_list

            exec.shutdown()
            # save out_dict as json
            import json

            with open("out_dict.json", "w") as f:
                json.dump(out_dict, f)
        except Exception:
            traceback.print_exc()
