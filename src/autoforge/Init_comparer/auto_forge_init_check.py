import concurrent
import sys
import os
import time
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
    learning_rate = 1e-2
    layer_height = 0.04
    max_layers = 75
    min_layers = 0
    background_height = 0.4
    background_color = "#000000"
    output_size = 128
    init_tau = 1.0
    final_tau = 0.01
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


def main(input_image, csv_file, init_method, cluster_layers, lab_space):
    # Create config object using default values
    args = Config()
    args.input_image = input_image
    args.csv_file = csv_file

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
        init_method=init_method,
        cluster_layers=cluster_layers,
        lab_space=lab_space,
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


def main_suppressed(input_image, csv_file, init_method, cluster_layers, lab_space):
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            result = main(input_image, csv_file, init_method, cluster_layers, lab_space)
    return result


if __name__ == "__main__":
    folder = "../../../images/test_images/"
    csv_file = "../../../bambulab.csv"
    images = [folder + "/" + img for img in os.listdir(folder) if img.endswith(".jpg")]
    parallel_limit = 15
    methods = [
        "kmeans",
        "quantize_median",
        "quantize_maxcoverage",
        "quantize_fastoctree",
    ]
    max_layers = 75
    cluster_layers = [
        max_layers // 4,
        max_layers // 2,
        max_layers,
        max_layers * 2,
        max_layers * 4,
    ]
    use_lab_space = [True, False]

    # test every permutation using itertools
    from itertools import product

    out_dict = {}
    do_list = list(product(methods, cluster_layers, use_lab_space))
    shuffle(do_list)

    for i, (method, cluster, lab) in enumerate(do_list):
        out_dict_str = f"{method}_{cluster}_{lab}"
        print(
            f"Running {method} with {cluster} clusters and lab={lab}, {i + 1}/{len(do_list)}"
        )
        exec = ProcessPoolExecutor(max_workers=parallel_limit)
        tlist = []
        for img in images:
            for i in range(1):
                tlist.append(
                    exec.submit(
                        main_suppressed,
                        img,
                        csv_file,
                        method,
                        cluster,
                        lab,
                    )
                )
        for t in tqdm(concurrent.futures.as_completed(tlist), total=len(tlist)):
            result_list = out_dict.get(out_dict_str, [])
            result_list.append(t.result())
            out_dict[out_dict_str] = result_list
        # save out_dict as json
        import json

        with open("out_dict.json", "w") as f:
            json.dump(out_dict, f)
