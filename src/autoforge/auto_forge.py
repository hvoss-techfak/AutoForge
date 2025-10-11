import argparse
import sys
import os
import time
import traceback
from typing import Optional

import configargparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

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
from autoforge.Helper.PruningHelper import disc_to_logits

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
        default=0.24,
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
        "--processing_reduction_factor",
        type=int,
        default=2,
        help="Reduction factor for reducing the processing size compared to the output size (default: 2 - half resolution)",
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

    parser.add_argument(
        "--discrete_check",
        type=int,
        default=100,
        help="Modulo how often to check for new discrete results.",
    )

    # New: number of depth-based segments (heightmaps)
    parser.add_argument(
        "--num_heightmaps",
        type=int,
        default=1,
        help="Number of depth-based segments to optimize independently (n=1 keeps current behavior).",
    )

    args = parser.parse_args()
    return args


def _compute_depth_map_norm(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """Return normalized depth map [0,1] using Depth Anything v2 if available; fallback to luminance."""
    try:
        from transformers import pipeline  # lazy import

        pipe = pipeline(
            task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
        )
        from PIL import Image

        depth_result = pipe(Image.fromarray(img_rgb_uint8))
        depth = depth_result["depth"]
        depth = np.array(depth).astype(np.float32)
        dmin, dmax = float(depth.min()), float(depth.max())
        return (depth - dmin) / (dmax - dmin + 1e-8)
    except Exception:
        print(
            "Depth model not available, falling back to luminance-based pseudo-depth."
        )
        # Fallback: normalized luminance (inverted so brighter ~ nearer by default)
        lum = (
            0.299 * img_rgb_uint8[..., 0]
            + 0.587 * img_rgb_uint8[..., 1]
            + 0.114 * img_rgb_uint8[..., 2]
        ).astype(np.float32)
        lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-8)
        return 1.0 - lum


def _segment_depth_kmeans(
    depth_norm: np.ndarray, k: int, seed: int, valid_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Segment depth into k regions with KMeans (or quantiles fallback).
    Returns label map [H,W] with labels 0..k_eff-1 ordered by mean depth asc.
    If valid_mask is provided, only pixels with valid_mask==True are clustered; others get label -1.
    """
    H, W = depth_norm.shape
    labels_full = np.full((H, W), fill_value=-1, dtype=np.int32)

    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=bool)

    flat = depth_norm.reshape(-1, 1)
    flat_valid = flat[valid_mask.reshape(-1)]
    n_valid = flat_valid.shape[0]

    if n_valid == 0 or k <= 0:
        return labels_full  # all invalid or no clusters requested

    k_eff = int(min(k, n_valid))
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k_eff, n_init="auto", random_state=seed)
        labels_valid = km.fit_predict(flat_valid)
    except Exception:
        # Fallback: k-quantiles on valid data
        q = np.quantile(flat_valid, np.linspace(0, 1, k_eff + 1))
        labels_valid = np.digitize(flat_valid, q[1:-1], right=False)

    # Scatter valid labels back into full image
    labels_flat = labels_full.reshape(-1)
    labels_flat[valid_mask.reshape(-1)] = labels_valid
    labels = labels_flat.reshape(H, W)

    # Order labels by mean depth ascending (only over valid pixels)
    means = []
    for i in range(k_eff):
        mask_i = (labels == i) & valid_mask
        means.append(depth_norm[mask_i].mean() if np.any(mask_i) else np.inf)
    order = np.argsort(np.array(means))
    remap = np.full(k_eff, fill_value=-1, dtype=np.int32)
    remap[order] = np.arange(k_eff)

    # Apply remap to valid labels; keep -1 for invalid
    labels_remap = labels.copy()
    valid_pos = labels_remap >= 0
    labels_remap[valid_pos] = remap[labels_remap[valid_pos]]
    return labels_remap


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
    computed_processing_size = int(
        round(computed_output_size / args.processing_reduction_factor)
    )
    print(f"Computed solving pixel size: {computed_output_size}")
    alpha = None
    if img.shape[2] == 4:
        # Extract alpha channel (2D)
        alpha = img[:, :, 3]
        # Resize alpha to output size; handle shape robustly
        alpha_resized = resize_image(alpha[..., None], computed_output_size)
        # Ensure 2D mask after resize
        if alpha_resized.ndim == 3 and alpha_resized.shape[2] == 1:
            alpha_mask_2d = alpha_resized[:, :, 0]
        elif alpha_resized.ndim == 2:
            alpha_mask_2d = alpha_resized
        else:
            alpha_mask_2d = np.squeeze(alpha_resized)
            if alpha_mask_2d.ndim != 2:
                raise ValueError("Alpha mask has unexpected shape after resize")
        # Remove alpha channel from the image
        img = img[:, :, :3]
    else:
        alpha_mask_2d = None

    # Convert image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # For the final resolution
    output_img_np = resize_image(img, computed_output_size)
    output_target = torch.tensor(output_img_np, dtype=torch.float32, device=device)

    # Build depth-based segmentation masks
    H_out, W_out = output_img_np.shape[:2]
    valid_alpha_mask = None if alpha_mask_2d is None else (alpha_mask_2d >= 128)
    if args.num_heightmaps > 1:
        depth_norm = _compute_depth_map_norm(output_img_np.astype(np.uint8))
        depth_labels = _segment_depth_kmeans(
            depth_norm, args.num_heightmaps, random_seed, valid_mask=valid_alpha_mask
        )
        # Collect only non-empty segment masks, always excluding transparent
        present_ids = [i for i in sorted(np.unique(depth_labels)) if i >= 0]
        masks_full = [
            (depth_labels == i)
            & (valid_alpha_mask if valid_alpha_mask is not None else True)
            for i in present_ids
        ]
        # If everything is transparent or no segments created, fall back to a single empty mask
        if len(masks_full) == 0:
            masks_full = []
    else:
        base_mask = np.ones((H_out, W_out), dtype=bool)
        if valid_alpha_mask is not None:
            base_mask &= valid_alpha_mask
        masks_full = [base_mask] if base_mask.any() else []

    if len(masks_full) == 0:
        print(
            "No valid (non-transparent) pixels found after alpha masking. Nothing to optimize."
        )
        # Save an empty outputs bundle and return gracefully
        os.makedirs(args.output_folder, exist_ok=True)
        cv2.imwrite(
            os.path.join(args.output_folder, "final_model.png"),
            np.zeros((H_out, W_out, 3), dtype=np.uint8),
        )
        with open(os.path.join(args.output_folder, "final_loss.txt"), "w") as f:
            f.write("0")
        return 0.0

    # Prepare processing target and masks
    processing_img_np = resize_image(
        output_img_np, computed_processing_size
    )  # For the processing resolution
    processing_target = torch.tensor(
        processing_img_np, dtype=torch.float32, device=device
    )

    masks_proc = [
        cv2.resize(
            mask.astype(np.uint8),
            (processing_target.shape[1], processing_target.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        for mask in masks_full
    ]

    # Precompute future masks (union of nearer segments) at both resolutions
    future_masks_full = []
    future_masks_proc = []
    for i in range(len(masks_full)):
        if i + 1 < len(masks_full):
            fut_full = np.zeros_like(masks_full[i], dtype=bool)
            fut_proc = np.zeros_like(masks_proc[i], dtype=bool)
            for j in range(i + 1, len(masks_full)):
                fut_full |= masks_full[j]
                fut_proc |= masks_proc[j]
        else:
            fut_full = np.zeros_like(masks_full[i], dtype=bool)
            fut_proc = np.zeros_like(masks_proc[i], dtype=bool)
        future_masks_full.append(fut_full)
        future_masks_proc.append(fut_proc)

    # VGG Perceptual Loss
    # We currently disable this as it is not used in the optimization.
    perception_loss_module = None  # MultiLayerVGGPerceptualLoss().to(device).eval()

    # Determine per-segment layer budget (distribute across non-empty segments)
    if len(masks_full) == 1:
        layers_per_segment = [args.max_layers]
    else:
        base = args.max_layers // len(masks_full)
        rem = args.max_layers % len(masks_full)
        layers_per_segment = [
            base + (1 if i < rem else 0) for i in range(len(masks_full))
        ]

    # Accumulators for final outputs (full resolution)
    final_comp_full = np.zeros((H_out, W_out, 3), dtype=np.uint8)
    # Running composite of previous segments (RGB)
    background_rgb_uint8 = (np.array(bgr_tuple) * 255.0).astype(np.uint8)
    prev_comp_rgb_full = np.tile(background_rgb_uint8, (H_out, W_out, 1))

    # Collect per-segment discrete results
    segment_results = []  # dicts with keys: mask_full, disc_height (H,W int), disc_global (L,)

    # Iterate segments in ascending depth (0..k-1)
    for seg_idx, (mask_full, mask_proc, seg_layers) in enumerate(
        zip(masks_full, masks_proc, layers_per_segment)
    ):
        # Skip empty or zero-layer segments
        if seg_layers <= 0:
            print(f"Skipping segment {seg_idx + 1}: zero layer budget")
            continue
        if not mask_proc.any() or not mask_full.any():
            print(f"Skipping empty segment {seg_idx + 1}")
            continue

        print(
            f"Optimizing segment {seg_idx + 1}/{len(masks_full)} with {seg_layers} layers"
        )

        # Independent initialization per segment at full resolution
        # Mask the image to the segment to avoid cross-contamination during init
        img_seg_full = output_img_np.copy()
        # Fill outside with appropriate background:
        if seg_idx == 0:
            # First segment: true background
            bg_uint8 = (np.array(bgr_tuple) * 255.0).astype(img_seg_full.dtype)
            img_seg_full[~mask_full] = bg_uint8
        else:
            # Higher segments: use previously composited color
            img_seg_full[~mask_full] = prev_comp_rgb_full[~mask_full]

        print("Initializing segment height map (per-segment init)...")
        (
            seg_pixel_height_logits_full,
            seg_global_logits_init,
            seg_pixel_height_labels_full,
        ) = run_init_threads(
            img_seg_full,
            int(seg_layers),
            args.layer_height,
            bgr_tuple,
            random_seed=random_seed,
            num_threads=args.num_init_rounds,
            init_method="kmeans",
            cluster_layers=args.num_init_cluster_layers,
            material_colors=material_colors_np,
        )

        # Resize per-segment initializations to processing resolution
        seg_proc_logits_init = cv2.resize(
            src=seg_pixel_height_logits_full,
            interpolation=cv2.INTER_NEAREST,
            dsize=(processing_target.shape[1], processing_target.shape[0]),
        )
        seg_proc_labels = cv2.resize(
            src=seg_pixel_height_labels_full,
            interpolation=cv2.INTER_NEAREST,
            dsize=(processing_target.shape[1], processing_target.shape[0]),
        )
        # Constrain logits outside segment:
        # - For future segments: force to max height (opaque)
        # - For all other outside pixels: force to min height (absent)
        fut_proc = future_masks_proc[seg_idx]
        outside_proc = ~mask_proc
        max_logit = 13.815512  # ~ sigmoid(13.8) ~ 0.999999
        min_logit = -13.815512
        seg_proc_logits_init[outside_proc & fut_proc] = max_logit
        seg_proc_logits_init[outside_proc & (~fut_proc)] = min_logit
        # Keep labels zero outside
        seg_proc_labels[outside_proc] = 0

        # Create a copy of args with per-segment max_layers
        seg_args = argparse.Namespace(**vars(args))
        seg_args.max_layers = int(seg_layers)

        # Create optimizer with focus map
        focus_map = torch.tensor(mask_proc.astype(np.float32), device=device)
        optimizer = FilamentOptimizer(
            args=seg_args,
            target=processing_target,
            pixel_height_logits_init=seg_proc_logits_init,
            pixel_height_labels=seg_proc_labels,
            global_logits_init=seg_global_logits_init,
            material_colors=torch.tensor(
                material_colors_np, dtype=torch.float32, device=device
            ),
            material_TDs=torch.tensor(
                material_TDs_np, dtype=torch.float32, device=device
            ),
            background=background,
            device=device,
            perception_loss_module=perception_loss_module,
            focus_map=focus_map,
        )

        # Main optimization loop per segment
        print("Starting optimization for segment...")
        tbar = tqdm(range(seg_args.iterations))
        dtype = torch.bfloat16 if not seg_args.mps else torch.float32
        with torch.autocast(device.type, dtype=dtype):
            for i in tbar:
                loss_val = optimizer.step(record_best=i % seg_args.discrete_check == 0)

                optimizer.visualize(interval=100)
                optimizer.log_to_tensorboard(interval=100)

                if (i + 1) % 100 == 0:
                    tbar.set_description(
                        f"Seg {seg_idx + 1}: Iter {i + 1}, Loss={loss_val:.4f}, best={optimizer.best_discrete_loss:.4f}, lr={optimizer.current_learning_rate:.6f}"
                    )
                if (
                    optimizer.best_step is not None
                    and optimizer.num_steps_done - optimizer.best_step
                    > seg_args.early_stopping
                ):
                    print(
                        "Early stopping after",
                        seg_args.early_stopping,
                        "steps without improvement (segment).",
                    )
                    break

        # Post-opt logging namespace per segment
        post_opt_step = 0
        optimizer.log_to_tensorboard(
            interval=1,
            namespace=f"post_opt/seg{seg_idx}",
            step=(post_opt_step := post_opt_step + 1),
        )

        # Upsample best logits to full resolution and apply mask rules for future pixels
        with torch.no_grad():
            best_logits_proc = (
                optimizer.best_params["pixel_height_logits"].detach().cpu().numpy()
            )
            best_logits_full = cv2.resize(
                best_logits_proc,
                (W_out, H_out),
                interpolation=cv2.INTER_LINEAR,
            )
            # Outside current mask and not future: min height; future: max height
            fut_full = future_masks_full[seg_idx]
            outside_full = ~mask_full
            best_logits_full[outside_full & fut_full] = max_logit
            best_logits_full[outside_full & (~fut_full)] = min_logit

        # Switch to full-size for discrete solution, applying full target
        optimizer.pixel_height_logits = torch.from_numpy(best_logits_full).to(device)
        optimizer.best_params["pixel_height_logits"] = torch.from_numpy(
            best_logits_full
        ).to(device)
        optimizer.target = output_target
        full_labels = torch.tensor(
            seg_pixel_height_labels_full, dtype=torch.int32, device=device
        )
        mask_t = torch.from_numpy(mask_full).to(device)
        full_labels[~mask_t] = 0
        optimizer.pixel_height_labels = full_labels

        with torch.no_grad():
            disc_global_seg, disc_height_image_full = (
                optimizer.get_discretized_solution(best=True)
            )

        # Store per-segment result for joint pruning
        seg_heights_np = disc_height_image_full.cpu().numpy().astype(np.int32)
        segment_results.append(
            {
                "mask_full": mask_full,
                "disc_height": seg_heights_np,
                "disc_global": disc_global_seg.cpu().numpy(),
            }
        )

        # Compose segment color image (preview only)
        comp_disc_seg = optimizer.get_best_discretized_image()
        # RGB composite for stacking logic
        comp_seg_rgb = np.clip(comp_disc_seg.cpu().numpy().astype(np.uint8), 0, 255)
        # Update running composite: place this segment on top where it prints anything
        seg_mask_any = seg_heights_np > 0
        prev_comp_rgb_full[seg_mask_any] = comp_seg_rgb[seg_mask_any]
        # BGR preview for optional saving/inspection (kept for parity)
        comp_seg_np = cv2.cvtColor(comp_seg_rgb, cv2.COLOR_RGB2BGR)
        final_comp_full[seg_mask_any] = comp_seg_np[seg_mask_any]

        # Free VRAM between segments
        torch.cuda.empty_cache()

    # Joint pruning across all segments
    print("Starting joint pruning across all segments...")
    if len(segment_results) == 0:
        print("No segments to process.")
        return 0.0

    # Build merged discrete globals and total discrete heights (stacking in far->near order)
    merged_disc_global = None
    total_disc_height = np.zeros((H_out, W_out), dtype=np.int32)
    for seg in segment_results:
        total_disc_height += seg["disc_height"]
        dg = seg["disc_global"]
        merged_disc_global = (
            dg
            if merged_disc_global is None
            else np.concatenate([merged_disc_global, dg], axis=0)
        )

    L_total = int(merged_disc_global.shape[0])

    # Create merged pixel_height_logits to match total_disc_height exactly
    eps = 1e-6
    denom = max(L_total, 1)
    ratio = np.clip(total_disc_height.astype(np.float32) / denom, eps, 1.0 - eps)
    merged_logits_full = np.log(ratio) - np.log(1.0 - ratio)

    # Build merged global logits from discrete ids
    num_materials = material_colors.shape[0]
    merged_global_logits = (
        disc_to_logits(
            torch.tensor(merged_disc_global, dtype=torch.int64),
            num_materials=int(num_materials),
        )
        .cpu()
        .numpy()
    )

    # Create a pruning-only optimizer at full resolution
    prune_args = argparse.Namespace(**vars(args))
    prune_args.max_layers = L_total
    merged_optimizer = FilamentOptimizer(
        args=prune_args,
        target=output_target,
        pixel_height_logits_init=merged_logits_full,
        pixel_height_labels=np.zeros_like(total_disc_height, dtype=np.int32),
        global_logits_init=merged_global_logits,
        material_colors=material_colors,
        material_TDs=material_TDs,
        background=background,
        device=device,
        perception_loss_module=None,
        focus_map=None,
    )
    # Seed best params for pruning
    merged_optimizer.best_params = merged_optimizer.get_current_parameters()
    merged_optimizer.best_seed = (
        0  # ensure rng_seed is an int before rendering best image
    )
    with torch.no_grad():
        out_img = merged_optimizer.get_best_discretized_image()
        from autoforge.Loss.LossFunctions import compute_loss as _compute_loss

        merged_optimizer.best_discrete_loss = _compute_loss(
            comp=out_img, target=merged_optimizer.target
        ).item()
        # best_seed already set above

    # Run pruning ONCE across all layers
    if args.perform_pruning:
        with torch.autocast(
            device.type, dtype=(torch.bfloat16 if not args.mps else torch.float32)
        ):
            merged_optimizer.prune(
                max_colors_allowed=args.pruning_max_colors,
                max_swaps_allowed=args.pruning_max_swaps,
                min_layers_allowed=args.min_layers,
                max_layers_allowed=args.pruning_max_layer,
                search_seed=True,
                fast_pruning=args.fast_pruning,
                fast_pruning_percent=args.fast_pruning_percent,
            )

    # Final discrete solution after joint pruning
    with torch.no_grad():
        combined_disc_global_t, combined_disc_height_t = (
            merged_optimizer.get_discretized_solution(best=True)
        )
    combined_disc_global = combined_disc_global_t.cpu().numpy()
    combined_disc_height_full = combined_disc_height_t.cpu().numpy().astype(np.int32)

    # Final composite image from merged solution
    comp_final = merged_optimizer.get_best_discretized_image()
    final_comp_full = cv2.cvtColor(
        np.clip(comp_final.cpu().numpy().astype(np.uint8), 0, 255), cv2.COLOR_RGB2BGR
    )

    # Save outputs
    print("Done. Saving outputs...")
    cv2.imwrite(os.path.join(args.output_folder, "final_model.png"), final_comp_full)

    final_loss = float(len(combined_disc_global))
    with open(os.path.join(args.output_folder, "final_loss.txt"), "w") as f:
        f.write(f"{final_loss}")

    # STL from total layers map
    stl_filename = os.path.join(args.output_folder, "final_model.stl")
    height_map_mm = (combined_disc_height_full.astype(np.float32)) * args.layer_height
    generate_stl(
        height_map_mm,
        stl_filename,
        args.background_height,
        maximum_x_y_size=args.stl_output_size,
        alpha_mask=alpha_mask_2d,
    )

    # Swap instructions and project file use combined arrays
    background_layers = int(args.background_height // args.layer_height)
    swap_instructions = generate_swap_instructions(
        combined_disc_global,
        combined_disc_height_full,
        args.layer_height,
        background_layers,
        args.background_height,
        material_names,
    )
    with open(os.path.join(args.output_folder, "swap_instructions.txt"), "w") as f:
        for line in swap_instructions:
            f.write(line + "\n")

    project_filename = os.path.join(args.output_folder, "project_file.hfp")
    generate_project_file(
        project_filename,
        args,
        combined_disc_global,
        combined_disc_height_full,
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

        # print statistics
        # median
        losses = [x[1] for x in ret]
        median_loss = np.median(losses)
        std_loss = np.std(losses)
        print(f"Best run folder: {best_run_folder}")
        print(f"Best run loss: {best_loss}")
        print(f"Median loss: {median_loss}")
        print(f"Standard deviation of losses: {std_loss}")

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
