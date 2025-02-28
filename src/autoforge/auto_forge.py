#!/usr/bin/env python
"""
Script for generating 3D printed layered models from an input image.

This script uses a learned optimization with a Gumbel softmax formulation
to assign materials per layer and produce both a discretized composite that
is exported as an STL file along with swap instructions.
"""
import argparse
import math
import os
from typing import Optional, List, Tuple, Dict

import configargparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
# --------- PyTorch ------------
import torch
from tqdm import tqdm

#set precision if available
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
import torch
from torch import optim
import torch.functional as F
#torch.autograd.set_detect_anomaly(True)

from autoforge.helper_functions import (
    hex_to_rgb, load_materials, generate_stl, generate_swap_instructions,
    generate_project_file, init_height_map, resize_image, merge_color, merge_bands,
    find_color_bands, MultiLayerVGGPerceptualLoss
)

def compute_mse(dg_candidate: torch.Tensor,
                pixel_height_logits: torch.Tensor,
                target: torch.Tensor,
                h: float,
                max_layers: int,
                material_colors: torch.Tensor,
                material_TDs: torch.Tensor,
                background: torch.Tensor,
                tau_for_comp: float = 1e-3) -> float:
    """
    Return the Mean Squared Error (MSE) to the target.

    Args:
        dg_candidate (torch.Tensor): Discrete global material assignment candidate.
        pixel_height_logits (torch.Tensor): Logits for pixel heights.
        target (torch.Tensor): Target image tensor.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (torch.Tensor): Tensor of material colors.
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters.
        background (torch.Tensor): Background color tensor.
        tau_for_comp (float, optional): Temperature for compositing. Defaults to 1e-3.

    Returns:
        float: The computed Mean Squared Error (MSE) value.
    """
    with torch.no_grad():
        comp = composite_image_combined_jit(
            pixel_height_logits, dg_candidate,
            tau_for_comp, tau_for_comp,
            h, max_layers, material_colors, material_TDs, background,
            mode="pruning"  # or "discrete"
        )
    mse_val = torch.mean((comp - target)**2).item()
    return mse_val

@torch.jit.script
def adaptive_round(x: torch.Tensor, tau: float, high_tau: float, low_tau: float, temp: float) -> torch.Tensor:
    """
    Smooth rounding based on temperature 'tau'.
    """
    if tau <= low_tau:
        return torch.round(x)
    elif tau >= high_tau:
        floor_val = torch.floor(x)
        diff = x - floor_val
        soft_round = floor_val + torch.sigmoid((diff - 0.5) / temp)
        return soft_round
    else:
        ratio = (tau - low_tau) / (high_tau - low_tau)
        hard_round = torch.round(x)
        floor_val = torch.floor(x)
        diff = x - floor_val
        soft_round = floor_val + torch.sigmoid((diff - 0.5) / temp)
        return ratio * soft_round + (1 - ratio) * hard_round


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """
    Compute the Huber loss between predictions and targets.
    """
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic**2 + delta * linear)


import torch
import torch.nn.functional as F


@torch.jit.script
def composite_image_combined_fast(
    pixel_height_logits: torch.Tensor,
    global_logits:     torch.Tensor,
    tau_height:        float,
    tau_global:        float,
    h:                 float,
    max_layers:        int,
    material_colors:   torch.Tensor,
    material_TDs:      torch.Tensor,
    background:        torch.Tensor,
    mode:              str = "continuous"
) -> torch.Tensor:
    """
    Vectorized compositing over all pixels (H x W).

    - pixel_height_logits: shape [H, W].
    - global_logits: shape [max_layers, num_materials] (or a 1D tensor if mode=="pruning").
    - tau_height: temperature for converting pixel_height_logits to discrete layer counts.
    - tau_global: temperature for material selection.
    - h: layer thickness.
    - max_layers: total number of layers.
    - material_colors: shape [num_materials, 3].
    - material_TDs: shape [num_materials].
    - background: shape [3].
    - mode: "continuous", "discrete", or "pruning".

    Returns:
        Tensor of shape [H, W, 3] with pixel values in [0,255].
    """
    # Compute continuous pixel height and convert to discrete layer counts.
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logits)
    continuous_layers = pixel_height / h
    adaptive_layers = adaptive_round(continuous_layers, tau_height, high_tau=0.1, low_tau=0.01, temp=0.1)
    discrete_layers_temp = torch.round(continuous_layers)
    discrete_layers = discrete_layers_temp + (adaptive_layers - discrete_layers_temp).detach()
    discrete_layers = discrete_layers.to(torch.int32)  # shape [H, W]

    if mode == "pruning":
        num_materials = material_colors.size(0)
        global_logits_oh = F.one_hot(global_logits.long(), num_classes=num_materials).float()
    else:
        global_logits_oh = global_logits

    H, W = pixel_height.shape
    comp = torch.zeros(H, W, 3, dtype=torch.float32, device=pixel_height.device)
    remaining = torch.ones(H, W, dtype=torch.float32, device=pixel_height.device)

    A = 0.1215
    k = 61.6970
    b = 0.4773

    for i in range(max_layers):
        layer_idx = max_layers - 1 - i
        p_print = (discrete_layers > layer_idx).float()  # shape [H, W]
        eff_thick = p_print * h  # shape [H, W]

        if mode == "pruning":
            p_i = global_logits_oh[layer_idx]  # shape [num_materials]
        elif mode == "discrete":
            p_i = F.gumbel_softmax(global_logits[layer_idx], tau_global, hard=True)
        elif mode == "continuous":
            if tau_global < 1e-3:
                p_i = F.gumbel_softmax(global_logits[layer_idx], tau_global, hard=True)
            else:
                p_i = F.gumbel_softmax(global_logits[layer_idx], tau_global, hard=False)
        else:
            p_i = global_logits[layer_idx]

        color_i = torch.matmul(p_i, material_colors)  # shape [3]
        TD_i = torch.matmul(p_i, material_TDs) * 0.1
        TD_i = torch.clamp(TD_i, 1e-8, 1e8)
        opac = A * torch.log1p(k * (eff_thick / TD_i)) + b * (eff_thick / TD_i)
        opac = torch.clamp(opac, 0.0, 1.0)

        # Use out-of-place assignments instead of in-place updates.
        comp = comp + ((remaining * opac).unsqueeze(-1) * color_i)
        remaining = remaining * (1 - opac)

    comp = comp + remaining.unsqueeze(-1) * background
    return comp * 255.0

composite_image_combined_jit = composite_image_combined_fast

######################################################################
# LOSS FUNCTION
######################################################################

def compute_loss(
    material_assignment: torch.Tensor,
    comp: torch.Tensor,
    target: torch.Tensor,
    perception_loss_module: torch.nn.Module,
    tau_global: float,
    num_materials: int,
    add_penalty_loss: bool = True
) -> torch.Tensor:
    """
    Compute the overall loss, combining:
      - Perceptual loss
      - MSE loss
      - Color change penalty
      - Few-colors penalty

    Handles both continuous (max_layers x num_materials) and discrete (max_layers,) assignments.
      - If material_assignment.dim() == 2 --> 'continuous' mode
      - If material_assignment.dim() == 1 --> 'discrete' mode

    Args:
        material_assignment (torch.Tensor): Tensor representing the material assignment.
        comp (torch.Tensor): Composite image tensor.
        target (torch.Tensor): Target image tensor.
        perception_loss_module (torch.nn.Module): Module to compute perceptual loss.
        tau_global (float): Temperature parameter for Gumbel-Softmax.
        num_materials (int): Number of materials.
        add_penalty_loss (bool, optional): Whether to add penalty loss. Defaults to True.

    Returns:
        torch.Tensor: The computed total loss.
    """

    # Compute the standard MSE & Perceptual loss on the composite
    mse_loss = F.mse_loss(comp, target) / 1000.0

    # Perceptual loss
    # Expecting comp and target in shape [H, W, 3] => transform to [B=1, C=3, H, W]
    comp_batch   = comp.permute(2, 0, 1).unsqueeze(0)
    target_batch = target.permute(2, 0, 1).unsqueeze(0)
    perception_loss = perception_loss_module(comp_batch, target_batch)

    #Compute color-change penalty and few-colors penalty
    if material_assignment.dim() == 2:
        # ============ Continuous case: (max_layers x num_materials) ============
        # "p" is shape (max_layers, num_materials)
        p = F.softmax(material_assignment, dim=1)  # continuous "prob" for each layer

        # Color-change penalty
        # dot_products[i] = sum(p[i] * p[i+1])
        # => 1 if same color, < 1 if there's a "transition"
        dot_products = torch.sum(p[:-1] * p[1:], dim=1)  # shape (max_layers-1,)
        color_change_penalty = torch.mean(1.0 - dot_products)

        # Few-colors penalty
        # average usage of each color across layers
        color_usage = torch.mean(p, dim=0)  # shape (num_materials,)
        few_colors_penalty = torch.sum(torch.sqrt(1e-8 + color_usage))

    else:
        # ============ Discrete case: (max_layers,) ============
        disc = material_assignment  # shape (max_layers,) of integer color-IDs

        # Color-change penalty:
        # For discrete, define dot_products[i] = 1 if disc[i] == disc[i+1], else 0
        same_color = (disc[:-1] == disc[1:]).float()
        # so dot_products == 1 where there's no color change
        # => penalty = mean(1 - dot_products) = fraction of boundaries that do change color
        dot_products = same_color
        color_change_penalty = torch.mean(1.0 - dot_products)  if add_penalty_loss else 0.0

        # Few-colors penalty:
        # usage_j = fraction of layers that are assigned color j
        max_layers = disc.shape[0]
        usage_counts = torch.bincount(disc, minlength=num_materials).float()
        color_usage = usage_counts / float(max_layers)
        few_colors_penalty = torch.sum(torch.sqrt(1e-8 + color_usage)) if add_penalty_loss else 0.0

    lambda_swap = (1.0 - tau_global)*0.1

    total_loss = (
        mse_loss
        + lambda_swap * color_change_penalty
        + lambda_swap * few_colors_penalty
    )
    return total_loss

def loss_fn(
    params: Dict[str, torch.Tensor],
    target: torch.Tensor,
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,
    material_TDs: torch.Tensor,
    background: torch.Tensor,
    perception_loss_module: torch.nn.Module
) -> torch.Tensor:
    """
    Full forward pass for continuous assignment:
    composite, then compute unified loss on (global_logits).

    Args:
        params (Dict[str, torch.Tensor]): Dictionary containing the parameters 'pixel_height_logits' and 'global_logits'.
        target (torch.Tensor): Target image tensor.
        tau_height (float): Temperature parameter for height compositing.
        tau_global (float): Temperature parameter for global material selection.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (torch.Tensor): Tensor of material colors.
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters.
        background (torch.Tensor): Background color tensor.
        perception_loss_module (torch.nn.Module): Module to compute perceptual loss.

    Returns:
        torch.Tensor: The computed total loss.
    """
    # Step 1: Composite with continuous "global_logits"
    comp = composite_image_combined_jit(
        params['pixel_height_logits'],
        params['global_logits'],
        tau_height,
        tau_global,
        h,
        max_layers,
        material_colors,
        material_TDs,
        background,
        mode="continuous"
    )
    # Step 2: Single unified loss
    global_logits = params["global_logits"]  # shape (max_layers, num_materials)
    return compute_loss(
        material_assignment=global_logits,
        comp=comp,
        target=target,
        perception_loss_module=perception_loss_module,
        tau_global=tau_global,
        num_materials=material_colors.shape[0],
    )


######################################################################
# OPTIMIZATION
######################################################################

def create_update_step(optimizer: torch.optim.Optimizer, loss_function: callable, h: float, max_layers: int,
                       material_colors: torch.Tensor, material_TDs: torch.Tensor, background: torch.Tensor) -> callable:
    """
    Create a PyTorch update step function using the specified loss function.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to use for the update step.
        loss_function (callable): The loss function to compute the loss.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (torch.Tensor): Tensor of material colors.
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters.
        background (torch.Tensor): Background color tensor.

    Returns:
        callable: A function that performs a single update step.
    """
    def update_step(params, target, tau_height, tau_global, perception_loss_module):
        """
        Perform a single update step.

        Args:
            params (dict): Dictionary containing the parameters 'pixel_height_logits' and 'global_logits'.
            target (torch.Tensor): Target image tensor.
            tau_height (float): Temperature parameter for height compositing.
            tau_global (float): Temperature parameter for global material selection.
            perception_loss_module (torch.nn.Module): Module to compute perceptual loss.

        Returns:
            tuple: Updated parameters and the computed loss value.
        """
        optimizer.zero_grad()
        loss_val = loss_function(
            params, target, tau_height, tau_global,
            h, max_layers, material_colors, material_TDs, background, perception_loss_module
        )
        loss_val.backward()
        optimizer.step()
        return params, loss_val

    return update_step


def discretize_solution_jax(params: Dict[str, torch.Tensor], tau_global: float, h: float, max_layers: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Discretize the continuous pixel height logits into integer layer counts,
    and force hard material selections.

    Args:
        params (Dict[str, torch.Tensor]): Dictionary containing 'pixel_height_logits' and 'global_logits'.
        tau_global (float): Temperature parameter for Gumbel-Softmax.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - disc_global (torch.Tensor): Discretized global material assignments.
            - disc_height_image (torch.Tensor): Discretized height image.
    """
    pixel_height_logits = params['pixel_height_logits']
    global_logits = params['global_logits']

    # Pixel heights
    pixel_heights = (max_layers * h) * torch.sigmoid(pixel_height_logits)
    discrete_height_image = torch.round(pixel_heights / h).to(torch.int32)
    discrete_height_image = torch.clamp(discrete_height_image, 0, max_layers)

    # For each layer, do a hard gumbel softmax (or you can do argmax if you prefer)
    num_layers = global_logits.shape[0]
    discrete_global_vals = []
    for j in range(num_layers):
        p = F.gumbel_softmax(global_logits[j], tau_global, hard=True)
        discrete_global_vals.append(torch.argmax(p))
    discrete_global = torch.stack(discrete_global_vals, dim=0)
    return discrete_global, discrete_height_image
######################################################################
# MAIN OPTIMIZATION LOOP
######################################################################

def run_optimizer(
        target: torch.Tensor, pixel_height_logits: torch.Tensor, H: int, W: int, max_layers: int, h: float,
        material_colors: torch.Tensor, material_TDs: torch.Tensor, background: torch.Tensor,
        num_iters: int, learning_rate: float, decay_v: float, loss_function: callable, visualize: bool = False,
        output_folder: Optional[str] = None, save_interval_pct: Optional[float] = None,
        img_width: Optional[int] = None, img_height: Optional[int] = None, background_height: Optional[float] = None,
        material_names: Optional[List[str]] = None, csv_file: Optional[str] = None, args: Optional[argparse.Namespace] = None,
        perception_loss_module: Optional[torch.nn.Module] = None

) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Run the optimization loop to learn per-pixel heights and per-layer material assignments.

    Args:
        target (torch.Tensor): Target image tensor.
        pixel_height_logits (torch.Tensor): Logits for pixel heights.
        H (int): Height of the image.
        W (int): Width of the image.
        max_layers (int): Maximum number of layers.
        h (float): Layer thickness.
        material_colors (torch.Tensor): Tensor of material colors.
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters.
        background (torch.Tensor): Background color tensor.
        num_iters (int): Number of optimization iterations.
        learning_rate (float): Learning rate for optimization.
        decay_v (float): Final tau value for Gumbel-Softmax.
        loss_function (callable): The loss function to compute the loss.
        visualize (bool, optional): Enable visualization during optimization. Defaults to False.
        output_folder (Optional[str], optional): Folder to write outputs. Defaults to None.
        save_interval_pct (Optional[float], optional): Percentage interval to save checkpoints. Defaults to None.
        img_width (Optional[int], optional): Width of the image. Defaults to None.
        img_height (Optional[int], optional): Height of the image. Defaults to None.
        background_height (Optional[float], optional): Height of the background. Defaults to None.
        material_names (Optional[List[str]], optional): List of material names. Defaults to None.
        csv_file (Optional[str], optional): Path to CSV file with material data. Defaults to None.
        args (Optional[argparse.Namespace], optional): Additional arguments. Defaults to None.
        perception_loss_module (Optional[torch.nn.Module], optional): Module to compute perceptual loss. Defaults to None.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing the best parameters and the best composite image.
    """
    # Convert to Torch Tensors if needed
    if not isinstance(pixel_height_logits, torch.Tensor):
        pixel_height_logits = torch.tensor(pixel_height_logits, dtype=torch.float32, device=args.device)
    pixel_height_logits.requires_grad = True

    num_materials = material_colors.shape[0]
    # Initialize global_logits
    global_logits_init = torch.ones((max_layers, num_materials), dtype=torch.float32, device=args.device) * -1.0
    for i in range(max_layers):
        global_logits_init[i, i % num_materials] = 1.0
    # Add a small random offset
    global_logits_init += (torch.rand_like(global_logits_init) * 0.2 - 0.1)
    global_logits_init.requires_grad = True

    params = {
        'global_logits': global_logits_init,
        'pixel_height_logits': pixel_height_logits
    }

    # Adam optimizer
    optimizer = optim.Adam([params['global_logits'], params['pixel_height_logits']], lr=learning_rate)
    update_step = create_update_step(optimizer, loss_function, h, max_layers, material_colors, material_TDs, background)

    # Tau decay schedule
    warmup_steps = num_iters // 4
    decay_rate = -math.log(decay_v) / (num_iters - warmup_steps)

    def get_tau(i, tau_init=1.0, tau_final=decay_v, decay_rate=decay_rate):
        """
        Compute the tau value for the current iteration.

        Args:
            i (int): Current iteration.
            tau_init (float): Initial tau value.
            tau_final (float): Final tau value.
            decay_rate (float): Decay rate for tau.

        Returns:
            float: The computed tau value.
        """
        if i < warmup_steps:
            return tau_init
        else:
            return max(tau_final, tau_init * math.exp(-decay_rate * (i - warmup_steps)))

    best_params = None
    best_loss = float('inf')
    best_params_since_last_save = None
    best_loss_since_last_save = float('inf')
    checkpoint_interval = None
    if save_interval_pct is not None:
        checkpoint_interval = int(num_iters * save_interval_pct / 100)

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        target_im_ax   = ax[0].imshow(np.array(target.detach().cpu(), dtype=np.uint8))
        ax[0].set_title("Target Image")
        current_comp_ax= ax[1].imshow(np.zeros((H, W, 3), dtype=np.uint8))
        ax[1].set_title("Current Composite")
        plt.pause(0.1)

    tbar = tqdm(range(num_iters))
    for i in tbar:
        tau_height = get_tau(i)
        tau_global = get_tau(i)

        params, loss_val = update_step(params, target, tau_height, tau_global, perception_loss_module)

        if i > args.best_loss_iterations * num_iters:
            # Check if best
            if loss_val.item() < best_loss_since_last_save:
                best_loss_since_last_save = loss_val.item()
                best_params_since_last_save = {
                    k: v.clone().detach() for k, v in params.items()
                }

            if loss_val.item() < best_loss or best_params is None:
                best_loss = loss_val.item()
                best_params = {
                    k: v.clone().detach() for k, v in params.items()
                }

        if visualize and (i % 25 == 0 or i == num_iters - 1):
            with torch.no_grad():
                comp = composite_image_combined_jit(
                    params['pixel_height_logits'], params['global_logits'],
                    tau_height, tau_global, h, max_layers,
                    material_colors, material_TDs, background, mode="continuous"
                )
            comp_np = np.clip(comp.cpu().numpy(), 0, 255).astype(np.uint8)
            current_comp_ax.set_data(comp_np)

            fig.suptitle(f"Iteration {i}, Loss: {loss_val.item():.4f}, Best Loss: {best_loss:.4f}, Tau: {tau_global:.4f}")
            plt.pause(0.01)

        if checkpoint_interval is not None and (i + 1) % checkpoint_interval == 0 and i > 10:
            print("Saving intermediate outputs (checkpoint).")
            save_intermediate_outputs(
                i, best_params_since_last_save, tau_global,
                h, max_layers, material_colors, material_TDs, background,
                output_folder, W, H, background_height, material_names, csv_file, args=args
            )
            best_params_since_last_save = None
            best_loss_since_last_save = float('inf')
        tbar.set_description(f"loss = {loss_val.item():.4f}, Best Loss = {best_loss:.4f}")

    if visualize:
        plt.ioff()
        plt.close()

    # Return best
    with torch.no_grad():
        best_comp = composite_image_combined_jit(
            best_params['pixel_height_logits'], best_params['global_logits'],
            tau_height, tau_global,
            h, max_layers, material_colors, material_TDs, background, mode="continuous"
        )
    return best_params, best_comp


def save_intermediate_outputs(
        iteration: int, params: Dict[str, torch.Tensor], tau_global: float,
        h: float, max_layers: int, material_colors: torch.Tensor, material_TDs: torch.Tensor, background: torch.Tensor,
        output_folder: str, img_width: int, img_height: int, background_height: float,
        material_names: List[str], csv_file: str, args: argparse.Namespace
) -> None:
    """
    Save intermediate outputs during the optimization process.

    Args:
        iteration (int): Current iteration number.
        params (Dict[str, torch.Tensor]): Dictionary containing 'pixel_height_logits' and 'global_logits'.
        tau_global (float): Temperature parameter for Gumbel-Softmax.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (torch.Tensor): Tensor of material colors.
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters.
        background (torch.Tensor): Background color tensor.
        output_folder (str): Folder to write outputs.
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        background_height (float): Height of the background.
        material_names (List[str]): List of material names.
        csv_file (str): Path to CSV file with material data.
        args (argparse.Namespace): Additional arguments.
    """
    import cv2
    import numpy as np

    # Compute discrete composite image
    disc_comp = composite_image_combined_jit(
        params['pixel_height_logits'], params['global_logits'],
        tau_global, tau_global,
        h, max_layers, material_colors, material_TDs, background, mode="discrete"
    )
    discrete_comp_np = np.clip(disc_comp.detach().cpu().numpy(), 0, 255).astype(np.uint8)
    image_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_comp.jpg")
    cv2.imwrite(image_filename, cv2.cvtColor(discrete_comp_np, cv2.COLOR_RGB2BGR))

    # Discretize
    disc_global, disc_height_image = discretize_solution_jax(params, tau_global, h, max_layers)

    # Generate STL
    height_map_mm = (disc_height_image.detach().cpu().numpy().astype(np.float32)) * h
    stl_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_model.stl")
    generate_stl(height_map_mm, stl_filename, background_height, scale=1.0)

    # Generate swap instructions
    background_layers = int(background_height // h)
    swap_instructions = generate_swap_instructions(
        disc_global.detach().cpu().numpy(),
        disc_height_image.detach().cpu().numpy(),
        h, background_layers, background_height, material_names
    )
    instructions_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_swap_instructions.txt")
    with open(instructions_filename, "w") as f:
        for line in swap_instructions:
            f.write(line + "\n")

    # Generate project file
    project_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_project.hfp")
    generate_project_file(
        project_filename, args,
        disc_global.detach().cpu().numpy(),
        disc_height_image.detach().cpu().numpy(),
        img_width, img_height, stl_filename, csv_file
    )

######################################################################
# PRUNING
######################################################################

def prune_num_colors(
        disc_global: torch.Tensor,
        pixel_height_logits: torch.Tensor,
        target: torch.Tensor,
        h: float,
        max_layers: int,
        material_colors: torch.Tensor,
        material_TDs: torch.Tensor,
        background: torch.Tensor,
        max_colors_allowed: int,
        tau_for_comp: float = 1e-3,
        perception_loss_module: Optional[torch.nn.Module] = None
) -> torch.Tensor:
    """
    Iteratively merge materials (colors) until the final number of distinct colors is less than or equal to max_colors_allowed.
    In each iteration, the function picks the single merge c_from->c_to that yields the smallest MSE increase.

    Args:
        disc_global (torch.Tensor): Discrete global material assignment.
        pixel_height_logits (torch.Tensor): Logits for pixel heights.
        target (torch.Tensor): Target image tensor.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (torch.Tensor): Tensor of material colors.
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters.
        background (torch.Tensor): Background color tensor.
        max_colors_allowed (int): Maximum number of colors allowed after pruning.
        tau_for_comp (float, optional): Temperature for compositing. Defaults to 1e-3.
        perception_loss_module (Optional[torch.nn.Module], optional): Module to compute perceptual loss. Defaults to None.

    Returns:
        torch.Tensor: The pruned discrete global material assignment.
    """

    def get_image_loss(dg_test):
        """
        Compute the loss for a given discrete global material assignment.

        Args:
            dg_test (torch.Tensor): Discrete global material assignment to test.

        Returns:
            torch.Tensor: The computed loss value.
        """
        with torch.no_grad():
            out_im = composite_image_combined_jit(
                pixel_height_logits,
                dg_test,
                tau_for_comp,
                tau_for_comp,
                h,
                max_layers,
                material_colors,
                material_TDs,
                background,
                mode="pruning"
            )
        return compute_loss(
            material_assignment=dg_test,  # discrete shape (max_layers,)
            comp=out_im,
            target=target,
            perception_loss_module=perception_loss_module,
            tau_global=tau_for_comp,
            num_materials=material_colors.shape[0],
            add_penalty_loss=False
        )

    # Current assignment
    best_dg = disc_global.clone()
    best_loss = get_image_loss(best_dg)

    tbar = tqdm(desc="Pruning - Merging colors", total=len(torch.unique(best_dg))-max_colors_allowed)
    while True:
        tbar.update(1)
        # Count distinct colors
        unique_mats = torch.unique(best_dg)
        cur_distinct = len(unique_mats)

        tbar.set_description(f"Pruning - Merging colors (Number of distinct colors: {cur_distinct}), Best Loss: {best_loss:.4f}")
        if len(unique_mats) <= max_colors_allowed:
            break  # Done!
        found_better_merge = False

        # Store the best candidate in this pass
        best_merge_loss = None
        best_merge_dg = None

        # Generate all pairs c_from, c_to among the distinct materials
        # c_from != c_to
        for c_from in unique_mats:
            for c_to in unique_mats:
                if c_to == c_from:
                    continue
                # Merge c_from -> c_to
                dg_test = merge_color(best_dg, c_from.item(), c_to.item())
                test_loss = get_image_loss(dg_test)

                mse_increase = test_loss - best_loss
                # pick the best (lowest final MSE) or smallest increase
                if (best_merge_loss is None) or (test_loss < best_merge_loss):
                    best_merge_loss = test_loss
                    best_merge_dg = dg_test

        # If we found a merge that improves or at least yields the smallest MSE:
        if best_merge_dg is not None and best_merge_loss is not None:
            # Accept that merge
            best_dg = best_merge_dg
            best_loss = best_merge_loss
            found_better_merge = True

        if not found_better_merge:
            # We couldn't find any beneficial merge. But we must reduce colors if possible.
            # If we still have more colors than allowed, we might have to forcibly do it anyway.
            # For demonstration, we'll just break to avoid wrecking MSE.
            break

    return best_dg


def prune_num_swaps(
        disc_global: torch.Tensor,
        pixel_height_logits: torch.Tensor,
        target: torch.Tensor,
        h: float,
        max_layers: int,
        material_colors: torch.Tensor,
        material_TDs: torch.Tensor,
        background: torch.Tensor,
        max_swaps_allowed: int,
        tau_for_comp: float = 1e-3,
        perception_loss_module: Optional[torch.nn.Module] = None
) -> torch.Tensor:
    """
    Iteratively reduce the number of color swaps (layer-to-layer changes)
    by merging adjacent layers' colors until the number of swaps is less than or equal to max_swaps_allowed.
    In each iteration, pick the single boundary and direction that yields the smallest final MSE.

    Args:
        disc_global (torch.Tensor): Discrete global material assignment.
        pixel_height_logits (torch.Tensor): Logits for pixel heights.
        target (torch.Tensor): Target image tensor.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (torch.Tensor): Tensor of material colors.
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters.
        background (torch.Tensor): Background color tensor.
        max_swaps_allowed (int): Maximum number of swaps allowed after pruning.
        tau_for_comp (float, optional): Temperature for compositing. Defaults to 1e-3.
        perception_loss_module (Optional[torch.nn.Module], optional): Module to compute perceptual loss. Defaults to None.

    Returns:
        torch.Tensor: The pruned discrete global material assignment.
    """

    def get_image_loss(dg_test):
        """
        Compute the loss for a given discrete global material assignment.

        Args:
            dg_test (torch.Tensor): Discrete global material assignment to test.

        Returns:
            torch.Tensor: The computed loss value.
        """
        with torch.no_grad():
            out_im = composite_image_combined_jit(
                pixel_height_logits,
                dg_test,
                tau_for_comp,
                tau_for_comp,
                h,
                max_layers,
                material_colors,
                material_TDs,
                background,
                mode="pruning"
            )
        return compute_loss(
            material_assignment=dg_test,  # discrete shape (max_layers,)
            comp=out_im,
            target=target,
            perception_loss_module=perception_loss_module,
            tau_global=tau_for_comp,
            num_materials=material_colors.shape[0],
            add_penalty_loss=False
        )

    best_dg = disc_global.clone()
    best_loss = get_image_loss(best_dg)
    bands = find_color_bands(best_dg)
    tbar = tqdm(desc="Pruning - Merging swaps", total=len(bands) - max_swaps_allowed)
    while True:
        # Identify color bands
        bands = find_color_bands(best_dg)
        num_bands = len(bands)
        tbar.update(1)
        tbar.set_description(f"Pruning - Merging swaps (Number of swaps: {num_bands - 1}), Best Loss: {best_loss:.4f}")
        # Number of swaps = num_bands - 1
        if (num_bands - 1) <= max_swaps_allowed:
            break  # we are done

        # Evaluate all adjacent merges and pick the single best
        best_merge_loss = None
        best_merge_dg_candidate = None

        for i in range(num_bands - 1):
            band_a = bands[i]
            band_b = bands[i + 1]
            if band_a[2] == band_b[2]:
                # Already the same color => no swap here
                continue
            # Forward merge: unify band_b's color to band_a
            dg_fwd = merge_bands(best_dg, band_a, band_b, direction="forward")
            loss_fwd = get_image_loss(dg_fwd)

            # Backward merge: unify band_a's color to band_b
            dg_bwd = merge_bands(best_dg, band_a, band_b, direction="backward")
            loss_bwd = get_image_loss(dg_bwd)

            # Pick whichever is better for this adjacency
            if loss_fwd < loss_bwd:
                candidate_loss = loss_fwd
                candidate_dg = dg_fwd
            else:
                candidate_loss = loss_bwd
                candidate_dg = dg_bwd

            # See if it's the best adjacency in this iteration
            if (best_merge_loss is None) or (candidate_loss < best_merge_loss):
                best_merge_loss = candidate_loss
                best_merge_dg_candidate = candidate_dg

        if best_merge_dg_candidate is not None:
            # Accept that single merge
            best_dg = best_merge_dg_candidate
            best_loss = best_merge_loss
        else:
            # No merges found that reduce or keep MSE in a "reasonable" range.
            # If you absolutely must reduce swaps, you'd forcibly do the best one anyway.
            break

    return best_dg

def prune_colors_and_swaps(
    disc_global: torch.Tensor,
    pixel_height_logits: torch.Tensor,
    target: torch.Tensor,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,
    material_TDs: torch.Tensor,
    background: torch.Tensor,
    max_colors_allowed: int,
    max_swaps_allowed: int,
    tau_for_comp: float = 1e-3,
    perception_loss_module: Optional[torch.nn.Module] = None
) -> torch.Tensor:
    """
    Prune the number of distinct colors and the number of color swaps in the material assignment.

    This function performs two steps:
    1) Prune down the number of distinct colors to <= max_colors_allowed.
    2) Prune down the number of swaps to <= max_swaps_allowed.

    Args:
        disc_global (torch.Tensor): Discrete global material assignment.
        pixel_height_logits (torch.Tensor): Logits for pixel heights.
        target (torch.Tensor): Target image tensor.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (torch.Tensor): Tensor of material colors.
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters.
        background (torch.Tensor): Background color tensor.
        max_colors_allowed (int): Maximum number of colors allowed after pruning.
        max_swaps_allowed (int): Maximum number of swaps allowed after pruning.
        tau_for_comp (float, optional): Temperature for compositing. Defaults to 1e-3.
        perception_loss_module (Optional[torch.nn.Module], optional): Module to compute perceptual loss. Defaults to None.

    Returns:
        torch.Tensor: The pruned discrete global material assignment.
    """
    # Step 1: Limit total number of colors.
    dg_after_color = prune_num_colors(
        disc_global,
        pixel_height_logits, target,
        h, max_layers,
        material_colors, material_TDs, background,
        max_colors_allowed,
        tau_for_comp,
        perception_loss_module = perception_loss_module
    )

    # Step 2: Limit total number of swaps.
    dg_after_swaps = prune_num_swaps(
        dg_after_color,
        pixel_height_logits, target,
        h, max_layers,
        material_colors, material_TDs, background,
        max_swaps_allowed,
        tau_for_comp,
        perception_loss_module = perception_loss_module
    )

    return dg_after_swaps


######################################################################
# MAIN
######################################################################

def main():
    parser = configargparse.ArgParser()
    parser.add_argument("--config", is_config_file=True, help="Path to config file")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with material data")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to write outputs")
    parser.add_argument("--iterations", type=int, default=5000, help="Number of optimization iterations")
    parser.add_argument("--best_loss_iterations", type=float, default=0.9, help="Percentage of optimization iterations after which we start to record the best loss")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate for optimization")
    parser.add_argument("--layer_height", type=float, default=0.04, help="Layer thickness in mm")
    parser.add_argument("--max_layers", type=int, default=75, help="Maximum number of layers")
    parser.add_argument("--background_height", type=float, default=0.4, help="Height of the background in mm")
    parser.add_argument("--background_color", type=str, default="#000000", help="Background color")
    parser.add_argument("--output_size", type=int, default=1024, help="Maximum dimension for target image")
    parser.add_argument("--solver_size", type=int, default=128, help="Maximum dimension for solver (fast) image")
    parser.add_argument("--decay", type=float, default=0.01, help="Final tau value for Gumbel-Softmax")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization during optimization")
    parser.add_argument("--perform_pruning", type=bool, default=True, help="Perform pruning after optimization")
    parser.add_argument("--pruning_max_colors", type=int, default=10, help="Max number of colors allowed after pruning")
    parser.add_argument("--pruning_max_swaps", type=int, default=20, help="Max number of swaps allowed after pruning")
    parser.add_argument("--save_interval_pct", type=float, default=20, help="Percentage interval to save checkpoints")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    os.makedirs(args.output_folder, exist_ok=True)
    print("Output folder:", args.output_folder)
    assert (args.background_height / args.layer_height).is_integer(), "Background height must be divisible by layer height."
    assert args.max_layers > 1, "max_layers must be positive."
    assert args.output_size > 0, "output_size must be positive."
    assert args.solver_size > 0, "solver_size must be positive."
    assert args.iterations > 0, "iterations must be positive."
    assert args.learning_rate > 0, "learning_rate must be positive."
    assert args.layer_height > 0, "layer_height must be positive."

    h_value = args.layer_height
    max_layers_value = args.max_layers
    background_height_value = args.background_height
    background_layers_value = int(background_height_value // h_value)
    decay_v_value = args.decay

    # Load background as torch
    background = torch.tensor(hex_to_rgb(args.background_color), dtype=torch.float32,device=device)

    # Load materials
    material_colors, material_TDs, material_names, _ = load_materials(args.csv_file)
    material_colors = torch.tensor(material_colors, dtype=torch.float32,device=device)
    material_TDs    = torch.tensor(material_TDs,    dtype=torch.float32,device=device)

    # Load and resize images
    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # This smaller 'target' is the solver resolution
    target_np   = resize_image(img, args.solver_size)
    new_h, new_w, _ = target_np.shape
    target      = torch.tensor(target_np, dtype=torch.float32,device=device)

    # This is the higher-res final version
    output_target_np = resize_image(img, args.output_size)
    output_target    = torch.tensor(output_target_np, dtype=torch.float32,device=device)

    # Initialize pixel_height logits from the full-size image
    pixel_height_logits_init = init_height_map(output_target_np, args.max_layers, h_value)
    pixel_height_logits_init = np.asarray(pixel_height_logits_init)

    # Reshape to solver size for the actual optimization
    o_shape = pixel_height_logits_init.shape
    # Add a dummy channel to resize properly
    phl_solver_np_3d = pixel_height_logits_init.reshape(o_shape[0], o_shape[1], 1)
    phl_solver_np = resize_image(phl_solver_np_3d, args.solver_size)

    # Initialize Perception loss module
    perception_loss_module = MultiLayerVGGPerceptualLoss().to(args.device).eval()

    # Solve at solver resolution
    best_params, best_comp = run_optimizer(
        target, phl_solver_np, new_h, new_w, max_layers_value, h_value,
        material_colors, material_TDs, background,
        args.iterations, args.learning_rate, decay_v_value, loss_function=loss_fn,
        visualize=args.visualize,
        output_folder=args.output_folder,
        save_interval_pct=args.save_interval_pct if args.save_interval_pct > 0 else None,
        img_width=new_w, img_height=new_h,
        background_height=background_height_value,
        material_names=material_names,
        csv_file=args.csv_file,
        args=args,
        perception_loss_module=perception_loss_module
    )

    # If you want to apply the best solution to the large image, just replace the pixel_height_logits:
    best_params["pixel_height_logits"] = torch.tensor(pixel_height_logits_init, dtype=torch.float32,device=device)

    # Discretize for final output
    disc_global, disc_height_image = discretize_solution_jax(
        best_params, decay_v_value, h_value, max_layers_value
    )

    # Optionally prune
    if args.perform_pruning:
        disc_global = prune_colors_and_swaps(
            disc_global,
            best_params["pixel_height_logits"],
            output_target,
            h_value,
            max_layers_value,
            material_colors,
            material_TDs,
            background,
            max_colors_allowed=args.pruning_max_colors,
            max_swaps_allowed=args.pruning_max_swaps,
            perception_loss_module = perception_loss_module
        )

    # Composite final
    with torch.no_grad():
        disc_comp = composite_image_combined_jit(
            best_params['pixel_height_logits'], disc_global,
            decay_v_value, decay_v_value,
            h_value, max_layers_value, material_colors, material_TDs, background,
            mode="pruning"
        )

    disc_comp_np = np.clip(disc_comp.detach().cpu().numpy(), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_folder, "discrete_comp.jpg"),
                cv2.cvtColor(disc_comp_np, cv2.COLOR_RGB2BGR))

    # Export final STL
    height_map_mm = (disc_height_image.detach().cpu().numpy().astype(np.float32)) * h_value
    stl_filename = os.path.join(args.output_folder, "final_model.stl")
    generate_stl(height_map_mm, stl_filename, background_height_value, scale=1.0)

    # Swap instructions
    swap_instructions = generate_swap_instructions(
        disc_global.detach().cpu().numpy(),
        disc_height_image.detach().cpu().numpy(),
        h_value, background_layers_value, background_height_value, material_names
    )
    with open(os.path.join(args.output_folder, "swap_instructions.txt"), "w") as f:
        for line in swap_instructions:
            f.write(line + "\n")

    # Final project file
    project_filename = os.path.join(args.output_folder, "project_file.hfp")
    generate_project_file(
        project_filename, args,
        disc_global.detach().cpu().numpy(),
        disc_height_image.detach().cpu().numpy(),
        output_target.shape[1], output_target.shape[0],
        stl_filename, args.csv_file
    )

    print("Project file saved to", project_filename)
    print("All outputs saved to", args.output_folder)
    print("Happy printing!")


if __name__ == '__main__':
    main()
