#!/usr/bin/env python
"""
Script for generating 3D printed layered models from an input image.

This script uses a learned optimization with a Gumbel softmax formulation
to assign materials per layer and produce both a discretized composite that
is exported as an STL file along with swap instructions.
"""
import time
import configargparse
import cv2
import torch

from autoforge.helper_functions import hex_to_rgb, load_materials, \
    generate_stl, generate_swap_instructions, generate_project_file, init_height_map, resize_image

import optax
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiLayerVGGPerceptualLoss(nn.Module):
    def __init__(self, layers: list = None, weights: list = None):
        """
        Uses a pretrained VGG16 model to extract features from multiple layers.
        By default, it uses layers [3, 8, 15, 22] (approximately conv1_2, conv2_2, conv3_3, conv4_3).
        """
        super(MultiLayerVGGPerceptualLoss, self).__init__()
        # Choose layers from VGG16.features
        if layers is None:
            layers = [3, 8, 15, 22]  # These indices roughly correspond to conv1_2, conv2_2, conv3_3, conv4_3.
        self.layers = layers

        # Load pretrained VGG16 and freeze parameters.
        vgg = models.vgg16(pretrained=True).features
        for param in vgg.parameters():
            param.requires_grad = False

        # We want to run the network up to the highest required layer.
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max(layers) + 1)]).eval()

        # Weights for each selected layer loss; default: equal weighting.
        if weights is None:
            weights = [1.0 / len(layers)] * len(layers)
        self.weights = weights

        # Register ImageNet normalization constants as buffers.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x and y are expected to be of shape (N, 3, H, W) with values in [0, 255].
        They are normalized to ImageNet stats and then passed through VGG16.
        The loss is computed as a weighted sum of MSE losses on the selected layers.
        """
        # Normalize images
        x = (x / 255.0 - self.mean) / self.std
        y = (y / 255.0 - self.mean) / self.std

        loss = 0.0
        out = x
        # Loop through VGG layers and compute losses at the selected layers.
        for i, layer in enumerate(self.vgg):
            out = layer(out)
            if i in self.layers:
                # Extract corresponding feature for y by running y through the same layers.
                with torch.no_grad():
                    out_y = y
                    for j in range(i + 1):
                        out_y = self.vgg[j](out_y)
                loss += self.weights[self.layers.index(i)] * F.mse_loss(out, out_y)
        return loss


def decode_assignment(swap_params: torch.Tensor,
                      color_logits: torch.Tensor,
                      max_layers: int,
                      material_colors: torch.Tensor,
                      material_TDs: torch.Tensor,
                      tau: float) -> (torch.Tensor, torch.Tensor):
    """
    Given learnable parameters:
      - swap_params: (num_swaps,) that control where color swaps occur,
      - color_logits: (num_swaps+1, num_materials) for each segment,
    this function computes a soft assignment for each layer.

    For each segment, a gumbel softmax is applied to the logits (with temperature tau)
    to obtain a probability distribution over the fixed palette. This is used to compute
    both a weighted color and a weighted TD. Then, using soft memberships (via sigmoids
    with steepness s), we combine the segments over layers.

    Returns a tuple (global_colors, global_TDs) where:
      - global_colors is (max_layers, 3)
      - global_TDs is (max_layers,)
    """
    num_swaps = swap_params.shape[0]
    num_segments = num_swaps + 1
    # Make swap_params positive and compute cumulative sum.
    pos = F.softplus(swap_params)  # (num_swaps,)
    cum = torch.cumsum(pos, dim=0)
    # Scale cumulative values to [1, max_layers-1]
    swap_positions = 1 + (max_layers - 2) * (cum - cum.min()) / (cum.max() - cum.min() + 1e-8)  # (num_swaps,)
    # Create layer indices 0,...,max_layers-1.
    j = torch.arange(max_layers, device=swap_params.device).float()  # (max_layers,)
    memberships = []
    # For segment 0:
    m0 = torch.sigmoid(swap_positions[0] - j)
    memberships.append(m0)
    # For segments 1 ... num_segments-1:
    for i in range(1, num_segments):
        if i < num_swaps:
            mi = torch.sigmoid(j - swap_positions[i - 1]) - torch.sigmoid(j - swap_positions[i])
        else:
            mi = 1 - torch.sigmoid(j - swap_positions[-1])
        memberships.append(mi)
    # memberships: (num_segments, max_layers)
    memberships = torch.stack(memberships, dim=0)  # (num_segments, max_layers)

    # For each segment, compute the weighted color and TD.
    segment_colors = []
    segment_TDs = []
    for i in range(num_segments):
        probs = F.gumbel_softmax(color_logits[i], tau=tau, hard=False)  # (num_materials,)
        color_i = torch.matmul(probs, material_colors)  # (3,)
        td_i = torch.dot(probs, material_TDs)  # scalar
        segment_colors.append(color_i)
        segment_TDs.append(td_i)
    segment_colors = torch.stack(segment_colors, dim=0)  # (num_segments, 3)
    segment_TDs = torch.stack(segment_TDs, dim=0)  # (num_segments,)

    # Combine segments for each layer:
    # memberships: (num_segments, max_layers) --> transpose to (max_layers, num_segments)
    global_colors = torch.matmul(memberships.t(), segment_colors)  # (max_layers, 3)
    global_TDs = torch.matmul(memberships.t(), segment_TDs.unsqueeze(1))  # (max_layers, 1)
    global_TDs = global_TDs.squeeze(1)  # (max_layers,)
    return global_colors, global_TDs

# An adaptive round function that chooses a soft or hard round based on tau.
@torch.jit.script
def adaptive_round(x: torch.Tensor, tau: float, high_tau: float, low_tau: float, temp: float) -> torch.Tensor:
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


@torch.jit.script
def composite_image_soft(pixel_height_logits: torch.Tensor,
                         global_colors: torch.Tensor,
                         global_TDs: torch.Tensor,
                         tau_height: float,
                         h: float,
                         max_layers: int,
                         background: torch.Tensor) -> torch.Tensor:
    """
    Compute the composite image given pixel height logits and a soft global assignment.
    global_colors: (max_layers, 3) and global_TDs: (max_layers,) are used per layer.
    """
    H, W = pixel_height_logits.shape
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logits)  # (H,W)
    continuous_layers = pixel_height / h  # (H,W)
    adaptive_layers = adaptive_round(continuous_layers, tau_height, 0.1, 0.01, 0.1)
    hard_round = torch.round(continuous_layers)
    discrete_layers = torch.round(hard_round + (adaptive_layers - hard_round).detach()).to(torch.int64)  # (H,W)

    layers = torch.arange(max_layers, device=pixel_height_logits.device).view(max_layers, 1, 1)
    mask = (layers < discrete_layers.unsqueeze(0)).to(torch.float32)  # (max_layers, H, W)
    eff_thick = mask * h  # (max_layers, H, W)

    A: float = 0.1215
    k: float = 61.6970
    b: float = 0.4773
    # For each layer j, use the corresponding TD from global_TDs:
    # Expand global_TDs to (max_layers, 1, 1)
    TDs = global_TDs.view(max_layers, 1, 1) * 0.1

    opac = A * torch.log(1 + k * (eff_thick / TDs)) + b * (eff_thick / TDs)
    opac = torch.clamp(opac, 0.0, 1.0)
    opac = opac.flip(0)
    colors = global_colors.flip(0)  # (max_layers, 3)

    comp = torch.zeros((H, W, 3), dtype=torch.float32, device=pixel_height_logits.device)
    remaining = torch.ones((H, W), dtype=torch.float32, device=pixel_height_logits.device)
    for j in range(max_layers):
        opac_j = opac[j]  # (H, W)
        color_j = colors[j].view(1, 1, 3)
        comp = comp + remaining.unsqueeze(-1) * opac_j.unsqueeze(-1) * color_j
        remaining = remaining * (1 - opac_j)
    result = comp + remaining.unsqueeze(-1) * background.view(1, 1, 3)
    return result * 255.0


@torch.jit.script
def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic * quadratic + delta * linear
    return torch.mean(loss)


def decoupled_perceptual_loss_fn(compare_image: torch.Tensor,
                                 target_image: torch.Tensor,
                                 perceptual_module: nn.Module,
                                 lum_weight: float = 1.0,
                                 color_weight: float = 1.0,
                                 mse_weight: float = 1.0) -> torch.Tensor:
    """
    Computes a decoupled perceptual loss by separating luminance and chrominance.

    - The luminance is computed as 0.299*R + 0.587*G + 0.114*B.
    - The chrominance is defined as the residual (RGB - luminance).

    Both components are passed separately through the perceptual network
    (after replicating the luminance to 3 channels) and then combined.

    Inputs:
      - compare_image and target_image are assumed to be (H, W, 3) tensors with values in [0,255].

    Returns:
      - A weighted sum of the luminance and color perceptual losses.
    """
    # Compute luminance (grayscale)
    comp_gray = 0.299 * compare_image[..., 0] + 0.587 * compare_image[..., 1] + 0.114 * compare_image[..., 2]
    target_gray = 0.299 * target_image[..., 0] + 0.587 * target_image[..., 1] + 0.114 * target_image[..., 2]

    # Replicate grayscale to 3 channels
    comp_gray = comp_gray.unsqueeze(-1).repeat(1, 1, 3)
    target_gray = target_gray.unsqueeze(-1).repeat(1, 1, 3)

    # Prepare batches for VGG (N,3,H,W)
    comp_batch_gray = comp_gray.permute(2, 0, 1).unsqueeze(0)
    target_batch_gray = target_gray.permute(2, 0, 1).unsqueeze(0)

    # Luminance perceptual loss
    lum_loss = perceptual_module(comp_batch_gray, target_batch_gray)

    # Compute chrominance: difference between the original and grayscale images.
    comp_color = compare_image - comp_gray
    target_color = target_image - target_gray
    comp_batch_color = comp_color.permute(2, 0, 1).unsqueeze(0)
    target_batch_color = target_color.permute(2, 0, 1).unsqueeze(0)

    # Color perceptual loss
    color_loss = perceptual_module(comp_batch_color, target_batch_color)

    mse_loss = mse_weight * huber_loss(compare_image, target_image, delta=0.1) if mse_weight > 0 else 0.0

    # Optionally, you can add a simple MSE term on one or both components.
    # Here we simply return the weighted sum of perceptual losses.
    return lum_weight * lum_loss + color_weight * color_loss + mse_loss


def optimize_color_assignment(pixel_height_logits: torch.Tensor,
                              target: torch.Tensor,
                              h: float,
                              max_layers: int,
                              material_colors: torch.Tensor,
                              material_TDs: torch.Tensor,
                              background: torch.Tensor,
                              num_swaps: int,
                              initial_tau: float,
                              final_tau: float,
                              perceptual_module: nn.Module,
                              num_iters: int,
                              lr: float,
                              device) -> (torch.Tensor, torch.Tensor):
    """
    Jointly optimize the low-dimensional parameters that define the global color assignment.
    Returns a tuple (global_colors, global_TDs) from the optimized parameters.
    """
    num_materials = material_colors.size(0)
    swap_params = nn.Parameter(torch.randn(num_swaps, device=device))
    color_logits = nn.Parameter(torch.randn(num_swaps + 1, num_materials, device=device))
    optimizer = torch.optim.AdamW([swap_params, color_logits], lr=lr,weight_decay=1e-3)

    warmup_steps = int(num_iters * 0.75)
    decay_rate = -math.log(final_tau) / (num_iters - warmup_steps)

    def get_tau(i, tau_init=1.0, tau_final=final_tau, decay_rate=decay_rate):
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


    best_loss = float('inf')
    best_params = None

    tbar = tqdm(range(num_iters))
    for it in tbar:

        tau = get_tau(it, tau_init=initial_tau, tau_final=final_tau, decay_rate=decay_rate)

        global_colors, global_TDs = decode_assignment(swap_params, color_logits, max_layers,
                                                      material_colors, material_TDs, tau)
        comp = composite_image_soft(pixel_height_logits, global_colors, global_TDs,
                                    tau_height=final_tau, h=h, max_layers=max_layers,
                                    background=background)

        loss = decoupled_perceptual_loss_fn(comp, target, perceptual_module, lum_weight=1.0, color_weight=1.0, mse_weight=0.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 10 == 0:

            global_colors, global_TDs = decode_assignment(swap_params, color_logits, max_layers,
                                                          material_colors, material_TDs, final_tau)
            comp_val = composite_image_soft(pixel_height_logits, global_colors, global_TDs,
                                        tau_height=final_tau, h=h, max_layers=max_layers,
                                        background=background)
            val_loss = decoupled_perceptual_loss_fn(comp, target, perceptual_module, lum_weight=1.0, color_weight=1.0, mse_weight=0)

            if val_loss < best_loss:
                best_loss = val_loss

                best_params = (swap_params.clone(), color_logits.clone())
                comp_val_np = comp_val.cpu().detach().numpy().clip(0, 255).astype(np.uint8)
                plt.imshow(comp_val_np)
                plt.title(f"Iter {it}")
                plt.pause(0.1)
            tbar.set_description(f"Iter {it}/{num_iters}, Loss: {loss.item():.4f}, Val Loss: {best_loss.item():.4f}, tau: {tau:.4f}")
    # Return the final global assignment.
    return best_params


def main():
    parser = configargparse.ArgParser()
    parser.add_argument("--config", is_config_file=True, help="Path to config file")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with material data")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to write outputs")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of optimization iterations")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate for optimization")
    parser.add_argument("--layer_height", type=float, default=0.04, help="Layer thickness in mm")
    parser.add_argument("--max_layers", type=int, default=75, help="Maximum number of layers")
    parser.add_argument("--background_height", type=float, default=0.4, help="Height of the background in mm")
    parser.add_argument("--background_color", type=str, default="#000000", help="Background color")
    parser.add_argument("--output_size", type=int, default=1024, help="Maximum dimension for target image")
    parser.add_argument("--solver_size", type=int, default=128, help="Maximum dimension for target image")

    parser.add_argument("--decay", type=float, default=0.001, help="Final tau value for optimization")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization during optimization")
    parser.add_argument("--max_num_color_swaps", type=int, default=5, help="Number of color swaps for genetic algorithm")


    parser.add_argument("--save_interval_pct", type=float, default=20,help="Percentage interval to save intermediate results")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    args = parser.parse_args()
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

    background_rgb = hex_to_rgb(args.background_color)
    background = torch.tensor(background_rgb, dtype=torch.float32, device=device)

    material_colors, material_TDs, material_names, material_hex = load_materials(args.csv_file)
    material_colors = torch.tensor(material_colors, dtype=torch.float32, device=device)
    material_TDs = torch.tensor(material_TDs, dtype=torch.float32, device=device)

    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target_img = resize_image(img, args.solver_size)
    H, W, _ = target_img.shape
    target = torch.tensor(target_img, dtype=torch.float32, device=device)

    output_target = resize_image(img, args.output_size)
    pixel_height_logits = init_height_map(output_target, args.max_layers, h_value)
    pixel_height_logits = np.array(pixel_height_logits)
    o_shape = pixel_height_logits.shape
    pixel_height_logits_solver = resize_image(pixel_height_logits.reshape(o_shape[0], o_shape[1], 1), args.solver_size)
    pixel_height_logits_solver = torch.tensor(pixel_height_logits_solver, dtype=torch.float32, device=device)

    perceptual_module = MultiLayerVGGPerceptualLoss(layers=[3,8,15,22]).to(device)
    perceptual_module.eval()
    perceptual_module = torch.compile(perceptual_module)

    # Optimize the color assignment parameters by gradient descent.
    global_assignment = optimize_color_assignment(pixel_height_logits_solver, target,
                                                  h=args.layer_height, max_layers=args.max_layers,
                                                  material_colors=material_colors,
                                                  material_TDs=material_TDs,
                                                  background=background,
                                                  num_swaps=args.max_num_color_swaps,
                                                  initial_tau=1,
                                                  final_tau=args.decay,
                                                  perceptual_module=perceptual_module,
                                                  num_iters=args.iterations,
                                                  lr=args.learning_rate,
                                                  device=device)


    if args.perform_pruning:
        disc_global = pruning(output_target,best_params,disc_global,tau_global_disc,val_gumbel_keys,h_value,args.max_layers,material_colors,material_TDs,background,max_loss_increase=args.pruning_max_loss_increase)

    disc_comp = composite_image_combined_jit(best_params['pixel_height_logits'], disc_global,
                                             tau_global_disc, tau_global_disc, val_gumbel_keys,
                                             h_value, max_layers_value, material_colors, material_TDs, background, mode="pruning")
    discrete_comp_np = np.clip(np.array(disc_comp), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_folder, "discrete_comp.jpg"),
                cv2.cvtColor(discrete_comp_np, cv2.COLOR_RGB2BGR))

    height_map_mm = (np.array(disc_height_image, dtype=np.float32)) * h_value
    stl_filename = os.path.join(args.output_folder, "final_model.stl")
    generate_stl(height_map_mm, stl_filename, background_height_value, scale=1.0)

    swap_instructions = generate_swap_instructions(np.array(disc_global), np.array(disc_height_image),
                                                   h_value, background_layers_value, background_height_value, material_names)
    instructions_filename = os.path.join(args.output_folder, "swap_instructions.txt")
    with open(instructions_filename, "w") as f:
        for line in swap_instructions:
            f.write(line + "\n")

    width_mm = new_w
    height_mm = new_h

    project_filename = os.path.join(args.output_folder, "project_file.hfp")
    generate_project_file(project_filename, args,
                          np.array(disc_global),
                          np.array(disc_height_image),
                          width_mm, height_mm, stl_filename, args.csv_file)
    print("Project file saved to", project_filename)
    print("All outputs saved to", args.output_folder)
    print("Happy printing!")





if __name__ == '__main__':
    main()
