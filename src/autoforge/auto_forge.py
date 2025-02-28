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
import jax
import torch

from autoforge.helper_functions import hex_to_rgb, load_materials, \
    generate_stl, generate_swap_instructions, generate_project_file, init_height_map, resize_image

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch.nn.functional as F

import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Define a VGG-based perceptual loss module.
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
def composite_pixel(pixel_height_logit: torch.Tensor,
                    global_material_indices: torch.Tensor,
                    tau_height: float,
                    h: float,
                    max_layers: int,
                    material_colors: torch.Tensor,
                    material_TDs: torch.Tensor,
                    background: torch.Tensor) -> torch.Tensor:

    # Compute continuous pixel height (in physical units)
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logit)
    continuous_layers = pixel_height / h
    adaptive_layers = adaptive_round(continuous_layers, tau_height, 0.1, 0.01, 0.1)
    hard_round_val = torch.round(continuous_layers)
    # Stop gradient equivalent: subtract hard round and add detached soft version.
    discrete_layers = hard_round_val + (adaptive_layers - hard_round_val).detach()
    d_layers = int(torch.round(discrete_layers).item())
    # Opacity parameters
    A = 0.1215
    k = 61.6970
    b = 0.4773
    comp = torch.zeros(3, dtype=torch.float32)
    remaining = 1.0
    for i in range(max_layers):
        j = max_layers - 1 - i  # process layers from top to bottom
        p_print = 1.0 if j < d_layers else 0.0
        eff_thick = p_print * h
        # Look up the material index (global_material_indices is a 1D tensor of length max_layers)
        material_index = int(global_material_indices[j].item())
        color_i = material_colors[material_index]
        TD_i = material_TDs[material_index] * 0.1
        opac = A * torch.log(1 + k * (eff_thick / TD_i)) + b * (eff_thick / TD_i)
        opac = torch.clamp(opac, 0.0, 1.0)
        comp = comp + remaining * opac * color_i
        remaining = remaining * (1 - opac)
    result = comp + remaining * background
    return result * 255.0


# Composite an entire image by iterating over pixels.
@torch.jit.script
def composite_image(pixel_height_logits: torch.Tensor,
                               global_material_indices: torch.Tensor,
                               tau_height: float,
                               h: float,
                               max_layers: int,
                               material_colors: torch.Tensor,
                               material_TDs: torch.Tensor,
                               background: torch.Tensor) -> torch.Tensor:

    H, W = pixel_height_logits.shape

    # Compute per-pixel continuous height
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logits)  # shape (H, W)
    continuous_layers = pixel_height / h  # shape (H, W)
    adaptive_layers = adaptive_round(continuous_layers, tau_height, 0.1, 0.01, 0.1)  # shape (H, W)
    hard_round = torch.round(continuous_layers)
    discrete_layers = torch.round(hard_round + (adaptive_layers - hard_round).detach()).to(torch.int64)  # shape (H, W)

    # Create a layers index tensor of shape (max_layers, 1, 1)
    layers = torch.arange(max_layers, device=pixel_height_logits.device).view(max_layers, 1, 1)
    mask = (layers < discrete_layers.unsqueeze(0)).to(torch.float32)  # shape (max_layers, H, W)
    eff_thick = mask * h  # shape (max_layers, H, W)

    # Get per-layer material properties from global_material_indices.
    # global_material_indices is of shape (max_layers,)
    colors = material_colors[global_material_indices]  # shape (max_layers, 3)
    TDs = (material_TDs[global_material_indices] * 0.1).view(max_layers, 1, 1)  # shape (max_layers, 1, 1)

    # Compute opacity for each layer at every pixel.
    A: float = 0.1215
    k: float = 61.6970
    b: float = 0.4773
    opac = A * torch.log(1 + k * (eff_thick / TDs)) + b * (eff_thick / TDs)
    opac = torch.clamp(opac, 0.0, 1.0)  # shape (max_layers, H, W)

    # We need to accumulate contributions from layers from top (highest index) down to bottom.
    # Reverse the order so that index 0 is the top layer.
    opac = opac.flip(0)  # shape (max_layers, H, W)
    colors = colors.flip(0)  # shape (max_layers, 3)

    # Initialize composite image and remaining "ink"
    comp = torch.zeros((H, W, 3), dtype=torch.float32, device=pixel_height_logits.device)
    remaining = torch.ones((H, W), dtype=torch.float32, device=pixel_height_logits.device)

    # Loop over layers (this loop now runs only max_layers times, which is small)
    for j in range(max_layers):
        # Get the j-th layer's opacity map (H, W)
        opac_j = opac[j]  # shape (H, W)
        # Get the j-th layer's color (3,), then expand to (H, W, 3)
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




def perceptual_loss_fn(pixel_height_logits: torch.Tensor,
                       global_material_indices: torch.Tensor,
                       target: torch.Tensor,
                       tau_height: float,
                       h: float,
                       max_layers: int,
                       material_colors: torch.Tensor,
                       material_TDs: torch.Tensor,
                       background: torch.Tensor,
                       perceptual_module: nn.Module) -> torch.Tensor:
    """
    Computes the perceptual loss between the composite image (computed via your
    vectorized composite_image function) and the target image. Both images are converted
    to shape (1, 3, H, W) and then passed through the perceptual network.
    """
    comp = composite_image(pixel_height_logits, global_material_indices, tau_height,
                           h, max_layers, material_colors, material_TDs, background)
    # Convert composite and target to (1, 3, H, W)
    comp_batch = comp.permute(2, 0, 1).unsqueeze(0)
    target_batch = target.permute(2, 0, 1).unsqueeze(0)
    return perceptual_module(comp_batch, target_batch) + F.mse_loss(comp_batch, target_batch)


def create_update_step(optimizer, loss_function, h, max_layers, material_colors, material_TDs, background):
    """
    Create a JIT-compiled update step function using the specified loss function.

    Args:
        optimizer (optax.GradientTransformation): The optimizer to use for updating parameters.
        loss_function (callable): The loss function to compute gradients.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (jnp.ndarray): Array of material colors.
        material_TDs (jnp.ndarray): Array of material transmission/opacity parameters.
        background (jnp.ndarray): Background color.

    Returns:
        callable: A JIT-compiled function that performs a single update step.
    """

    @jax.jit
    def update_step(params, target, global_material_indices, tau_height, opt_state):
        loss_val, grads = jax.value_and_grad(loss_function)(
            params, global_material_indices, target, tau_height, h, max_layers, material_colors, material_TDs,
            background)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    return update_step



# A helper for decoding a GA individual into a full material assignment.
def decode_gene(swap_positions, segment_colors, max_layers):
    gene_array = np.empty(max_layers, dtype=np.int32)
    prev = 0
    for pos, color in zip(swap_positions, segment_colors):
        gene_array[prev:pos] = color
        prev = pos
    gene_array[prev:max_layers] = segment_colors[-1]
    return gene_array


def genetic_algorithm_color_swaps(pixel_height_logits, target, tau_height, h, max_layers,
                                  material_colors, material_TDs, background, num_swaps,
                                  material_count, population_size=50, generations=100,
                                  mutation_rate=0.1, visualize=False, perceptual_module=None):
    """
    Use a genetic algorithm to optimize the global color assignment.
    Each individual is represented as a tuple (swap_positions, segment_colors),
    and is decoded into a full material assignment.
    """
    import random as pyrandom

    def random_individual():
        swap_positions = np.sort(np.random.choice(np.arange(1, max_layers), size=num_swaps, replace=False))
        segment_colors = np.random.randint(0, material_count, size=num_swaps + 1)
        return (swap_positions, segment_colors)

    def fitness(individual):
        swap_positions, segment_colors = individual
        candidate = decode_gene(swap_positions, segment_colors, max_layers)
        candidate_tensor = torch.tensor(candidate, dtype=torch.int64, device=pixel_height_logits.device)
        # Use the composite image (from your fast vectorized function) as input.
        loss_val = perceptual_loss_fn(pixel_height_logits, candidate_tensor, target,
                                      tau_height, h, max_layers,
                                      material_colors, material_TDs, background,
                                      perceptual_module)
        return loss_val.item()

    population = [random_individual() for _ in range(population_size)]
    best_individual = None
    best_fitness = float('inf')
    for gen in tqdm(range(generations)):
        fitness_values = [fitness(ind) for ind in population]
        for ind, fit in zip(population, fitness_values):
            if fit < best_fitness:
                best_fitness = fit
                best_individual = ind
        print(f"Generation {gen}, best fitness: {best_fitness:.4f}, best gene: {best_individual}")
        if visualize:
            swap_positions, segment_colors = best_individual
            candidate = decode_gene(swap_positions, segment_colors, max_layers)
            candidate_tensor = torch.tensor(candidate, dtype=torch.int64)
            comp = composite_image(pixel_height_logits, candidate_tensor, tau_height,
                                   h, max_layers, material_colors, material_TDs, background)
            comp_np = comp.cpu().numpy().astype(np.uint8)
            plt.imshow(comp_np)
            plt.title(f"Generation {gen}, Loss: {best_fitness:.4f}")
            plt.pause(0.001)
        new_population = []
        while len(new_population) < population_size:
            ind1 = min(pyrandom.sample(population, 3), key=lambda x: fitness(x))
            ind2 = min(pyrandom.sample(population, 3), key=lambda x: fitness(x))
            swap_pos1, seg_colors1 = ind1
            swap_pos2, seg_colors2 = ind2
            child_swap = np.array([np.random.choice([a, b]) for a, b in zip(swap_pos1, swap_pos2)])
            child_swap = np.sort(child_swap)
            child_seg = np.array([np.random.choice([a, b]) for a, b in zip(seg_colors1, seg_colors2)])
            if np.random.rand() < mutation_rate:
                idx = np.random.randint(0, num_swaps)
                child_swap[idx] = np.clip(child_swap[idx] + np.random.randint(-5, 6), 1, max_layers - 1)
                child_swap = np.sort(child_swap)
            if np.random.rand() < mutation_rate:
                idx = np.random.randint(0, num_swaps + 1)
                child_seg[idx] = np.random.randint(0, material_count)
            new_population.append((child_swap, child_seg))
        population = new_population
    best_global_material_indices = decode_gene(best_individual[0], best_individual[1], max_layers)
    return torch.tensor(best_global_material_indices, dtype=torch.int64), best_fitness

def main():
    parser = configargparse.ArgParser()
    parser.add_argument("--config", is_config_file=True, help="Path to config file")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with material data")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to write outputs")
    parser.add_argument("--iterations", type=int, default=5000, help="Number of optimization iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate for optimization")
    parser.add_argument("--layer_height", type=float, default=0.04, help="Layer thickness in mm")
    parser.add_argument("--max_layers", type=int, default=75, help="Maximum number of layers")
    parser.add_argument("--background_height", type=float, default=0.4, help="Height of the background in mm")
    parser.add_argument("--background_color", type=str, default="#000000", help="Background color")
    parser.add_argument("--output_size", type=int, default=1024, help="Maximum dimension for target image")
    parser.add_argument("--solver_size", type=int, default=128, help="Maximum dimension for target image")
    parser.add_argument("--decay", type=float, default=0.01, help="Final tau value for optimization")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization during optimization")
    parser.add_argument("--num_color_swaps", type=int, default=40, help="Number of color swaps for genetic algorithm")
    parser.add_argument("--ga_population", type=int, default=50, help="GA population size")
    parser.add_argument("--ga_generations", type=int, default=200, help="Number of GA generations")
    parser.add_argument("--ga_mutation_rate", type=float, default=0.1, help="GA mutation rate")
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
    decay_v_value = args.decay

    background_rgb = hex_to_rgb(args.background_color)
    background = torch.tensor(background_rgb, dtype=torch.float32, device=device)

    material_colors, material_TDs, material_names, material_hex = load_materials(args.csv_file)
    material_colors = torch.tensor(material_colors, dtype=torch.float32, device=device)
    material_TDs = torch.tensor(material_TDs, dtype=torch.float32, device=device)
    material_count = material_colors.size(0)

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


    # For optimization, we use a default material assignment (e.g. all zeros)
    #global_material_indices = torch.zeros(max_layers_value, dtype=torch.int64)

    # Optimize pixel height logits using PyTorch.
    # best_pixel_height_logits, opt_loss = run_optimizer(
    #     target, pixel_height_logits_solver, global_material_indices,
    #     tau_init=1.0, h=h_value, max_layers=max_layers_value,
    #     material_colors=material_colors, material_TDs=material_TDs,
    #     background=background, num_iters=args.iterations,
    #     learning_rate=args.learning_rate, decay_v=decay_v_value,
    #     visualize=args.visualize
    # )
    # print("Optimization loss:", opt_loss)

    # Run the genetic algorithm to optimize material (color) swaps.
    best_global_material_indices, ga_fitness = genetic_algorithm_color_swaps(
        pixel_height_logits_solver, target, decay_v_value, h_value, max_layers_value,
        material_colors, material_TDs, background, args.num_color_swaps,
        material_count, population_size=args.ga_population,
        generations=args.ga_generations, mutation_rate=args.ga_mutation_rate,
        visualize=args.visualize, perceptual_module=perceptual_module
    )
    print("GA best fitness (loss):", ga_fitness)


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
