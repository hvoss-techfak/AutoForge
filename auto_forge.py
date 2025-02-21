#!/usr/bin/env python
"""
Script for generating 3D printed layered models from an input image.

This script uses a learned optimization with a Gumbel softmax formulation
to assign materials per layer and produce both a discretized composite that
is exported as an STL file along with swap instructions.
"""
import json
import os
import uuid

import configargparse
import cv2
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import pandas as pd

import os
import json
import pandas as pd
import numpy as np


def load_materials_data(csv_filename):
    """
    Load the full material data from the CSV file.
    Returns a list of dictionaries (one per material) with keys such as
    "Brand", "Type", "Color", "Name", "TD", "Owned", and "Uuid".
    """
    df = pd.read_csv(csv_filename)
    # Use a consistent key naming. For example, convert 'TD' to 'Transmissivity' and 'Uuid' to 'uuid'
    records = df.to_dict(orient="records")
    return records


def extract_filament_swaps(disc_global, disc_height_image, background_layers):
    """
    Given the discrete global material assignment (disc_global) and the discrete height image,
    extract the list of material indices (one per swap point) and the corresponding slider
    values (which indicate at which layer the material change occurs).

    This mimics the logic used in your generate_swap_instructions function.
    """
    # L is the total number of layers printed (maximum value in the height image)
    L = int(np.max(np.array(disc_height_image)))
    filament_indices = []
    slider_values = []
    prev = None
    for i in range(L):
        current = int(disc_global[i])
        # If this is the first layer or the material changes from the previous layer…
        if i == 0 or current != prev:
            # As in your swap instructions: the layer (1-indexed) is offset by the background layer count
            slider = i + background_layers
            slider_values.append(slider)
            filament_indices.append(current)
        prev = current
    return filament_indices, slider_values


def generate_project_file(project_filename, args, disc_global, disc_height_image,
                          width_mm, height_mm, stl_filename, csv_filename):
    """
    Export a project file containing the printing parameters, including:
      - Key dimensions and layer information (from your command-line args and computed outputs)
      - The filament_set: a list of filament definitions (each corresponding to a color swap)
        where the same material may be repeated if used at different swap points.
      - slider_values: a list of layer numbers (indices) where a filament swap occurs.

    The filament_set entries are built using the full material data from the CSV file.
    """
    # Compute the number of background layers (as in your main())
    background_layers = int(args.background_height / args.layer_height)

    # Load full material data from CSV
    material_data = load_materials_data(csv_filename)

    # Extract the swap points from the discrete solution
    filament_indices, slider_values = extract_filament_swaps(disc_global, disc_height_image, background_layers)

    # Build the filament_set list. For each swap point, we look up the corresponding material from CSV.
    # Here we map CSV columns to the project file’s expected keys.
    filament_set = []
    for idx in filament_indices:
        mat = material_data[idx]
        filament_entry = {
            "Brand": mat["Brand"],
            "Color": mat[" Color"],
            "Name": mat[" Name"],
            # Convert Owned to a boolean (in case it is read as a string)
            "Owned": str(mat[" Owned"]).strip().lower() == "true",
            "Transmissivity": float(mat[" TD"]) if not float(mat[" TD"]).is_integer() else int(mat[" TD"]),
            "Type": mat[" Type"],
            "uuid": mat[" Uuid"]
        }
        filament_set.append(filament_entry)

    # add black as the first filament with background height as the first slider value
    filament_set.insert(0, {
            "Brand": "Black",
            "Color": "#000000",
            "Name": "Black",
            "Owned": False,
            "Transmissivity": 0.1,
            "Type": "PLA",
            "uuid": str(uuid.uuid4())
    })
    #add black to slider value
    slider_values.insert(0, (args.background_height//args.layer_height)-1)

    # reverse order of filament set
    filament_set = filament_set[::-1]



    # Build the project file dictionary.
    # Many keys are filled in with default or derived values.
    project_data = {
        "base_layer_height": args.layer_height,  # you may adjust this if needed
        "blue_shift": 0,
        "border_height": args.background_height,  # here we use the background height
        "border_width": 3,
        "borderless": True,
        "bright_adjust_zero": False,
        "brightness_compensation_name": "Standard",
        "bw_tolerance": 8,
        "color_match_method": 0,
        "depth_mode": 2,
        "edit_image": False,
        "extra_gap": 2,
        "filament_set": filament_set,
        "flatten": False,
        "full_range": False,
        "green_shift": 0,
        "gs_threshold": 0,
        "height_in_mm": height_mm,
        "hsl_invert": False,
        "ignore_blue": False,
        "ignore_green": False,
        "ignore_red": False,
        "invert_blue": False,
        "invert_green": False,
        "invert_red": False,
        "inverted_color_pop": False,
        "layer_height": args.layer_height,
        "legacy_luminance": False,
        "light_intensity": -1,
        "light_temperature": 1,
        "lighting_visualizer": 0,
        "luminance_factor": 0,
        "luminance_method": 2,
        "luminance_offset": 0,
        "luminance_offset_max": 100,
        "luminance_power": 2,
        "luminance_weight": 100,
        "max_depth": args.background_height+args.layer_height*args.max_layers,
        "median": 0,
        "mesh_style_edit": True,
        "min_depth": 0.48,
        "min_detail": 0.2,
        "negative": True,
        "red_shift": 0,
        "reverse_litho": True,
        "slider_values": slider_values,
        "smoothing": 0,
        "srgb_linearize": False,
        "stl": os.path.basename(stl_filename),
        "strict_tolerance": False,
        "transparency": True,
        "version": "0.7.0",
        "width_in_mm": width_mm
    }

    # Write out the project file as JSON
    with open(project_filename, "w") as f:
        json.dump(project_data, f, indent=4)



def hex_to_rgb(hex_str):
    """
    Convert a hex color string to a normalized RGB list.
    """
    hex_str = hex_str.lstrip('#')
    return [int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)]


def load_materials(csv_filename):
    """
    Load material data from a CSV file.
    """
    df = pd.read_csv(csv_filename)
    material_names = [brand + " - " + name for brand, name in zip(df["Brand"].tolist(), df[" Name"].tolist())]
    material_TDs = (df[' TD'].astype(float)).to_numpy()
    colors_list = df[' Color'].tolist()
    # Use float64 for material colors.
    material_colors = jnp.array([hex_to_rgb(color) for color in colors_list], dtype=jnp.float64)
    material_TDs = jnp.array(material_TDs, dtype=jnp.float64)
    return material_colors, material_TDs, material_names, colors_list


def sample_gumbel(shape, key, eps=1e-20):
    """
    Sample from a Gumbel distribution.
    """
    U = random.uniform(key, shape=shape, minval=0.0, maxval=1.0)
    return -jnp.log(-jnp.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, key):
    """
    Sample from the Gumbel-Softmax distribution.
    """
    g = sample_gumbel(logits.shape, key)
    return jax.nn.softmax((logits + g) / temperature)


def gumbel_softmax(logits, temperature, key, hard=False):
    """
    Compute the Gumbel-Softmax.
    """
    y = gumbel_softmax_sample(logits, temperature, key)
    if hard:
        y_hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), logits.shape[-1])
        y = y_hard + jax.lax.stop_gradient(y - y_hard)
    return y


# ------------------ Color Conversion for Perceptual Loss ------------------

def srgb_to_linear_lab(rgb):
    """
    Convert sRGB (range [0,1]) to linear RGB.
    """
    return jnp.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_xyz(rgb_linear):
    """
    Convert linear RGB to XYZ using sRGB D65.
    """
    R = rgb_linear[..., 0]
    G = rgb_linear[..., 1]
    B = rgb_linear[..., 2]
    X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B
    Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
    Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B
    return jnp.stack([X, Y, Z], axis=-1)


def xyz_to_lab(xyz):
    """
    Convert XYZ to CIELAB. Assumes D65 reference white.
    This could become a problem, but I'll keep it for now.
    """
    # Reference white for D65:
    xyz_ref = jnp.array([0.95047, 1.0, 1.08883])
    xyz = xyz / xyz_ref
    delta = 6/29
    f = jnp.where(xyz > delta**3, xyz ** (1/3), (xyz / (3 * delta**2)) + (4/29))
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return jnp.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb):
    """
    Convert an sRGB image (values in [0,1]) to CIELAB.
    """
    rgb_linear = srgb_to_linear_lab(rgb)
    xyz = linear_to_xyz(rgb_linear)
    lab = xyz_to_lab(xyz)
    return lab


def adaptive_round(x, tau, high_tau=1.0, low_tau=0.01, temp=0.1):
    """
    Compute a soft (adaptive) rounding of x.

    When tau is high (>= high_tau) returns x (i.e. no rounding).
    When tau is low (<= low_tau) returns round(x).
    In between, linearly interpolates between x and round(x).
    """
    beta = jnp.clip((high_tau - tau) / (high_tau - low_tau), 0.0, 1.0)
    return (1 - beta) * x + beta * jnp.round(x)

def composite_pixel_combined(pixel_height_logit, global_logits, tau_height, tau_global,
                             h, max_layers, material_colors, material_TDs,
                             background, gumbel_keys, mode="continuous"):
    """
    Composite one pixel using either a continuous or discrete method,
    depending on the `mode` parameter.

    Args:
        pixel_height_logit: Raw logit for pixel height.
        global_logits: Global logits per layer for material selection.
        tau_height: Temperature parameter for height (soft printing).
        tau_global: Temperature parameter for material selection.
        h: Layer thickness.
        max_layers: Maximum number of layers.
        material_colors: Array of material colors.
        material_TDs: Array of material transmission/opacity parameters.
        background: Background color.
        gumbel_keys: Random keys for sampling in each layer.
        mode: "continuous" for soft compositing, "discrete" for hard discretization.

    Returns:
        Composite color for the pixel (scaled to [0,255]).
    """
    # Compute continuous pixel height (in physical units)
    pixel_height = (max_layers * h) * jax.nn.sigmoid(pixel_height_logit)
    # Continuous number of layers
    continuous_layers = pixel_height / h
    # Adaptive rounding: when tau_height is high, we get a soft round; when tau_height is low (<=0.01), we get hard rounding.
    adaptive_layers = adaptive_round(continuous_layers, tau_height, high_tau=0.1, low_tau=0.01, temp=0.1)
    # For the forward pass we want a crisp decision; however, we want to use gradients from the adaptive value.
    discrete_layers = jnp.round(continuous_layers) + jax.lax.stop_gradient(
        adaptive_layers - jnp.round(continuous_layers))
    discrete_layers = discrete_layers.astype(jnp.int32)

    # Parameters for opacity calculation.
    A = 0.1490
    k = 94.9845
    b = 0.3423

    def step_fn(carry, i):
        comp, remaining = carry
        # Process layers from top (last layer) to bottom (first layer)
        j = max_layers - 1 - i

        # Use a crisp (binary) decision:
        p_print = jnp.where(j < discrete_layers, 1.0, 0.0)
        eff_thick = p_print * h

        # For material selection, force a one-hot (hard) result when tau_global is very small.
        p_i = jax.lax.cond(
            tau_global < 1e-3,
            lambda _: gumbel_softmax(global_logits[j], tau_global, gumbel_keys[j], hard=True),
            lambda _: gumbel_softmax(global_logits[j], tau_global, gumbel_keys[j], hard=False),
            operand=None
        )
        color_i = jnp.dot(p_i, material_colors)
        TD_i = jnp.dot(p_i, material_TDs) * 0.1

        # Compute opacity
        opac = A * jnp.log(1 + k * (eff_thick / TD_i)) + b * (eff_thick / TD_i)
        opac = jnp.clip(opac, 0.0, 1.0)
        new_comp = comp + remaining * opac * color_i
        new_remaining = remaining * (1 - opac)
        return (new_comp, new_remaining), None

    init_state = (jnp.zeros(3), 1.0)
    (comp, remaining), _ = jax.lax.scan(step_fn, init_state, jnp.arange(max_layers))
    result = comp + remaining * background
    return result * 255.0


def composite_image_combined(pixel_height_logits, global_logits, tau_height, tau_global, gumbel_keys,
                             h, max_layers, material_colors, material_TDs, background, mode="continuous"):
    """
    Apply composite_pixel_combined over the entire image.

    Args:
        pixel_height_logits: 2D array of pixel height logits.
        global_logits: Global logits for each layer.
        tau_height: Temperature for height compositing.
        tau_global: Temperature for material selection.
        gumbel_keys: Random keys per layer.
        h: Layer thickness.
        max_layers: Maximum number of layers.
        material_colors: Array of material colors.
        material_TDs: Array of material transmission/opacity parameters.
        background: Background color.
        mode: "continuous" or "discrete".

    Returns:
        The composite image (with values scaled to [0,255]).
    """
    return jax.vmap(jax.vmap(
        lambda ph_logit: composite_pixel_combined(
            ph_logit, global_logits, tau_height, tau_global, h, max_layers,
            material_colors, material_TDs, background, gumbel_keys, mode
        )
    ))(pixel_height_logits)


composite_image_combined_jit = jax.jit(composite_image_combined, static_argnums=(5,6,10))


def loss_fn(params, target, tau_height, tau_global, gumbel_keys, h, max_layers, material_colors, material_TDs, background):
    """
    Compute the mean squared error loss between the composite and target images.
    By default, we use continuous (soft) compositing.
    """
    comp = composite_image_combined_jit(params['pixel_height_logits'], params['global_logits'],
                                        tau_height, tau_global, gumbel_keys,
                                        h, max_layers, material_colors, material_TDs, background, mode="continuous")
    return jnp.mean((comp - target) ** 2)


def create_update_step(optimizer, loss_function, h, max_layers, material_colors, material_TDs, background):
    """
    Create a JIT-compiled update step function using the specified loss function.
    """
    @jax.jit
    def update_step(params, target, tau_height, tau_global, gumbel_keys, opt_state):

        loss_val, grads = jax.value_and_grad(loss_function)(
            params, target, tau_height, tau_global, gumbel_keys,
            h, max_layers, material_colors, material_TDs, background)

        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss_val
    return update_step


def discretize_solution_jax(params, tau_global, gumbel_keys, h, max_layers):
    """
    Discretize the continuous pixel height logits into integer layer counts,
    and force hard material selections.
    """
    pixel_height_logits = params['pixel_height_logits']
    global_logits = params['global_logits']
    pixel_heights = (max_layers * h) * jax.nn.sigmoid(pixel_height_logits)
    discrete_height_image = jnp.round(pixel_heights / h).astype(jnp.int32)
    discrete_height_image = jnp.clip(discrete_height_image, 0, max_layers)

    def discretize_layer(logits, key):
        p = gumbel_softmax(logits, tau_global, key, hard=True)
        return jnp.argmax(p)
    discrete_global = jax.vmap(discretize_layer)(global_logits, gumbel_keys)
    return discrete_global, discrete_height_image

def initialize_pixel_height_logits(target):
    """
    Initialize pixel height logits based on the luminance of the target image.

    Assumes target is a jnp.array of shape (H, W, 3) in the range [0, 255].
    Uses the formula: L = 0.299*R + 0.587*G + 0.114*B.
    """
    # Compute normalized luminance in [0,1]
    normalized_lum = (0.299 * target[..., 0] +
                      0.587 * target[..., 1] +
                      0.114 * target[..., 2]) / 255.0
    # To avoid log(0) issues, add a small epsilon.
    eps = 1e-6
    # Convert normalized luminance to logits using the inverse sigmoid (logit) function.
    # This ensures that jax.nn.sigmoid(pixel_height_logits) approximates normalized_lum.
    pixel_height_logits = jnp.log((normalized_lum + eps) / (1 - normalized_lum + eps))
    return pixel_height_logits

def run_optimizer(rng_key, target, H, W, max_layers, h, material_colors, material_TDs, background,
                  num_iters, learning_rate, decay_v, loss_function, visualize=False, save_max_tau=0.001):
    """
    Run the optimization loop to learn per-pixel heights and per-layer material assignments.
    """
    num_materials = material_colors.shape[0]
    rng_key, subkey = random.split(rng_key)
    # Initialize global_logits with a base bias and distinct per-layer bias.
    global_logits = jnp.ones((max_layers, num_materials)) * -1.0
    for i in range(max_layers):
        global_logits = global_logits.at[i, i % num_materials].set(1.0)
    global_logits += random.uniform(subkey, global_logits.shape, minval=-0.1, maxval=0.1)

    rng_key, subkey = random.split(rng_key)

    pixel_height_logits = initialize_pixel_height_logits(target)
    #pixel_height_logits = random.uniform(subkey, (H, W), minval=-2.0, maxval=2.0)

    params = {'global_logits': global_logits, 'pixel_height_logits': pixel_height_logits}

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    update_step = create_update_step(optimizer, loss_function, h, max_layers, material_colors, material_TDs, background)

    warmup_steps = num_iters // 4
    decay_rate = -math.log(decay_v) / (num_iters - warmup_steps)

    def get_tau(i, tau_init=1.0, tau_final=decay_v, decay_rate=decay_rate):
        if i < warmup_steps:
            return tau_init
        else:
            return max(tau_final, tau_init * math.exp(-decay_rate * (i - warmup_steps)))

    best_params = None
    best_loss = float('inf')

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(1, 5, figsize=(17, 6))
        target_im = ax[0].imshow(np.array(target, dtype=np.uint8))
        ax[0].set_title("Target Image")
        comp_im = ax[1].imshow(np.zeros((H, W, 3), dtype=np.uint8))
        ax[1].set_title("Current Composite (Continuous)")
        best_comp_im = ax[2].imshow(np.zeros((H, W, 3), dtype=np.uint8))
        ax[2].set_title("Best Composite (Continuous)")
        height_map_im = ax[3].imshow(np.zeros((H, W)), cmap='viridis')
        height_map_im.set_clim(0, max_layers * h)
        ax[3].set_title("Height Map")
        disc_comp_im = ax[4].imshow(np.zeros((H, W, 3), dtype=np.uint8))
        ax[4].set_title("Composite (Discrete)")
        plt.pause(0.1)

    saved_new_tau = False
    tbar = tqdm(range(num_iters))
    for i in tbar:
        tau_height = get_tau(i, tau_init=1.0, tau_final=decay_v, decay_rate=decay_rate)
        tau_global = get_tau(i, tau_init=1.0, tau_final=decay_v, decay_rate=decay_rate)
        rng_key, subkey = random.split(rng_key)
        gumbel_keys = random.split(subkey, max_layers)
        params, opt_state, loss_val = update_step(params, target, tau_height, tau_global, gumbel_keys, opt_state)
        save_tau_bool = (tau_global < save_max_tau and not saved_new_tau)
        if loss_val < best_loss or save_tau_bool:
            if save_tau_bool:
                saved_new_tau = True
            best_loss = loss_val
            best_params = {k: jnp.array(v) for k, v in params.items()}
            if visualize:
                comp = composite_image_combined_jit(best_params['pixel_height_logits'], best_params['global_logits'],
                                                      tau_height, tau_global, gumbel_keys,
                                                      h, max_layers, material_colors, material_TDs, background, mode="continuous")
                comp_np = np.clip(np.array(comp), 0, 255).astype(np.uint8)
                best_comp_im.set_data(comp_np)
                # For discrete composite visualization, force discrete mode.
                disc_comp = composite_image_combined_jit(best_params['pixel_height_logits'], best_params['global_logits'],
                                                         tau_height, tau_global, gumbel_keys,
                                                         h, max_layers, material_colors, material_TDs, background, mode="discrete")
                disc_comp_np = np.clip(np.array(disc_comp), 0, 255).astype(np.uint8)
                disc_comp_im.set_data(disc_comp_np)
        if visualize and (i % 50 == 0):
            comp = composite_image_combined_jit(params['pixel_height_logits'], params['global_logits'],
                                                tau_height, tau_global, gumbel_keys,
                                                h, max_layers, material_colors, material_TDs, background, mode="continuous")
            comp_np = np.clip(np.array(comp), 0, 255).astype(np.uint8)
            comp_im.set_data(comp_np)
            # Update height map.
            height_map = (max_layers * h) * jax.nn.sigmoid(best_params['pixel_height_logits'])
            height_map_np = np.array(height_map)
            height_map_im.set_data(height_map_np)
            highest_layer = np.max(height_map_np)
            fig.suptitle(f"Iteration {i}, Loss: {loss_val:.4f}, Best Loss: {best_loss:.4f}, Tau: {tau_height:.3f}, Highest Layer: {highest_layer:.2f}mm")
            plt.pause(0.01)
        tbar.set_description(f"loss = {loss_val:.4f}, Best Loss = {best_loss:.4f}")

    if visualize:
        plt.ioff()
        plt.close()
    best_comp = composite_image_combined_jit(best_params['pixel_height_logits'], best_params['global_logits'],
                                              tau_height, tau_global, gumbel_keys,
                                              h, max_layers, material_colors, material_TDs, background, mode="continuous")
    return best_params, best_comp


import struct

def generate_stl(height_map, filename, background_height, scale=1.0):
    """
    Generate a binary STL file from a height map.
    """
    H, W = height_map.shape
    vertices = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            # Original coordinates: x = j*scale, y = (H - 1 - i), z = height + background
            vertices[i, j, 0] = j * scale
            vertices[i, j, 1] = (H - 1 - i)  # (Consider applying scale if needed)
            vertices[i, j, 2] = height_map[i, j] + background_height

    triangles = []

    def add_triangle(v1, v2, v3):
        triangles.append((v1, v2, v3))

    for i in range(H - 1):
        for j in range(W - 1):
            v0 = vertices[i, j]
            v1 = vertices[i, j + 1]
            v2 = vertices[i + 1, j + 1]
            v3 = vertices[i + 1, j]
            # Reversed order so normals face upward
            add_triangle(v2, v1, v0)
            add_triangle(v3, v2, v0)

    for j in range(W - 1):
        v0 = vertices[0, j]
        v1 = vertices[0, j + 1]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v0, v1, v1b)
        add_triangle(v0, v1b, v0b)
    for j in range(W - 1):
        v0 = vertices[H - 1, j]
        v1 = vertices[H - 1, j + 1]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v1, v0, v1b)
        add_triangle(v0, v0b, v1b)
    for i in range(H - 1):
        v0 = vertices[i, 0]
        v1 = vertices[i + 1, 0]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v1, v0, v1b)
        add_triangle(v0, v0b, v1b)
    for i in range(H - 1):
        v0 = vertices[i, W - 1]
        v1 = vertices[i + 1, W - 1]
        v0b = np.array([v0[0], v0[1], 0], dtype=np.float32)
        v1b = np.array([v1[0], v1[1], 0], dtype=np.float32)
        add_triangle(v0, v1, v1b)
        add_triangle(v0, v1b, v0b)

    v0 = np.array([0, 0, 0], dtype=np.float32)
    v1 = np.array([(W - 1) * scale, 0, 0], dtype=np.float32)
    v2 = np.array([(W - 1) * scale, (H - 1) * scale, 0], dtype=np.float32)
    v3 = np.array([0, (H - 1) * scale, 0], dtype=np.float32)
    add_triangle(v2, v1, v0)
    add_triangle(v3, v2, v0)

    num_triangles = len(triangles)

    # Write the binary STL file.
    with open(filename, 'wb') as f:
        header_str = "Binary STL generated from heightmap"
        header = header_str.encode('utf-8')
        header = header.ljust(80, b' ')
        f.write(header)
        f.write(struct.pack('<I', num_triangles))
        for tri in triangles:
            v1, v2, v3 = tri
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            if norm == 0:
                normal = np.array([0, 0, 0], dtype=np.float32)
            else:
                normal = normal / norm
            f.write(struct.pack('<12fH',
                                  normal[0], normal[1], normal[2],
                                  v1[0], v1[1], v1[2],
                                  v2[0], v2[1], v2[2],
                                  v3[0], v3[1], v3[2],
                                  0))




def generate_swap_instructions(discrete_global, discrete_height_image, h, background_layers, background_height, material_names):
    """
    Generate swap instructions based on discrete material assignments.
    """
    L = int(np.max(np.array(discrete_height_image)))
    instructions = []
    if L == 0:
        instructions.append("No layers printed.")
        return instructions
    instructions.append("Start with your background color")
    for i in range(0, L):
        if i == 0 or int(discrete_global[i]) != int(discrete_global[i - 1]):
            ie = i + 1
            instructions.append(f"At layer #{ie + background_layers} ({(ie * h) + background_height:.2f}mm) swap to {material_names[int(discrete_global[i])]}")
    instructions.append("For the rest, use " + material_names[int(discrete_global[L - 1])])
    return instructions

def main():
    parser = configargparse.ArgParser()
    parser.add_argument("--config", is_config_file=True, help="Path to config file")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with material data")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to write outputs")
    parser.add_argument("--iterations", type=int, default=20000, help="Number of optimization iterations")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Learning rate for optimization")
    parser.add_argument("--layer_height", type=float, default=0.04, help="Layer thickness in mm")
    parser.add_argument("--max_layers", type=int, default=75, help="Maximum number of layers")
    parser.add_argument("--background_height", type=float, default=0.4, help="Height of the background in mm")
    parser.add_argument("--background_color", type=str, default="#8e9089", help="Background color")
    parser.add_argument("--max_size", type=int, default=512, help="Maximum dimension for target image")
    parser.add_argument("--save_max_tau", type=float, default=0.005, help="Tau threshold to save best result")
    parser.add_argument("--decay", type=float, default=0.0001, help="Final tau value for Gumbel-Softmax")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization during optimization")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    assert (args.background_height / args.layer_height).is_integer(), "Background height must be divisible by layer height."
    assert args.save_max_tau > args.decay, "save_max_tau must be greater than decay."
    assert args.max_size > 0, "max_size must be positive."
    assert args.iterations > 0, "iterations must be positive."
    assert args.learning_rate > 0, "learning_rate must be positive."
    assert args.layer_height > 0, "layer_height must be positive."

    h_value = args.layer_height
    max_layers_value = args.max_layers
    background_height_value = args.background_height
    background_layers_value = background_height_value // h_value
    decay_v_value = args.decay

    background = jnp.array(hex_to_rgb(args.background_color), dtype=jnp.float64)
    material_colors, material_TDs, material_names, material_hex = load_materials(args.csv_file)

    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img.shape
    if w_img >= h_img:
        new_w = args.max_size
        new_h = int(args.max_size * h_img / w_img)
    else:
        new_h = args.max_size
        new_w = int(args.max_size * w_img / h_img)
    target = cv2.resize(img, (new_w, new_h))
    target = jnp.array(target, dtype=jnp.float64)

    rng_key = random.PRNGKey(0)
    best_params, _ = run_optimizer(rng_key, target, new_h, new_w, max_layers_value, h_value,
                                   material_colors, material_TDs, background,
                                   args.iterations, args.learning_rate, decay_v_value,
                                   loss_function=loss_fn,
                                   visualize=args.visualize,
                                   save_max_tau=args.save_max_tau)

    # Generate discrete outputs for STL and instructions.
    rng_key, subkey = random.split(rng_key)
    gumbel_keys_disc = random.split(subkey, max_layers_value)
    tau_global_disc = decay_v_value
    disc_global, disc_height_image = discretize_solution_jax(best_params, tau_global_disc, gumbel_keys_disc, h_value, max_layers_value)
    # Use the combined composite function in discrete mode for visualization.
    disc_comp = composite_image_combined_jit(best_params['pixel_height_logits'], best_params['global_logits'],
                                             tau_global_disc, tau_global_disc, gumbel_keys_disc,
                                             h_value, max_layers_value, material_colors, material_TDs, background, mode="discrete")
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
