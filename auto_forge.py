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



def srgb_to_linear(c):
    """
    Convert sRGB color to linear color.
    Assumes c is in [0,1].
    """
    return jnp.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(c):
    """
    Convert linear color to sRGB.
    Assumes c is in [0,1].
    """
    return jnp.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1/2.4)) - 0.055)




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
    material_TDs = df[' TD'].astype(float).to_numpy()
    colors_list = df[' Color'].tolist()
    material_colors = jnp.array([hex_to_rgb(color) for color in colors_list], dtype=jnp.float32)
    return material_colors, material_TDs, material_names,colors_list


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


# ------------------ Compositing Functions ------------------

def composite_pixel_tempered(pixel_height_logit, global_logits, tau_height, tau_global, h, max_layers,
                             material_colors, material_TDs, background, gumbel_keys):
    pixel_height = (max_layers * h) * jax.nn.sigmoid(pixel_height_logit)

    # Convert sRGB colors to linear space for proper blending.
    material_colors_linear = srgb_to_linear(material_colors)
    background_linear = srgb_to_linear(background)

    def step_fn(carry, i):
        comp, remaining = carry
        j = max_layers - 1 - i  # process from top to bottom
        p_print = jax.nn.sigmoid((pixel_height - j * h) / tau_height)
        eff_thick = p_print * h
        p_i = gumbel_softmax(global_logits[j], tau_global, gumbel_keys[j], hard=False)
        color_linear = jnp.dot(p_i, material_colors_linear)
        TD_i = jnp.dot(p_i, material_TDs)
        opac = 1.0 - jnp.exp(-46.05 * (eff_thick / TD_i))
        new_comp = comp + remaining * opac * color_linear
        new_remaining = remaining * (1 - opac)
        return (new_comp, new_remaining), None

    init_state = (jnp.zeros(3), 1.0)
    (comp, remaining), _ = jax.lax.scan(step_fn, init_state, jnp.arange(max_layers))

    # Composite against the background in linear space.
    result_linear = comp + remaining * background_linear

    # Convert the result back to sRGB.
    result_srgb = linear_to_srgb(result_linear)
    return result_srgb * 255.0


def composite_image_tempered_fn(pixel_height_logits, global_logits, tau_height, tau_global, gumbel_keys,
                                h, max_layers, material_colors, material_TDs, background):
    """
    Composite an entire image using tempered Gumbel compositing.

    """
    return jax.vmap(jax.vmap(
        lambda ph_logit: composite_pixel_tempered(ph_logit, global_logits, tau_height, tau_global,
                                                  h, max_layers, material_colors, material_TDs, background, gumbel_keys)
    ))(pixel_height_logits)


# Apply jit with static_argnums for the static argument "max_layers" (index 6)
composite_image_tempered_fn = jax.jit(composite_image_tempered_fn, static_argnums=(6,))


def loss_fn(params, target, tau_height, tau_global, gumbel_keys, h, max_layers, material_colors, material_TDs, background):
    """
    Compute the mean squared error loss between the composite and target images (in sRGB).
    """
    comp = composite_image_tempered_fn(params['pixel_height_logits'], params['global_logits'],
                                       tau_height, tau_global, gumbel_keys,
                                       h, max_layers, material_colors, material_TDs, background)
    return jnp.mean((comp - target) ** 2)


def loss_fn_perceptual(params, target, tau_height, tau_global, gumbel_keys, h, max_layers, material_colors, material_TDs, background):
    """
    Compute a perceptual loss between the composite and target images.

    Both images are normalized to [0,1], converted to CIELAB, and then the MSE is computed.
    """
    comp = composite_image_tempered_fn(params['pixel_height_logits'], params['global_logits'],
                                       tau_height, tau_global, gumbel_keys,
                                       h, max_layers, material_colors, material_TDs, background)
    comp_norm = comp / 255.0
    target_norm = target / 255.0
    comp_lab = rgb_to_lab(comp_norm)
    target_lab = rgb_to_lab(target_norm)
    return jnp.mean((comp_lab - target_lab) ** 2)

def huber_loss(pred, target, delta=0.1):
    """
    Compute the Huber loss between predictions and targets.

    Parameters:
        pred (jnp.array): Predicted values.
        target (jnp.array): Ground-truth values.
        delta (float): Threshold at which to change between quadratic and linear loss.

    Returns:
        jnp.array: The Huber loss.
    """
    error = pred - target
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return jnp.mean(0.5 * quadratic**2 + delta * linear)

def loss_fn_perceptual_l1(params, target, tau_height, tau_global, gumbel_keys, h, max_layers, material_colors, material_TDs, background):
    """
    Compute a perceptual loss between the composite and target images.

    Both images are normalized to [0,1], converted to CIELAB, and then the MSE is computed.
    """
    comp = composite_image_tempered_fn(params['pixel_height_logits'], params['global_logits'],
                                       tau_height, tau_global, gumbel_keys,
                                       h, max_layers, material_colors, material_TDs, background)
    comp_norm = comp / 255.0
    target_norm = target / 255.0
    comp_lab = rgb_to_lab(comp_norm)
    target_lab = rgb_to_lab(target_norm)
    return huber_loss(comp_lab, target_lab, delta=1.0)


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
    pixel_height_logits = params['pixel_height_logits']
    global_logits = params['global_logits']
    pixel_heights = (max_layers * h) * jax.nn.sigmoid(pixel_height_logits)
    # Use floor instead of ceil to avoid counting a partially reached layer.
    discrete_height_image = jnp.floor(pixel_heights / h).astype(jnp.int32)
    discrete_height_image = jnp.clip(discrete_height_image, 0, max_layers)

    def discretize_layer(logits, key):
        p = gumbel_softmax(logits, tau_global, key, hard=True)
        return jnp.argmax(p)

    discrete_global = jax.vmap(discretize_layer)(global_logits, gumbel_keys)
    return discrete_global, discrete_height_image


def composite_image_discrete_jax(discrete_height_image, discrete_global, h, max_layers, mat_colors, mat_TDs,
                                 background):
    # Convert sRGB colors to linear space.
    mat_colors_linear = srgb_to_linear(mat_colors)
    background_linear = srgb_to_linear(background)

    def composite_pixel(pixel_printed_layers):
        def step_fn(carry, l):
            comp, remaining = carry
            idx = max_layers - 1 - l
            do_layer = idx < pixel_printed_layers

            def true_fn(carry):
                comp, remaining = carry
                mat_idx = discrete_global[idx]
                color_linear = mat_colors_linear[mat_idx]
                TD = mat_TDs[mat_idx]
                opac = 1.0 - jnp.exp(-46.05 * (h / TD))
                new_comp = comp + remaining * opac * color_linear
                new_remaining = remaining * (1 - opac)
                return (new_comp, new_remaining)

            new_carry = jax.lax.cond(do_layer, true_fn, lambda c: c, (comp, remaining))
            return new_carry, None

        init_state = (jnp.zeros(3), 1.0)
        (comp, remaining), _ = jax.lax.scan(step_fn, init_state, jnp.arange(max_layers))
        result_linear = comp + remaining * background_linear
        result_srgb = linear_to_srgb(result_linear)
        return result_srgb * 255.0

    return jax.vmap(jax.vmap(composite_pixel))(discrete_height_image)

# Apply jit with static_argnums for "max_layers" (argument index 3)
composite_image_discrete_jax = jax.jit(composite_image_discrete_jax, static_argnums=(3,))


def run_optimizer(rng_key, target, H, W, max_layers, h, material_colors, material_TDs, background,
                  num_iters, learning_rate, decay_v, loss_function, visualize=False,save_max_tau=0.1):
    """
    Run the optimization loop to learn per-pixel heights and per-layer material assignments.
    """
    num_materials = material_colors.shape[0]
    rng_key, subkey = random.split(rng_key)
    global_logits = random.normal(subkey, (max_layers, num_materials)) * 0.1
    rng_key, subkey = random.split(rng_key)
    pixel_height_logits = random.normal(subkey, (H, W)) * 0.1
    params = {'global_logits': global_logits, 'pixel_height_logits': pixel_height_logits}

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    update_step = create_update_step(optimizer, loss_function, h, max_layers, material_colors, material_TDs, background)

    decay_rate = -math.log(decay_v) / num_iters

    def get_tau(i, tau_init=1.0, tau_final=decay_v, decay_rate=decay_rate):
        return max(tau_final, tau_init * math.exp(-decay_rate * i))

    best_params = None
    best_loss = float('inf')

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(1, 4, figsize=(17, 6))
        target_im = ax[0].imshow(np.array(target, dtype=np.uint8))
        ax[0].set_title("Target Image")
        comp_im = ax[1].imshow(np.zeros((H, W, 3), dtype=np.uint8))
        ax[1].set_title("Current Gumbel Composite")
        best_comp_im = ax[2].imshow(np.zeros((H, W, 3), dtype=np.uint8))
        ax[2].set_title("Best Gumbel Composite")
        disc_comp_im = ax[3].imshow(np.zeros((H, W, 3), dtype=np.uint8))
        ax[3].set_title("Discretized Composite")
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
                comp = composite_image_tempered_fn(best_params['pixel_height_logits'], best_params['global_logits'],
                                                   tau_height, tau_global, gumbel_keys,
                                                   h, max_layers, material_colors, material_TDs, background)
                comp_np = np.clip(np.array(comp), 0, 255).astype(np.uint8)
                best_comp_im.set_data(comp_np)
                rng_key, subkey = random.split(rng_key)
                gumbel_keys_disc = random.split(subkey, max_layers)
                disc_global, disc_height_image = discretize_solution_jax(best_params, decay_v, gumbel_keys_disc, h, max_layers)
                disc_comp = composite_image_discrete_jax(disc_height_image, disc_global, h, max_layers,
                                                         material_colors, material_TDs, background)
                disc_comp_np = np.clip(np.array(disc_comp), 0, 255).astype(np.uint8)
                disc_comp_im.set_data(disc_comp_np)
        if visualize and (i % 50 == 0):
            comp = composite_image_tempered_fn(params['pixel_height_logits'], params['global_logits'],
                                               tau_height, tau_global, gumbel_keys,
                                               h, max_layers, material_colors, material_TDs, background)
            comp_np = np.clip(np.array(comp), 0, 255).astype(np.uint8)
            comp_im.set_data(comp_np)
            actual_layer_height = (max_layers * h) * jax.nn.sigmoid(best_params['pixel_height_logits'])
            highest_layer = np.max(np.array(actual_layer_height))
            fig.suptitle(f"Iteration {i}, Loss: {loss_val:.4f}, Best Loss: {best_loss:.4f}, Tau: {tau_height:.3f}, Highest Layer: {highest_layer:.2f}mm")
            plt.pause(0.01)
        tbar.set_description(f"loss = {loss_val:.4f}, Best Loss = {best_loss:.4f}")

    if visualize:
        plt.ioff()
        plt.close()
    best_comp = composite_image_tempered_fn(best_params['pixel_height_logits'], best_params['global_logits'],
                                            tau_height, tau_global, gumbel_keys,
                                            h, max_layers, material_colors, material_TDs, background)
    return best_params, best_comp


def generate_stl(height_map, filename, background_height, scale=1.0):
    """
    Generate an ASCII STL file from a height map.

    """
    H, W = height_map.shape
    vertices = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            # Original coordinates: x = j*scale, y = i*scale, z = height + background
            vertices[i, j, 0] = j * scale
            vertices[i, j, 1] = i * scale
            vertices[i, j, 2] = height_map[i, j] + background_height

    triangles = []

    def add_triangle(v1, v2, v3):
        triangles.append((v1, v2, v3))

    # Top surface (each grid cell as two triangles)
    for i in range(H - 1):
        for j in range(W - 1):
            v0 = vertices[i, j]
            v1 = vertices[i, j + 1]
            v2 = vertices[i + 1, j + 1]
            v3 = vertices[i + 1, j]
            add_triangle(v0, v1, v2)
            add_triangle(v0, v2, v3)

    # Walls along the boundaries:
    for j in range(W - 1):
        v0 = vertices[0, j]
        v1 = vertices[0, j + 1]
        v0b = np.array([v0[0], v0[1], 0])
        v1b = np.array([v1[0], v1[1], 0])
        add_triangle(v0, v1, v1b)
        add_triangle(v0, v1b, v0b)
    for j in range(W - 1):
        v0 = vertices[H - 1, j]
        v1 = vertices[H - 1, j + 1]
        v0b = np.array([v0[0], v0[1], 0])
        v1b = np.array([v1[0], v1[1], 0])
        add_triangle(v1, v0, v1b)
        add_triangle(v0, v0b, v1b)
    for i in range(H - 1):
        v0 = vertices[i, 0]
        v1 = vertices[i + 1, 0]
        v0b = np.array([v0[0], v0[1], 0])
        v1b = np.array([v1[0], v1[1], 0])
        add_triangle(v1, v0, v1b)
        add_triangle(v0, v0b, v1b)
    for i in range(H - 1):
        v0 = vertices[i, W - 1]
        v1 = vertices[i + 1, W - 1]
        v0b = np.array([v0[0], v0[1], 0])
        v1b = np.array([v1[0], v1[1], 0])
        add_triangle(v0, v1, v1b)
        add_triangle(v0, v1b, v0b)

    # Bottom face
    v0 = np.array([0, 0, 0])
    v1 = np.array([(W - 1) * scale, 0, 0])
    v2 = np.array([(W - 1) * scale, (H - 1) * scale, 0])
    v3 = np.array([0, (H - 1) * scale, 0])
    add_triangle(v0, v1, v2)
    add_triangle(v0, v2, v3)

    with open(filename, 'w') as f:
        f.write("solid heightmap\n")
        for tri in triangles:
            v1, v2, v3 = tri
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            if norm == 0:
                normal = np.array([0, 0, 0])
            else:
                normal = normal / norm
            f.write("  facet normal {} {} {}\n".format(normal[0], normal[1], normal[2]))
            f.write("    outer loop\n")
            f.write("      vertex {} {} {}\n".format(v1[0], v1[1], v1[2]))
            f.write("      vertex {} {} {}\n".format(v2[0], v2[1], v2[2]))
            f.write("      vertex {} {} {}\n".format(v3[0], v3[1], v3[2]))
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid heightmap\n")


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


def generate_project_file(project_filename, args, disc_global, disc_height_image,
                          image_width_mm, image_height_mm,
                          stl_filename, material_names, material_TDs, material_hex):
    """
    Generate a project file (JSON) that follows the expected external program format.

    This function sets a number of fields from args and our code defaults.
    In particular, it builds:

      - slider_values: a list of 1-indexed printed layer positions where the material changes.
      - filament_set: a list of filaments (one per slider event) corresponding to the material
        that is applied at that layer. Duplicates are allowed if the same filament is used in multiple
        positions.

    Parameters:
        project_filename (str): Path to write the JSON project file.
        args: The argparse namespace (contains layer_height, background_height, etc.).
        disc_global (array-like): 1D array (length max_layers) of material indices per layer.
        disc_height_image (array-like): 2D array of printed layer counts (per pixel); its maximum
            determines the number of printed layers (L).
        image_width_mm (float): Printed object width in millimeters.
        image_height_mm (float): Printed object height (printed region, excluding background) in mm.
        stl_filename (str): Path/filename of the generated STL.
        material_names (list): List of material names (from CSV).
        material_TDs (list/array): Material transmissivity values.
        material_hex (list): List of hex color strings for each material.
    """
    project = {}

    # Basic settings
    project["version"] = "0.7.0"
    project["layer_height"] = args.layer_height
    project["base_layer_height"] = args.background_height  # background as base layer height
    project["border_height"] = args.background_height
    project["border_width"] = 3
    project["borderless"] = True
    project["bright_adjust_zero"] = False
    project["brightness_compensation_name"] = "Standard"
    project["bw_tolerance"] = 8
    project["color_match_method"] = 0
    project["depth_mode"] = 2
    project["edit_image"] = False
    project["extra_gap"] = 2

    # Determine printed layer count L (only layers 0 to L-1 are printed).
    L = int(np.max(np.array(disc_height_image)))

    # Build slider_values and filament_set.
    # We assume that disc_global is ordered from bottom (layer 0) to top (layer max_layers-1).
    slider_values = []
    filament_set = []

    if L > 0:
        # Always add the first printed layer.
        slider_values.append(1)  # 1-indexed
        # Record the filament used at layer 0.
        mat_idx = int(disc_global[0])
        filament_set.append({
            "Brand": "BambuLab Basic",  # or load from CSV if available
            "Color": material_hex[mat_idx] if mat_idx < len(material_hex) else "#000000",
            "Name": material_names[mat_idx],
            "Owned": True,
            "Transmissivity": float(material_TDs[mat_idx]),
            "Type": "PLA",
            "uuid": str(uuid.uuid4())
        })
        # For each subsequent printed layer (from layer 1 to L-1) add a slider event if the material changes.
        for i in range(1, L):
            if disc_global[i] != disc_global[i - 1]:
                slider_values.append(i + 1)  # use 1-indexing
                mat_idx = int(disc_global[i])
                filament_set.append({
                    "Brand": "BambuLab Basic",
                    "Color": material_hex[mat_idx] if mat_idx < len(material_hex) else "#000000",
                    "Name": material_names[mat_idx],
                    "Owned": True,
                    "Transmissivity": float(material_TDs[mat_idx]),
                    "Type": "PLA",
                    "uuid": str(uuid.uuid4())
                })

    project["slider_values"] = slider_values
    project["filament_set"] = filament_set

    # Other settings
    project["flatten"] = False
    project["full_range"] = True
    project["green_shift"] = 0
    project["gs_threshold"] = 0
    project["width_in_mm"] = float(image_width_mm)
    # Total printed height includes the printed part plus the background.
    project["height_in_mm"] = float(image_height_mm) + args.background_height
    project["hsl_invert"] = False
    project["ignore_blue"] = False
    project["ignore_green"] = False
    project["ignore_red"] = False
    project["invert_blue"] = False
    project["invert_green"] = False
    project["invert_red"] = False
    project["inverted_color_pop"] = False
    project["legacy_luminance"] = False
    project["light_intensity"] = -1
    project["light_temperature"] = 1
    project["lighting_visualizer"] = 0
    project["luminance_factor"] = 0
    project["luminance_method"] = 2
    project["luminance_offset"] = 0
    project["luminance_offset_max"] = 100
    project["luminance_power"] = 2
    project["luminance_weight"] = 100
    project["max_depth"] = args.background_height
    project["median"] = 0
    project["mesh_style_edit"] = True
    project["min_depth"] = args.background_height / 2  # adjust as needed
    project["min_detail"] = 0.2
    project["negative"] = True
    project["red_shift"] = 0
    project["reverse_litho"] = True
    project["smoothing"] = 0
    project["srgb_linearize"] = False
    project["stl"] = stl_filename
    project["strict_tolerance"] = False
    project["transparency"] = True

    # Write out the JSON file.
    with open(project_filename, "w") as f:
        json.dump(project, f, indent=4)


def main():
    """
    Main function to run the optimization and generate outputs.
    """
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
    parser.add_argument("--save_max_tau", type=float, default=0.05, help="We start to save the best result after this tau value, to ensure convergence and color separation")
    parser.add_argument("--decay", type=float, default=0.005, help="Final tau value for Gumbel-Softmax")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "perceptual","perceptual_l1"], help="Loss function to use")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization during optimization")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Ensure background height is divisible by layer height.
    assert (args.background_height / args.layer_height).is_integer(), "Background height must be divisible by layer height."
    assert args.save_max_tau > args.decay, "save_max_tau must be less than decay."
    assert args.max_size > 0, "max_size must be positive."
    assert args.iterations > 0, "iterations must be positive."
    assert args.learning_rate > 0, "learning_rate must be positive."
    assert args.layer_height > 0, "layer_height must be positive."

    h_value = args.layer_height
    max_layers_value = args.max_layers
    background_height_value = args.background_height
    background_layers_value = background_height_value // h_value
    decay_v_value = args.decay

    background = jnp.array(hex_to_rgb(args.background_color), dtype=jnp.float32)
    material_colors, material_TDs, material_names,material_hex = load_materials(args.csv_file)

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
    target = jnp.array(target, dtype=jnp.float32)

    rng_key = random.PRNGKey(0)
    # Choose loss function
    if args.loss == "mse":
        loss_fn_to_use = loss_fn
    elif args.loss == "perceptual":
        loss_fn_to_use = loss_fn_perceptual
    elif args.loss == "perceptual_l1":
        loss_fn_to_use = loss_fn_perceptual_l1
    else:
        raise ValueError("Invalid loss type")

    best_params, _ = run_optimizer(rng_key, target, new_h, new_w, max_layers_value, h_value,
                                   material_colors, material_TDs, background,
                                   args.iterations, args.learning_rate, decay_v_value,
                                   loss_function=loss_fn_to_use,
                                   visualize=args.visualize,
                                   save_max_tau=args.save_max_tau)

    rng_key, subkey = random.split(rng_key)
    gumbel_keys_disc = random.split(subkey, max_layers_value)
    tau_global_disc = decay_v_value
    disc_global, disc_height_image = discretize_solution_jax(best_params, tau_global_disc, gumbel_keys_disc, h_value, max_layers_value)
    discrete_comp = composite_image_discrete_jax(disc_height_image, disc_global, h_value, max_layers_value,
                                                 material_colors, material_TDs, background)

    discrete_comp_np = np.clip(np.array(discrete_comp), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_folder, "discrete_comp.jpg"),
                cv2.cvtColor(discrete_comp_np, cv2.COLOR_RGB2BGR))
    #additionally write as pil image
    #from PIL import Image
    #Image.fromarray(discrete_comp_np).save(os.path.join(args.output_folder, "discrete_comp_pil.jpg"))

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

    #project saving is not yet implemented correctly

    # project_filename = os.path.join(args.output_folder, "project_file.json")
    # generate_project_file(project_filename, args, np.array(disc_global), np.array(disc_height_image),
    #                       width_mm, height_mm, stl_filename,
    #                       material_names, material_TDs, material_hex)
    print("All outputs saved to", args.output_folder)
    print("Happy printing!")


if __name__ == '__main__':
    main()
