#!/usr/bin/env python
"""
Script for generating 3D printed layered models from an input image.

This script uses a learned optimization with a Gumbel softmax formulation
to assign materials per layer and produce both a discretized composite that
is exported as an STL file along with swap instructions.
"""
import os

import configargparse
import cv2
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt
import numpy as np
import math

from tqdm import tqdm
import pandas as pd


def hex_to_rgb(hex_str):
    """
    Convert a hex color string to a normalized RGB list.

    Parameters:
        hex_str (str): Hex color string (e.g. "#ff381e").

    Returns:
        list: Normalized RGB values.
    """
    hex_str = hex_str.lstrip('#')
    return [int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

def load_materials(csv_filename):
    """
        Load material data from a CSV file.

        Parameters:
            csv_filename (str): Path to the CSV file.

        Returns:
            tuple: (material_colors (jnp.array), material_TDs (np.array), material_names (list))
        """
    #For some reason the csv has whitespaces before each column name. Weird choice, but we can work around that
    df = pd.read_csv(csv_filename)
    material_names = [brand + " - " + name for brand, name in zip(df[" Brand"].tolist(), df[" Name"].tolist())]
    material_TDs = df[' TD'].astype(float).to_numpy()
    colors_list = df[' Color'].tolist()
    material_colors = jnp.array([hex_to_rgb(color) for color in colors_list], dtype=jnp.float32)
    return material_colors, material_TDs, material_names


def sample_gumbel(shape, key, eps=1e-20):
    """
    Sample from a Gumbel distribution.

    Parameters:
        shape (tuple): Shape of the sample.
        key: JAX random key.
        eps (float): Small constant for numerical stability.

    Returns:
        jnp.array: Sample from the Gumbel distribution.
    """
    U = random.uniform(key, shape=shape, minval=0.0, maxval=1.0)
    return -jnp.log(-jnp.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, key):
    """
    Sample from the Gumbel-Softmax distribution.

    Parameters:
        logits (jnp.array): Input logits.
        temperature (float): Temperature parameter.
        key: JAX random key.

    Returns:
        jnp.array: Sample from the Gumbel-Softmax distribution.
    """
    g = sample_gumbel(logits.shape, key)
    return jax.nn.softmax((logits + g) / temperature)

def gumbel_softmax(logits, temperature, key, hard=False):
    """
    Compute the Gumbel-Softmax.

    Parameters:
        logits (jnp.array): Input logits.
        temperature (float): Temperature parameter.
        key: JAX random key.
        hard (bool): Whether to produce a hard one-hot sample.

    Returns:
        jnp.array: Gumbel-Softmax sample.
    """
    y = gumbel_softmax_sample(logits, temperature, key)
    if hard:
        y_hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), logits.shape[-1])
        y = y_hard + jax.lax.stop_gradient(y - y_hard)
    return y

def composite_pixel_tempered(pixel_height_logit, global_logits, tau_height, tau_global, h, max_layers,
                             material_colors, material_TDs, background, gumbel_keys):
    """
    Composite one pixel using a learned height and per-layer soft indicator.

    Parameters:
        pixel_height_logit (float): Logit for the pixel height.
        global_logits (jnp.array): Global logits for each layer.
        tau_height (float): Temperature for the height indicator.
        tau_global (float): Temperature for the material selection.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (jnp.array): Array of material colors.
        material_TDs (jnp.array): Array of material TD values.
        background (jnp.array): Background color.
        gumbel_keys (jnp.array): Array of random keys for the Gumbel softmax.

    Returns:
        jnp.array: Composited pixel color (RGB).
    """
    pixel_height = (max_layers * h) * jax.nn.sigmoid(pixel_height_logit)

    def body_fn(i, state):
        comp, remaining = state
        j = max_layers - 1 - i  # process from top to bottom
        p_print = jax.nn.sigmoid((pixel_height - j * h) / tau_height)
        eff_thick = p_print * h
        p_i = gumbel_softmax(global_logits[j], tau_global, gumbel_keys[j], hard=False)
        color_i = jnp.dot(p_i, material_colors)
        TD_i = jnp.dot(p_i, material_TDs)
        opac = jnp.minimum(1.0, eff_thick / (TD_i * 0.1))
        new_comp = comp + remaining * opac * color_i
        new_remaining = remaining * (1 - opac)
        return (new_comp, new_remaining)

    init_state = (jnp.zeros(3), 1.0)
    comp, remaining = jax.lax.fori_loop(0, max_layers, body_fn, init_state)
    result = comp + remaining * background
    return result * 255.0

@jax.jit
def composite_image_tempered_fn(pixel_height_logits, global_logits, tau_height, tau_global, gumbel_keys,
                                 h, max_layers, material_colors, material_TDs, background):
    """
    Composite an entire image using tempered Gumbel compositing.

    Parameters:
        pixel_height_logits (jnp.array): 2D array of pixel height logits.
        global_logits (jnp.array): Global logits for each layer.
        tau_height (float): Temperature for the height indicator.
        tau_global (float): Temperature for the material selection.
        gumbel_keys (jnp.array): Random keys for each layer.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (jnp.array): Array of material colors.
        material_TDs (jnp.array): Array of material TD values.
        background (jnp.array): Background color.

    Returns:
        jnp.array: Composited image (H x W x 3).
    """
    return jax.vmap(jax.vmap(
        lambda ph_logit: composite_pixel_tempered(ph_logit, global_logits, tau_height, tau_global, h, max_layers,
                                                   material_colors, material_TDs, background, gumbel_keys)
    ))(pixel_height_logits)

def loss_fn(params, target, tau_height, tau_global, gumbel_keys, h, max_layers, material_colors, material_TDs, background):
    """
    Compute the mean squared error loss between the composite and target images.

    Parameters:
        params (dict): Contains 'pixel_height_logits' and 'global_logits'.
        target (jnp.array): Target image.
        tau_height (float): Temperature for height.
        tau_global (float): Temperature for material selection.
        gumbel_keys (jnp.array): Random keys.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (jnp.array): Array of material colors.
        material_TDs (jnp.array): Array of material TD values.
        background (jnp.array): Background color.

    Returns:
        float: Mean squared error loss.
    """
    comp = composite_image_tempered_fn(params['pixel_height_logits'], params['global_logits'],
                                       tau_height, tau_global, gumbel_keys,
                                       h, max_layers, material_colors, material_TDs, background)
    return jnp.mean((comp - target) ** 2)

def create_update_step(optimizer, h, max_layers, material_colors, material_TDs, background):
    """
    Create a JIT-compiled update step function.

    Parameters:
        optimizer: An optax optimizer.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (jnp.array): Array of material colors.
        material_TDs (jnp.array): Array of material TD values.
        background (jnp.array): Background color.

    Returns:
        function: The update step function.
    """
    @jax.jit
    def update_step(params, target, tau_height, tau_global, gumbel_keys, opt_state):
        loss_val, grads = jax.value_and_grad(loss_fn)(
            params, target, tau_height, tau_global, gumbel_keys,
            h, max_layers, material_colors, material_TDs, background)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val
    return update_step

def discretize_solution_jax(params, tau_global, gumbel_keys, h, max_layers):
    """
    Discretize continuous parameters into discrete global assignments and per-pixel layer counts.

    Parameters:
        params (dict): Contains 'pixel_height_logits' and 'global_logits'.
        tau_global (float): Temperature for discretizing global logits.
        gumbel_keys (jnp.array): Random keys.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.

    Returns:
        tuple: (discrete_global, discrete_height_image)
    """
    pixel_height_logits = params['pixel_height_logits']
    global_logits = params['global_logits']
    pixel_heights = (max_layers * h) * jax.nn.sigmoid(pixel_height_logits)
    discrete_height_image = jnp.floor(pixel_heights / h).astype(jnp.int32)
    discrete_height_image = jnp.clip(discrete_height_image, 0, max_layers)

    def discretize_layer(logits, key):
        p = gumbel_softmax(logits, tau_global, key, hard=True)
        return jnp.argmax(p)

    discrete_global = jax.vmap(discretize_layer)(global_logits, gumbel_keys)
    return discrete_global, discrete_height_image

@jax.jit
def composite_image_discrete_jax(discrete_height_image, discrete_global, h, max_layers, mat_colors, mat_TDs, background):
    """
    Composite a discrete image from per-pixel layer counts and discrete global assignments.

    Parameters:
        discrete_height_image (jnp.array): 2D array of printed layer counts.
        discrete_global (jnp.array): 1D array of discrete material assignments.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        mat_colors (jnp.array): Array of material colors.
        mat_TDs (jnp.array): Array of material TD values.
        background (jnp.array): Background color.

    Returns:
        jnp.array: Discrete composited image (H x W x 3).
    """
    def composite_pixel(pixel_printed_layers):
        def body_fn(l, state):
            comp, remaining = state
            idx = max_layers - 1 - l
            do_layer = idx < pixel_printed_layers

            def true_fn(state):
                comp, remaining = state
                mat_idx = discrete_global[idx]
                color = mat_colors[mat_idx]
                TD = mat_TDs[mat_idx]
                opac = jnp.minimum(1.0, h / (TD * 0.1))
                new_comp = comp + remaining * opac * color
                new_remaining = remaining * (1 - opac)
                return (new_comp, new_remaining)

            new_state = jax.lax.cond(do_layer, true_fn, lambda state: state, state)
            return new_state

        init_state = (jnp.zeros(3), 1.0)
        comp, remaining = jax.lax.fori_loop(0, max_layers, body_fn, init_state)
        result = comp + remaining * background
        return result * 255.0

    return jax.vmap(jax.vmap(composite_pixel))(discrete_height_image)



def run_optimizer(rng_key, target, H, W, max_layers, h, material_colors, material_TDs, background,
                  num_iters, learning_rate, decay_v, visualize=False):
    """
    Run the optimization loop to learn per-pixel heights and per-layer material assignments.

    Parameters:
        rng_key: JAX random key.
        target (jnp.array): Target image.
        H (int): Target image height.
        W (int): Target image width.
        max_layers (int): Maximum number of layers.
        h (float): Layer thickness.
        material_colors (jnp.array): Array of material colors.
        material_TDs (jnp.array): Array of material TD values.
        background (jnp.array): Background color.
        num_iters (int): Number of iterations.
        learning_rate (float): Learning rate.
        decay_v (float): Final tau value.
        visualize (bool): Whether to display live visualization.

    Returns:
        tuple: (best_params, best_composite)
    """
    num_materials = material_colors.shape[0]
    rng_key, subkey = random.split(rng_key)
    global_logits = random.normal(subkey, (max_layers, num_materials)) * 0.1
    rng_key, subkey = random.split(rng_key)
    pixel_height_logits = random.normal(subkey, (H, W)) * 0.1
    params = {'global_logits': global_logits, 'pixel_height_logits': pixel_height_logits}

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    update_step = create_update_step(optimizer, h, max_layers, material_colors, material_TDs, background)

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
        ax[1].set_title("Gumbel Composite")
        best_comp_im = ax[2].imshow(np.zeros((H, W, 3), dtype=np.uint8))
        ax[2].set_title("Best Gumbel Composite")
        disc_comp_im = ax[3].imshow(np.zeros((H, W, 3), dtype=np.uint8))
        ax[3].set_title("Discretized Composite")
        plt.pause(0.1)

    tbar = tqdm(range(num_iters))
    for i in tbar:
        tau_height = get_tau(i, tau_init=1.0, tau_final=decay_v, decay_rate=decay_rate)
        tau_global = get_tau(i, tau_init=1.0, tau_final=decay_v, decay_rate=decay_rate)
        rng_key, subkey = random.split(rng_key)
        gumbel_keys = random.split(subkey, max_layers)
        params, opt_state, loss_val = update_step(params, target, tau_height, tau_global, gumbel_keys, opt_state)
        if loss_val < best_loss:
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
        if visualize and (i % 200 == 0):
            comp = composite_image_tempered_fn(params['pixel_height_logits'], params['global_logits'],
                                               tau_height, tau_global, gumbel_keys,
                                               h, max_layers, material_colors, material_TDs, background)
            comp_np = np.clip(np.array(comp), 0, 255).astype(np.uint8)
            comp_im.set_data(comp_np)
            actual_layer_height = (max_layers * h) * jax.nn.sigmoid(best_params['pixel_height_logits'])
            highest_layer = np.max(np.array(actual_layer_height))
            fig.suptitle(f"Iteration {i}, Loss: {loss_val:.2f}, Tau: {tau_height:.3f}, Highest Layer: {highest_layer:.3f}mm")
            plt.pause(0.01)
        tbar.set_description(f"loss = {loss_val:.2f}, Best Loss = {best_loss:.2f}")

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

    Parameters:
        height_map (np.array): 2D array of height values.
        filename (str): Output STL file path.
        background_height (float): Height of the background.
        scale (float): Scale factor for the vertices.
    """
    H, W = height_map.shape
    vertices = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            vertices[i, j, 0] = j * scale
            vertices[i, j, 1] = i * scale
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
            add_triangle(v0, v1, v2)
            add_triangle(v0, v2, v3)

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

    Parameters:
        discrete_global (np.array): 1D array of discrete material assignments.
        discrete_height_image (np.array): 2D array of printed layer counts.
        h (float): Layer thickness.
        background_layers (float): Number of background layers.
        background_height (float): Height of the background.
        material_names (list): List of material names.

    Returns:
        list: Swap instructions.
    """
    L = int(np.max(np.array(discrete_height_image)))
    instructions = []
    if L == 0:
        instructions.append("No layers printed.")
        return instructions
    instructions.append(f"At layer #{background_layers + h} ({background_height + h:.2f}mm) swap to {material_names[int(discrete_global[0])]}")
    for i in range(1, L):
        if int(discrete_global[i]) != int(discrete_global[i - 1]):
            instructions.append(f"At layer #{i + background_layers} ({(i * h) + background_height:.2f}mm) swap to {material_names[int(discrete_global[i])]}")
    instructions.append("For the rest, use " + material_names[int(discrete_global[L - 1])])
    return instructions

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
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate for optimization")
    parser.add_argument("--target_max", type=int, default=512, help="Maximum dimension for target image")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization during optimization")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Model and optimization parameters
    h_value = 0.04
    max_layers_value = round(3.0 / h_value)
    background_height_value = 0.4
    background_layers_value = background_height_value // h_value
    decay_v_value = 0.01

    # Define background color
    background = jnp.array([0.0, 0.0, 0.0])

    # Load materials
    material_colors, material_TDs, material_names = load_materials(args.csv_file)
    num_materials = material_colors.shape[0]

    # Load and resize target image
    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img.shape
    if w_img >= h_img:
        new_w = args.target_max
        new_h = int(args.target_max * h_img / w_img)
    else:
        new_h = args.target_max
        new_w = int(args.target_max * w_img / h_img)
    target = cv2.resize(img, (new_w, new_h))
    target = jnp.array(target, dtype=jnp.float32)

    rng_key = random.PRNGKey(0)
    update_step_fn = create_update_step(optax.adam(args.learning_rate),
                                        h_value, max_layers_value, material_colors, material_TDs, background)
    best_params, _ = run_optimizer(rng_key, target, new_h, new_w, max_layers_value, h_value,
                                   material_colors, material_TDs, background,
                                   args.iterations, args.learning_rate, decay_v_value, update_step_fn,
                                   visualize=args.visualize)

    # Discretize the final solution and create discrete composite
    rng_key, subkey = random.split(rng_key)
    gumbel_keys_disc = random.split(subkey, max_layers_value)
    tau_global_disc = decay_v_value
    disc_global, disc_height_image = discretize_solution_jax(best_params, tau_global_disc, gumbel_keys_disc, h_value, max_layers_value)
    discrete_comp = composite_image_discrete_jax(disc_height_image, disc_global, h_value, max_layers_value,
                                                 material_colors, material_TDs, background)

    # Save discrete composite image
    discrete_comp_np = np.clip(np.array(discrete_comp), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_folder, "discrete_comp.png"),
                cv2.cvtColor(discrete_comp_np, cv2.COLOR_RGB2BGR))

    # Generate and save STL file
    height_map_mm = (np.array(disc_height_image, dtype=np.float32)) * h_value
    stl_filename = os.path.join(args.output_folder, "final_model.stl")
    generate_stl(height_map_mm, stl_filename, background_height_value, scale=1.0)

    # Generate and save swap instructions
    swap_instructions = generate_swap_instructions(np.array(disc_global), np.array(disc_height_image),
                                                   h_value, background_layers_value, background_height_value, material_names)
    instructions_filename = os.path.join(args.output_folder, "swap_instructions.txt")
    with open(instructions_filename, "w") as f:
        for line in swap_instructions:
            f.write(line + "\n")
    print("All outputs saved to", args.output_folder)
    print("Happy printing!")


if __name__ == '__main__':
    main()