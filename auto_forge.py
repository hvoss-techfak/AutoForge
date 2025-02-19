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
    return material_colors, material_TDs, material_names


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


def composite_pixel_tempered(pixel_height_logit, global_logits, tau_height, tau_global, h, max_layers,
                             material_colors, material_TDs, background, gumbel_keys):
    """
    Composite one pixel using a learned height and per-layer soft indicator.

    Uses jax.lax.scan.
    """
    pixel_height = (max_layers * h) * jax.nn.sigmoid(pixel_height_logit)

    def step_fn(carry, i):
        comp, remaining = carry
        j = max_layers - 1 - i  # process from top to bottom
        p_print = jax.nn.sigmoid((pixel_height - j * h) / tau_height)
        eff_thick = p_print * h
        p_i = gumbel_softmax(global_logits[j], tau_global, gumbel_keys[j], hard=False)
        color_i = jnp.dot(p_i, material_colors)
        TD_i = jnp.dot(p_i, material_TDs)
        opac = jnp.minimum(1.0, eff_thick / (TD_i * 0.1))
        new_comp = comp + remaining * opac * color_i
        new_remaining = remaining * (1 - opac)
        return (new_comp, new_remaining), None

    init_state = (jnp.zeros(3), 1.0)
    (comp, remaining), _ = jax.lax.scan(step_fn, init_state, jnp.arange(max_layers))
    result = comp + remaining * background
    return result * 255.0


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
    Compute the mean squared error loss between the composite and target images.
    """
    comp = composite_image_tempered_fn(params['pixel_height_logits'], params['global_logits'],
                                       tau_height, tau_global, gumbel_keys,
                                       h, max_layers, material_colors, material_TDs, background)
    return jnp.mean((comp - target) ** 2)


def create_update_step(optimizer, h, max_layers, material_colors, material_TDs, background):
    """
    Create a JIT-compiled update step function.
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


def composite_image_discrete_jax(discrete_height_image, discrete_global, h, max_layers, mat_colors, mat_TDs, background):
    """
    Composite a discrete image from per-pixel layer counts and discrete global assignments.

    Uses jax.lax.scan.
    """
    def composite_pixel(pixel_printed_layers):
        def step_fn(carry, l):
            comp, remaining = carry
            idx = max_layers - 1 - l
            do_layer = idx < pixel_printed_layers

            def true_fn(carry):
                comp, remaining = carry
                mat_idx = discrete_global[idx]
                color = mat_colors[mat_idx]
                TD = mat_TDs[mat_idx]
                opac = jnp.minimum(1.0, h / (TD * 0.1))
                new_comp = comp + remaining * opac * color
                new_remaining = remaining * (1 - opac)
                return (new_comp, new_remaining)

            new_carry = jax.lax.cond(do_layer, true_fn, lambda c: c, (comp, remaining))
            return new_carry, None

        init_state = (jnp.zeros(3), 1.0)
        (comp, remaining), _ = jax.lax.scan(step_fn, init_state, jnp.arange(max_layers))
        result = comp + remaining * background
        return result * 255.0

    return jax.vmap(jax.vmap(composite_pixel))(discrete_height_image)

# Apply jit with static_argnums for "max_layers" (argument index 3)
composite_image_discrete_jax = jax.jit(composite_image_discrete_jax, static_argnums=(3,))


def run_optimizer(rng_key, target, H, W, max_layers, h, material_colors, material_TDs, background,
                  num_iters, learning_rate, decay_v, visualize=False):
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
        if visualize and (i % 50 == 0):
            comp = composite_image_tempered_fn(params['pixel_height_logits'], params['global_logits'],
                                               tau_height, tau_global, gumbel_keys,
                                               h, max_layers, material_colors, material_TDs, background)
            comp_np = np.clip(np.array(comp), 0, 255).astype(np.uint8)
            comp_im.set_data(comp_np)
            actual_layer_height = (max_layers * h) * jax.nn.sigmoid(best_params['pixel_height_logits'])
            highest_layer = np.max(np.array(actual_layer_height))
            fig.suptitle(f"Iteration {i}, Loss: {loss_val:.2f}, Best Loss: {best_loss:.2f}, Tau: {tau_height:.3f}, Highest Layer: {highest_layer:.3f}mm")
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
            ie = i+1
            instructions.append(f"At layer #{ie + background_layers} ({(ie * h) + background_height:.2f}mm) swap to {material_names[int(discrete_global[i])]}")
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
    parser.add_argument("--layer_height", type=float, default=0.04, help="Layer thickness in mm")
    parser.add_argument("--max_layers", type=int, default=50, help="Maximum number of layers")
    parser.add_argument("--background_height", type=float, default=0.4, help="Height of the background in mm")
    parser.add_argument("--background_color", type=str, default="#000000", help="Background color")
    parser.add_argument("--max_size", type=int, default=512, help="Maximum dimension for target image")
    parser.add_argument("--decay", type=float, default=0.01, help="Final tau value for Gumbel-Softmax")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization during optimization")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Ensure background height is divisible by layer height.
    assert (args.background_height / args.layer_height).is_integer(), "Background height must be divisible by layer height."

    h_value = args.layer_height
    max_layers_value = args.max_layers
    background_height_value = args.background_height
    background_layers_value = background_height_value // h_value
    decay_v_value = args.decay

    background = jnp.array(hex_to_rgb(args.background_color), dtype=jnp.float32)
    material_colors, material_TDs, material_names = load_materials(args.csv_file)

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
    best_params, _ = run_optimizer(rng_key, target, new_h, new_w, max_layers_value, h_value,
                                   material_colors, material_TDs, background,
                                   args.iterations, args.learning_rate, decay_v_value,
                                   visualize=args.visualize)

    rng_key, subkey = random.split(rng_key)
    gumbel_keys_disc = random.split(subkey, max_layers_value)
    tau_global_disc = decay_v_value
    disc_global, disc_height_image = discretize_solution_jax(best_params, tau_global_disc, gumbel_keys_disc, h_value, max_layers_value)
    discrete_comp = composite_image_discrete_jax(disc_height_image, disc_global, h_value, max_layers_value,
                                                 material_colors, material_TDs, background)

    discrete_comp_np = np.clip(np.array(discrete_comp), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_folder, "discrete_comp.png"),
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
    print("All outputs saved to", args.output_folder)
    print("Happy printing!")


if __name__ == '__main__':
    main()
