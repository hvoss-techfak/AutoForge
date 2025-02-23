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

from helper_functions import adaptive_round, gumbel_softmax, initialize_pixel_height_logits, hex_to_rgb, load_materials, \
    generate_stl, generate_swap_instructions, generate_project_file

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

    A = 0.178763
    k = 39.302848
    b = 0.351177

    def step_fn(carry, i):
        comp, remaining = carry
        # Process layers from top (last layer) to bottom (first layer)
        j = max_layers - 1 - i

        # Use a crisp (binary) decision:
        p_print = jnp.where(j < discrete_layers, 1.0, 0.0)
        eff_thick = p_print * h

        # For material selection, force a one-hot (hard) result when tau_global is very small.
        if mode == "discrete":
            p_i = gumbel_softmax(global_logits[j], tau_global, gumbel_keys[j], hard=True)
        else:
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

    Args:
        params (dict): Dictionary containing the parameters 'pixel_height_logits' and 'global_logits'.
        target (jnp.ndarray): The target image array.
        tau_height (float): Temperature parameter for height compositing.
        tau_global (float): Temperature parameter for material selection.
        gumbel_keys (jnp.ndarray): Random keys for sampling in each layer.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.
        material_colors (jnp.ndarray): Array of material colors.
        material_TDs (jnp.ndarray): Array of material transmission/opacity parameters.
        background (jnp.ndarray): Background color.

    Returns:
        jnp.ndarray: The mean squared error loss.
    """
    comp = composite_image_combined_jit(params['pixel_height_logits'], params['global_logits'],
                                        tau_height, tau_global, gumbel_keys,
                                        h, max_layers, material_colors, material_TDs, background, mode="continuous")
    return jnp.mean((comp - target) ** 2)


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
    def update_step(params, target, tau_height, tau_global, gumbel_keys, opt_state):
        """
        Perform a single update step.

        Args:
            params (dict): Dictionary containing the model parameters.
            target (jnp.ndarray): The target image array.
            tau_height (float): Temperature parameter for height compositing.
            tau_global (float): Temperature parameter for material selection.
            gumbel_keys (jnp.ndarray): Random keys for sampling in each layer.
            opt_state (optax.OptState): The current state of the optimizer.

        Returns:
            tuple: A tuple containing the updated parameters, new optimizer state, and the loss value.
        """
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

    Args:
        params (dict): Dictionary containing the parameters 'pixel_height_logits' and 'global_logits'.
        tau_global (float): Temperature parameter for material selection.
        gumbel_keys (jnp.ndarray): Random keys for sampling in each layer.
        h (float): Layer thickness.
        max_layers (int): Maximum number of layers.

    Returns:
        tuple: A tuple containing the discrete global material assignments and the discrete height image.
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


def run_optimizer(rng_key, target, H, W, max_layers, h, material_colors, material_TDs, background,
                  num_iters, learning_rate, decay_v, loss_function, visualize=False, save_max_tau=0.001,
                  output_folder=None, save_interval_pct=None,
                  img_width=None, img_height=None, background_height=None,
                  material_names=None, csv_file=None,args=None):
    """
    Run the optimization loop to learn per-pixel heights and per-layer material assignments.

    Args:
        rng_key (jax.random.PRNGKey): The random key for JAX operations.
        target (jnp.ndarray): The target image array.
        H (int): Height of the target image.
        W (int): Width of the target image.
        max_layers (int): Maximum number of layers.
        h (float): Layer thickness.
        material_colors (jnp.ndarray): Array of material colors.
        material_TDs (jnp.ndarray): Array of material transmission/opacity parameters.
        background (jnp.ndarray): Background color.
        num_iters (int): Number of optimization iterations.
        learning_rate (float): Learning rate for optimization.
        decay_v (float): Final tau value for Gumbel-Softmax.
        loss_function (callable): The loss function to compute gradients.
        visualize (bool, optional): Enable visualization during optimization. Defaults to False.
        save_max_tau (float, optional): Tau threshold to save best result. Defaults to 0.001.

    Returns:
        tuple: A tuple containing the best parameters and the best composite image.
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
    # pixel_height_logits = random.uniform(subkey, (H, W), minval=-2.0, maxval=2.0)

    params = {'global_logits': global_logits, 'pixel_height_logits': pixel_height_logits}

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    update_step = create_update_step(optimizer, loss_function, h, max_layers, material_colors, material_TDs, background)

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
    # Determine the checkpoint interval (in iterations) based on percentage progress.
    checkpoint_interval = int(num_iters * save_interval_pct / 100) if save_interval_pct is not None else None

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
        if loss_val < best_loss_since_last_save:
            best_loss_since_last_save = loss_val
            best_params_since_last_save = {k: jnp.array(v) for k, v in params.items()}
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
        if checkpoint_interval is not None and (i+1) % checkpoint_interval == 0 and i > 10:

            print("Saving intermediate outputs. This can take some time. You can turn off this feature by setting save_interval_pct to 0.")
            save_intermediate_outputs(i, best_params_since_last_save, tau_global, gumbel_keys, h, max_layers,
                                      material_colors, material_TDs, background,
                                      output_folder, W, H, background_height, material_names, csv_file,args=args)
            best_params_since_last_save = None
            best_loss_since_last_save = float('inf')
        tbar.set_description(f"loss = {loss_val:.4f}, Best Loss = {best_loss:.4f}")

    if visualize:
        plt.ioff()
        plt.close()
    best_comp = composite_image_combined_jit(best_params['pixel_height_logits'], best_params['global_logits'],
                                             tau_height, tau_global, gumbel_keys,
                                             h, max_layers, material_colors, material_TDs, background, mode="continuous")
    return best_params, best_comp


def save_intermediate_outputs(iteration, params, tau_global, gumbel_keys, h, max_layers,
                              material_colors, material_TDs, background,
                              output_folder, img_width, img_height, background_height,
                              material_names, csv_file, args):
    import os
    import cv2
    import numpy as np

    # Compute the discrete composite image.
    disc_comp = composite_image_combined_jit(
        params['pixel_height_logits'], params['global_logits'],
        tau_global, tau_global, gumbel_keys,
        h, max_layers, material_colors, material_TDs, background, mode="discrete")
    discrete_comp_np = np.clip(np.array(disc_comp), 0, 255).astype(np.uint8)
    image_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_comp.jpg")
    cv2.imwrite(image_filename, cv2.cvtColor(discrete_comp_np, cv2.COLOR_RGB2BGR))

    # Discretize the solution to obtain layer counts and material assignments.
    disc_global, disc_height_image = discretize_solution_jax(params, tau_global, gumbel_keys, h, max_layers)

    # Generate the STL file.
    height_map_mm = (np.array(disc_height_image, dtype=np.float32)) * h
    stl_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_model.stl")
    generate_stl(height_map_mm, stl_filename, background_height, scale=1.0)

    # Generate swap instructions.
    background_layers = int(background_height // h)
    swap_instructions = generate_swap_instructions(np.array(disc_global), np.array(disc_height_image),
                                                   h, background_layers, background_height, material_names)
    instructions_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_swap_instructions.txt")
    with open(instructions_filename, "w") as f:
        for line in swap_instructions:
            f.write(line + "\n")

    # Generate the project file.
    project_filename = os.path.join(output_folder, f"intermediate_iter_{iteration}_project.hfp")
    generate_project_file(project_filename, args,
                          np.array(disc_global),
                          np.array(disc_height_image),
                          img_width, img_height, stl_filename, csv_file)


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
    parser.add_argument("--save_interval_pct", type=float, default=20,help="Percentage interval to save intermediate results")

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
    best_params, _ = run_optimizer(
        rng_key, target, new_h, new_w, max_layers_value, h_value,
        material_colors, material_TDs, background,
        args.iterations, args.learning_rate, decay_v_value,
        loss_function=loss_fn,
        visualize=args.visualize,
        save_max_tau=args.save_max_tau,
        output_folder=args.output_folder,
        save_interval_pct=args.save_interval_pct if args.save_interval_pct > 0 else None,
        img_width=new_w, img_height=new_h,
        background_height=background_height_value,
        material_names=material_names,
        csv_file=args.csv_file,
        args=args
    )

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
