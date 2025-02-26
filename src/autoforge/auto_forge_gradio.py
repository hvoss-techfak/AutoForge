#!/usr/bin/env python
"""
gradio_interface.py

This file demonstrates a Gradio interface that wraps your optimization code
and also adds a "Height Map Annotation" tab. In that tab the initial height map
is previewed (using your init_height_map function) and you can annotate it manually
with color labels loaded from your CSV file.
"""
import json
import threading
import time
import queue
import os
import cv2
import numpy as np
import gradio as gr
import jax
import jax.numpy as jnp
import optax
import math
import time

import pandas as pd

# Import your functions from your original module.
# (Make sure these functions are available in your PYTHONPATH or adjust the import as needed.)
from autoforge.auto_forge import (
    run_optimizer,
    loss_fn,
    composite_image_combined_jit,
    init_height_map,
    hex_to_rgb,
    load_materials,
    generate_stl,
    generate_swap_instructions,
    generate_project_file,
    gumbal_bruteforce,
    pruning
)
from autoforge.filamentcolors_library import download_filament_info
from autoforge.helper_functions import extract_colors_from_swatches, swatch_data_to_table

# -------------------------------
# Global state for background optimization
# -------------------------------
pause_flag = threading.Event()
stop_flag = threading.Event()
update_queue = queue.Queue()
optimizer_thread = None
current_optimizer_state = {}  # Holds the latest state (iteration, loss, composite image, etc.)


# =============================================================================
# Helper functions for height map preview and annotation
# =============================================================================
def compute_initial_height_map(input_image, max_layers, layer_height):
    """
    Compute and return an initial height map preview from the input image.
    The height map is normalized to 0-255 for display.
    """
    # Convert the input image to a jax array (assumed to be RGB)
    target = jnp.array(input_image, dtype=jnp.float64)
    height_map = init_height_map(target, max_layers, layer_height)
    height_map_np = np.array(height_map)
    # Normalize for display purposes.
    norm = (height_map_np - height_map_np.min())
    norm = norm / (norm.max() + 1e-6)
    normalized = (255 * norm).astype(np.uint8)
    return normalized


def get_material_labels(csv_file):
    """
    Load the material names from the CSV file.
    """
    _, _, material_names, _ = load_materials(csv_file)
    return material_names


def init_height_map_preview(input_image, max_layers, layer_height, csv_file):
    """
    Compute the initial height map from the uploaded image and update the color labels.
    Returns:
      - The height map image for preview.
      - An update for the dropdown choices containing the material labels.
    """
    if input_image is None:
        return None, gr.Dropdown.update(choices=[])
    hm = compute_initial_height_map(input_image, max_layers, layer_height)
    labels = get_material_labels(csv_file)
    default_val = labels[0] if labels else None
    return hm, gr.Dropdown.update(choices=labels, value=default_val)


def save_annotated_height_map(annotated_hm):
    """
    Save the annotated height map image to a file and return the file path.
    """
    output_path = "annotated_height_map.png"
    # Assume the annotated image is in RGB format.
    cv2.imwrite(output_path, cv2.cvtColor(annotated_hm, cv2.COLOR_RGB2BGR))
    return output_path


# =============================================================================
# Background Optimization Function (Simulated)
# =============================================================================
def run_optimization_background(
        input_image, csv_file,
        iterations, learning_rate,
        layer_height, max_layers,
        background_height, background_color,
        output_size, solver_size,
        decay, perform_gumbal_search, perform_pruning,
        save_interval_pct
):
    """
    This function loads the image and material data, then runs the optimizer.
    It updates a shared queue with intermediate outputs for the UI.
    (Here we simulate the iterative updates; integrate your actual optimizer loop as needed.)
    """
    global current_optimizer_state

    pause_flag.clear()
    stop_flag.clear()

    # Load and prepare the image.
    img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    solver_size = int(solver_size)
    target_solver = cv2.resize(img, (solver_size, solver_size))
    target_solver = jnp.array(target_solver, dtype=jnp.float64)

    # Load materials.
    material_colors, material_TDs, material_names, material_hex = load_materials(csv_file)
    background = jnp.array(hex_to_rgb(background_color), dtype=jnp.float64)

    # Initialize optimizer state (simulate initialization).
    rng_key = jax.random.PRNGKey(int(time.time()))

    # Simulated iterative optimization loop.
    for i in range(iterations):
        if stop_flag.is_set():
            current_optimizer_state["message"] = "Optimization stopped."
            update_queue.put(current_optimizer_state)
            return
        # Pause if needed.
        while pause_flag.is_set():
            current_optimizer_state["message"] = f"Paused at iteration {i}."
            update_queue.put(current_optimizer_state)
            time.sleep(0.2)

        # Simulate an update step.
        simulated_loss = np.exp(-i / iterations) * 100 + np.random.rand() * 5
        simulated_comp = np.clip(np.random.randn(solver_size, solver_size, 3) * 20 + 128, 0, 255).astype(np.uint8)
        current_optimizer_state = {
            "iteration": i,
            "loss": simulated_loss,
            "composite": simulated_comp,
            "message": ""
        }
        update_queue.put(current_optimizer_state)
        time.sleep(0.1)

    current_optimizer_state["message"] = "Optimization complete."
    update_queue.put(current_optimizer_state)


# =============================================================================
# Live update generator for Gradio streaming
# =============================================================================
def live_updates():
    """
    Generator that yields the latest composite image and status text.
    """
    while True:
        try:
            state = update_queue.get(timeout=1.0)
            composite = state.get("composite", np.zeros((128, 128, 3), dtype=np.uint8))
            iteration = state.get("iteration", 0)
            loss = state.get("loss", 0)
            message = state.get("message", "")
            status_text = f"Iteration: {iteration} | Loss: {loss:.4f} {message}"
            yield composite, status_text
            if message in ["Optimization complete.", "Optimization stopped."]:
                break
        except queue.Empty:
            if stop_flag.is_set():
                break


# =============================================================================
# Button callback functions for optimization control.
# =============================================================================
def start_optimization(
        input_image, csv_file,
        iterations, learning_rate,
        layer_height, max_layers,
        background_height, background_color,
        output_size, solver_size,
        decay, perform_gumbal_search, perform_pruning,
        save_interval_pct
):
    global optimizer_thread
    if optimizer_thread is None or not optimizer_thread.is_alive():
        optimizer_thread = threading.Thread(
            target=run_optimization_background,
            args=(
                input_image, csv_file, iterations, learning_rate,
                layer_height, max_layers, background_height, background_color,
                output_size, solver_size, decay, perform_gumbal_search, perform_pruning,
                save_interval_pct
            ),
            daemon=True
        )
        optimizer_thread.start()
    return "Optimization started."


def pause_optimization():
    pause_flag.set()
    return "Optimization paused."


def resume_optimization():
    pause_flag.clear()
    return "Optimization resumed."


def stop_optimization():
    stop_flag.set()
    return "Optimization stopped."

# =============================================================================
# Filament Tab Helper
# =============================================================================
def filter_table(search_text, table):
    if not search_text:
        return table
    filtered = [
        row for row in table
        if search_text.lower() in row["Brand"].lower() or search_text.lower() in row["Name"].lower()
    ]
    return filtered

# Function to update the active list based on selection.
def update_active(selected, table):
    # 'selected' is a list of strings in the format "Brand - Name"
    active = [row for row in table if f'{row["Brand"]} - {row["Name"]}' in selected]
    return active

# =============================================================================
# Postprocessing callbacks (simulated)
# =============================================================================
def run_gumbal_search_callback():
    updated_comp = np.clip(np.random.randn(128, 128, 3) * 20 + 150, 0, 255).astype(np.uint8)
    status = "Gumbal search completed."
    return updated_comp, status


def run_pruning_callback():
    updated_comp = np.clip(np.random.randn(128, 128, 3) * 20 + 170, 0, 255).astype(np.uint8)
    status = "Pruning completed."
    return updated_comp, status


# =============================================================================
# Export callback.
# =============================================================================
def export_outputs_callback():
    output_folder = "temp_output"
    os.makedirs(output_folder, exist_ok=True)
    stl_path = os.path.join(output_folder, "final_model.stl")
    swap_path = os.path.join(output_folder, "swap_instructions.txt")
    project_path = os.path.join(output_folder, "project_file.hfp")

    with open(stl_path, "w") as f:
        f.write("STL file content")
    with open(swap_path, "w") as f:
        f.write("Swap instructions content")
    with open(project_path, "w") as f:
        f.write("Project file content")

    # For this example, we return the STL file for download.
    return stl_path

#download filament info
print(f"Downloading filament info...")
print("If this is your first time running this script, it may take a few minutes.")
download_filament_info()
#load swatch json file
with open("swatches.json", "r") as f:
    swatch_data = json.load(f)

table = swatch_data_to_table(swatch_data)
table_df = pd.DataFrame(table)


def filter_swatches_callback(search_query, table_data):
    """
    Given a search query and the full swatch table data,
    returns a filtered table (including the header) where the query is
    found in any field (case-insensitive).
    """
    if table_data is None or len(table_data) == 0:
        return []
    header = table_data[0]
    rows = table_data[1:]
    if search_query is None or search_query.strip() == "":
        filtered = table_data
    else:
        filtered_rows = [
            row for row in rows
            if any(search_query.lower() in str(cell).lower() for cell in row)
        ]
        filtered = [header] + filtered_rows
    return filtered


def add_to_active_callback(selected, table_data, active_data):
    """
    Given a list of selected material names (e.g., "Brand - Name"),
    adds those rows from the full table_data (if not already present)
    to the active list.
    """
    if table_data is None or len(table_data) == 0:
        return active_data
    header = table_data[0]
    if active_data is None or len(active_data) == 0:
        active_data = [header]
    else:
        # Ensure header is set
        if active_data[0] != header:
            active_data[0] = header
    # Build a set of already-added materials (by "Brand - Name")
    existing = {row[0] + " - " + row[1] for row in active_data[1:]}
    # Loop through table_data rows and add those whose "Brand - Name" is in selected
    for row in table_data[1:]:
        material = row[0] + " - " + row[1]
        if material in selected and material not in existing:
            active_data.append(row)
    return active_data

def highlight_cols(x):
    df = x.copy()
    df.loc[:, :] = 'color: purple'
    df[['B', 'C', 'E']] = 'color: green'
    return df

# ------------------------------------------------------------------------------
# Gradio Interface (including the new Color Filament Tab)
# ------------------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("## Autoforge")

    # -------------------------
    # Color Filament Tab
    # -------------------------
    with gr.Tab("Color Filament"):
        gr.Markdown("### Manage and Select Filament Colors")

        with gr.Row():
            #fill with data from table_df
            material_select = gr.Dropdown(label="Select Filaments", multiselect=True,interactive=Trueasdasdasdsseee, choices=[brand + " - " + name + " - " + hex_color for brand,name,hex_color in table_df[["Brand", "Name", "Hex Color"]].values.tolist()])
            add_active_btn = gr.Button("Add to Active List")
        with gr.Row():
            active_filaments = gr.DataFrame(label="Active Filaments", interactive=True)

        with gr.Row():
            #s = table_df.grsdfstyle.apply(highlight_cols, axis=None)
            swatch_table = gr.DataFrame(value=table_df,label="All Filaments", interactive=True)

        # 3. Add the selected materials to the active list.grgr
        add_active_btn.click(
            add_to_active_callback,
            inputs=[material_select, swatch_table, active_filaments],
            outputs=active_filaments
        )

    with gr.Tab("Height Map Annotation"):
        gr.Markdown("### Preview and Annotate Height Map")
        with gr.Row():
            hm_input_image = gr.Image(label="Upload Image",
              sources=['upload'],
              type='numpy'
            )
            hm_csv_file = gr.Textbox(label="CSV File Path", value="materials.csv")
        with gr.Row():
            hm_max_layers = gr.Slider(10, 150, label="Max Layers", value=75, step=1)
            hm_layer_height = gr.Slider(0.01, 0.1, label="Layer Height (mm)", value=0.04, step=0.01)
        load_hm_btn = gr.Button("Load Height Map Preview")
        with gr.Row():
            height_map_preview = gr.ImageEditor(
                label="Height Map",
                sources=['upload'],
                type='pil'
            )

            color_label_dropdown = gr.Dropdown(label="Select Material Label", choices=[], value=None)
        save_hm_btn = gr.Button("Save Annotated Height Map")
        hm_download = gr.File(label="Download Annotated Height Map")

        # When "Load Height Map Preview" is clicked, compute the initial height map and update the dropdown.
        load_hm_btn.click(
            init_height_map_preview,
            inputs=[hm_input_image, hm_max_layers, hm_layer_height, hm_csv_file],
            outputs=[height_map_preview, color_label_dropdown]
        )
        # When "Save Annotated Height Map" is clicked, save the current annotated image.
        save_hm_btn.click(
            save_annotated_height_map,
            inputs=[height_map_preview],
            outputs=hm_download
        )

    with gr.Tab("Optimization"):
            with gr.Row():
                input_image = gr.ImageEditor(label="Upload Image",
                  sources=['upload'],
                  type= 'pil'
                )
                csv_file = gr.Textbox(label="CSV File Path", value="materials.csv")
            with gr.Row():
                iterations_slider = gr.Slider(100, 10000, step=100, label="Iterations", value=5000)
                learning_rate_slider = gr.Slider(1e-4, 1e-1, label="Learning Rate", value=1e-2, step=1e-4)
            with gr.Row():
                layer_height_slider = gr.Slider(0.01, 0.1, label="Layer Height (mm)", value=0.04, step=0.01)
                max_layers_slider = gr.Slider(10, 150, label="Max Layers", value=75, step=1)
            with gr.Row():
                background_height_slider = gr.Slider(0.1, 2.0, label="Background Height (mm)", value=0.4, step=0.1)
                output_size_slider = gr.Slider(256, 2048, label="Output Size", value=1024, step=64)
                solver_size_slider = gr.Slider(64, 512, label="Solver Size", value=128, step=16)
            with gr.Row():
                decay_slider = gr.Slider(0.001, 0.1, label="Decay", value=0.01, step=0.001)
            with gr.Row():
                perform_gumbal_search_checkbox = gr.Checkbox(label="Perform Gumbal Search", value=True)
                perform_pruning_checkbox = gr.Checkbox(label="Perform Pruning", value=True)
            with gr.Row():
                save_interval_slider = gr.Slider(1, 100, label="Save Interval (%)", value=20, step=1)
            # Create a separate textbox component for background color
            background_color_textbox = gr.Textbox(label="Background Color", value="#000000")
            with gr.Row():
                start_btn = gr.Button("Start Optimization")
                pause_btn = gr.Button("Pause")
                resume_btn = gr.Button("Resume")
                stop_btn = gr.Button("Stop")
            status_box = gr.Textbox(label="Status", interactive=False)
            live_image = gr.Image(label="Current Composite")
            live_status = gr.Textbox(label="Live Status", interactive=False)

            start_btn.click(
                start_optimization,
                inputs=[
                    input_image, csv_file,
                    iterations_slider, learning_rate_slider,
                    layer_height_slider, max_layers_slider,
                    background_height_slider, background_color_textbox,
                    output_size_slider, solver_size_slider,
                    decay_slider, perform_gumbal_search_checkbox, perform_pruning_checkbox,
                    save_interval_slider
                ],
                outputs=status_box
            )
            pause_btn.click(pause_optimization, outputs=status_box)
            resume_btn.click(resume_optimization, outputs=status_box)
            stop_btn.click(stop_optimization, outputs=status_box)

            live_interface = gr.Interface(
                fn=live_updates,
                inputs=[],
                outputs=[live_image, live_status],
                live=True,
                title="Live Optimization Updates"
            )

    with gr.Tab("Post-Processing"):
        gr.Markdown("### Run Post-Processing Steps")
        with gr.Row():
            gumbal_btn = gr.Button("Run Gumbal Search")
            pruning_btn = gr.Button("Run Pruning")
        post_image = gr.Image(label="Post-Processed Composite")
        post_status = gr.Textbox(label="Post-Processing Status", interactive=False)
        gumbal_btn.click(run_gumbal_search_callback, inputs=[], outputs=[post_image, post_status])
        pruning_btn.click(run_pruning_callback, inputs=[], outputs=[post_image, post_status])

    with gr.Tab("Export Outputs"):
        gr.Markdown("### Export Files")
        export_btn = gr.Button("Export Outputs")
        export_file = gr.File(label="Download Output (STL)")
        export_btn.click(export_outputs_callback, inputs=[], outputs=export_file)

demo.launch()
