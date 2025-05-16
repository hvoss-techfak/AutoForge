import gradio as gr
import pandas as pd
import os
import subprocess
import time
import shutil
import sys
from datetime import datetime
import re

# --- Configuration ---
#AUTFORGE_SCRIPT_PATH = "auto_forge.py"  # Make sure this points to your script
DEFAULT_MATERIALS_CSV = "default_materials.csv"
GRADIO_OUTPUT_BASE_DIR = "output"
os.makedirs(GRADIO_OUTPUT_BASE_DIR, exist_ok=True)

REQUIRED_SCRIPT_COLS = ["Brand", " Name", " TD", " Color"]
DISPLAY_COL_MAP = {
    "Brand": "Brand",
    " Name": "Name",
    " TD": "TD",
    " Color": "Color (Hex)",
}


def ensure_required_cols(df, *, in_display_space):
    """
    Return a copy of *df* with every required column present.
    If *in_display_space* is True we use the display names
    (Brand, Name, TD, Color (Hex)); otherwise we use the script names.
    """
    target_cols = (
        DISPLAY_COL_MAP if in_display_space else {k: k for k in REQUIRED_SCRIPT_COLS}
    )
    df_fixed = df.copy()
    for col_script, col_display in target_cols.items():
        if col_display not in df_fixed.columns:
            # sensible defaults
            if "TD" in col_display:
                default = 0.0
            elif "Color" in col_display:
                default = "#000000"
            elif "Owned" in col_display:  # NEW
                default = "false"
            else:
                default = ""
            df_fixed[col_display] = default
    # order columns nicely
    return df_fixed[list(target_cols.values())]


def rgba_to_hex(col: str) -> str:
    """
    Turn 'rgba(r, g, b, a)' or 'rgb(r, g, b)' into '#RRGGBB'.
    If the input is already a hex code or anything unexpected,
    return it unchanged.
    """
    if not isinstance(col, str):
        return col
    col = col.strip()
    if col.startswith("#"):          # already fine
        return col.upper()

    m = re.match(
        r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*[\d.]+)?\s*\)",
        col,
    )
    if not m:
        return col                   # not something we recognise

    r, g, b = (int(float(x)) for x in m.groups()[:3])
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


# --- Helper Functions ---
def get_script_args_info(exclude_args=None):
    if exclude_args is None:
        exclude_args = []

    all_args_info = [
        # input_image is handled separately in the UI
        {
            "name": "--iterations",
            "type": "number",
            "default": 2000,
            "help": "Number of optimization iterations",
        },
        {
            "name": "--layer_height",
            "type": "number",
            "default": 0.04,
            "step": 0.01,
            "help": "Layer thickness in mm",
        },
        {
            "name": "--max_layers",
            "type": "number",
            "default": 75,
            "precision": 0,
            "help": "Maximum number of layers",
        },
        {
            "name": "--learning_rate",
            "type": "number",
            "default": 0.015,
            "step": 0.001,
            "help": "Learning rate for optimization",
        },
        {
            "name": "--background_height",
            "type": "number",
            "default": 0.4,
            "step": 0.01,
            "help": "Height of the background in mm",
        },
        {
            "name": "--background_color",
            "type": "colorpicker",
            "default": "#000000",
            "help": "Background color",
        },
        {
            "name": "--stl_output_size",
            "type": "number",
            "default": 100,
            "precision": 0,
            "help": "Size of the longest dimension of the output STL file in mm",
        },
        {
            "name": "--nozzle_diameter",
            "type": "number",
            "default": 0.4,
            "step": 0.1,
            "help": "Diameter of the printer nozzle in mm",
        },
        {
            "name": "--pruning_max_colors",
            "type": "number",
            "default": 10,
            "precision": 0,
            "help": "Max number of colors allowed after pruning",
        },
        {
            "name": "--pruning_max_swaps",
            "type": "number",
            "default": 20,
            "precision": 0,
            "help": "Max number of swaps allowed after pruning",
        },
        {
            "name": "--pruning_max_layer",
            "type": "number",
            "default": 75,
            "precision": 0,
            "help": "Max number of layers allowed after pruning",
        },
        {
            "name": "--warmup_fraction",
            "type": "slider",
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Fraction of iterations for keeping the tau at the initial value",
        },
        {
            "name": "--learning_rate_warmup_fraction",
            "type": "slider",
            "default": 0.25,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "help": "Fraction of iterations that the learning rate is increasing (warmup)",
        },
        {
            "name": "--init_tau",
            "type": "number",
            "default": 1.0,
            "help": "Initial tau value for Gumbel-Softmax",
        },
        {
            "name": "--final_tau",
            "type": "number",
            "default": 0.01,
            "help": "Final tau value for Gumbel-Softmax",
        },
        {
            "name": "--min_layers",
            "type": "number",
            "default": 0,
            "precision": 0,
            "help": "Minimum number of layers. Used for pruning.",
        },
        {
            "name": "--early_stopping",
            "type": "number",
            "default": 1500,
            "precision": 0,
            "help": "Number of steps without improvement before stopping",
        },
        {
            "name": "--random_seed",
            "type": "number",
            "default": 0,
            "precision": 0,
            "help": "Specify the random seed, or use 0 for automatic generation",
        },
        {
            "name": "--num_init_rounds",
            "type": "number",
            "default": 32,
            "precision": 0,
            "help": "Number of rounds to choose the starting height map from.",
        },
    ]
    return [arg for arg in all_args_info if arg["name"] not in exclude_args]


# Initial filament data
initial_filament_data = {
    "Brand": ["Generic", "Generic", "Generic"],
    " Name": ["PLA Black", "PLA Grey", "PLA White"],
    " TD": [1.0, 1.0, 1.0],
    " Color": ["#000000", "#808080", "#FFFFFF"],
    " Owned": ["true", "true", "true"],  # ← add
}
initial_df = pd.DataFrame(initial_filament_data)

if os.path.exists(DEFAULT_MATERIALS_CSV):
    try:
        initial_df = pd.read_csv(DEFAULT_MATERIALS_CSV)
        for col in ["Brand", " Name", " TD", " Color"]:
            if col not in initial_df.columns:
                initial_df[col] = None
        initial_df = initial_df[["Brand", " Name", " TD", " Color"]].astype(
            {" TD": float, " Color": str}
        )
    except Exception as e:
        print(f"Warning: Could not load {DEFAULT_MATERIALS_CSV}: {e}. Using default.")
        initial_df = pd.DataFrame(initial_filament_data)
else:
    initial_df.to_csv(DEFAULT_MATERIALS_CSV, index=False)


# Helper for creating an empty 10-tuple for error returns
def create_empty_error_outputs(log_message=""):
    return (
        log_message,  # progress_output
        None,  # final_image_preview
        gr.update(visible=False, interactive=False),  # ### ZIP: download_zip
    )


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Autoforge Web UI")

    filament_df_state = gr.State(initial_df.copy())
    current_run_output_dir = gr.State(None)

    with gr.Tabs():
        with gr.TabItem("Filament Management"):
            gr.Markdown(
                'Manage your filament list. This list will be saved as a CSV and used by the Autoforge process. \n To remove a filament simply rightclick on any of the fields and select "Delete Row"'
            )
            with gr.Row():
                load_csv_button = gr.UploadButton(
                    "Load Filaments CSV", file_types=[".csv"]
                )
                save_csv_button = gr.Button("Save Current Filaments to CSV")
            filament_table = gr.DataFrame(
                value=ensure_required_cols(
                    initial_df.copy().rename(
                        columns={" Name": "Name", " TD": "TD", " Color": "Color (Hex)"}
                    ),
                    in_display_space=True,
                ),
                headers=["Brand", "Name", "TD", "Color (Hex)"],
                datatype=["str", "str", "number", "str"],
                interactive=True,
                label="Filaments",
            )
            gr.Markdown("### Add New Filament")
            with gr.Row():
                new_brand = gr.Textbox(label="Brand")
                new_name = gr.Textbox(label="Name")
            with gr.Row():
                new_td = gr.Number(
                    label="TD (Transmission/Opacity)",
                    value=1.0,
                    minimum=0,
                    maximum=100,
                    step=0.1,
                )
                new_color_hex = gr.ColorPicker(label="Color", value="#FF0000")
            add_filament_button = gr.Button("Add Filament to Table")
            download_csv_trigger = gr.File(
                label="Download Filament CSV", visible=False, interactive=False
            )

            def update_filament_df_state_from_table(display_df):
                display_df = ensure_required_cols(display_df, in_display_space=True)

                # make sure every colour is hex
                if "Color (Hex)" in display_df.columns:
                    display_df["Color (Hex)"] = display_df["Color (Hex)"].apply(
                        rgba_to_hex
                    )

                script_df = display_df.rename(
                    columns={"Name": " Name", "TD": " TD", "Color (Hex)": " Color"}
                )
                script_df = ensure_required_cols(script_df, in_display_space=False)
                filament_df_state.value = script_df

            def add_filament_to_table(current_display_df, brand, name, td, color_hex):
                if not brand or not name:
                    gr.Warning("Brand and Name cannot be empty.")
                    return current_display_df

                color_hex = rgba_to_hex(color_hex)  # <-- new line

                new_row = pd.DataFrame(
                    [{"Brand": brand, "Name": name, "TD": td, "Color (Hex)": color_hex}]
                )
                updated_display_df = pd.concat(
                    [current_display_df, new_row], ignore_index=True
                )
                update_filament_df_state_from_table(updated_display_df)
                return updated_display_df

            def load_filaments_from_csv_upload(file_obj):
                if file_obj is None:
                    current_script_df = filament_df_state.value
                    if current_script_df is not None and not current_script_df.empty:
                        return current_script_df.rename(
                            columns={
                                " Name": "Name",
                                " TD": "TD",
                                " Color": "Color (Hex)",
                            }
                        )
                    return initial_df.copy().rename(
                        columns={" Name": "Name", " TD": "TD", " Color": "Color (Hex)"}
                    )
                try:
                    loaded_script_df = pd.read_csv(file_obj.name)
                    loaded_script_df = ensure_required_cols(
                        loaded_script_df, in_display_space=False
                    )
                    expected_cols = ["Brand", " Name", " TD", " Color"]
                    if not all(
                        col in loaded_script_df.columns for col in expected_cols
                    ):
                        gr.Error(
                            f"CSV must contain columns: {', '.join(expected_cols)}. Found: {loaded_script_df.columns.tolist()}"
                        )
                        current_script_df = filament_df_state.value
                        if (
                            current_script_df is not None
                            and not current_script_df.empty
                        ):
                            return current_script_df.rename(
                                columns={
                                    " Name": "Name",
                                    " TD": "TD",
                                    " Color": "Color (Hex)",
                                }
                            )
                        return initial_df.copy().rename(
                            columns={
                                " Name": "Name",
                                " TD": "TD",
                                " Color": "Color (Hex)",
                            }
                        )
                    filament_df_state.value = loaded_script_df.copy()
                    return loaded_script_df.rename(
                        columns={" Name": "Name", " TD": "TD", " Color": "Color (Hex)"}
                    )
                except Exception as e:
                    gr.Error(f"Error loading CSV: {e}")
                    current_script_df = filament_df_state.value
                    if current_script_df is not None and not current_script_df.empty:
                        return current_script_df.rename(
                            columns={
                                " Name": "Name",
                                " TD": "TD",
                                " Color": "Color (Hex)",
                            }
                        )
                    return initial_df.copy().rename(
                        columns={" Name": "Name", " TD": "TD", " Color": "Color (Hex)"}
                    )

            def save_filaments_to_file_for_download(current_script_df_from_state):
                if (
                    current_script_df_from_state is None
                    or current_script_df_from_state.empty
                ):
                    gr.Warning("Filament table is empty. Nothing to save.")
                    return None
                df_to_save = current_script_df_from_state.copy()
                required_cols = ["Brand", " Name", " TD", " Color"]
                if not all(col in df_to_save.columns for col in required_cols):
                    gr.Error(
                        f"Cannot save. DataFrame missing required script columns. Expected: {required_cols}. Found: {df_to_save.columns.tolist()}"
                    )
                    return None
                temp_dir = os.path.join(GRADIO_OUTPUT_BASE_DIR, "_temp_downloads")
                os.makedirs(temp_dir, exist_ok=True)
                temp_filament_csv_path = os.path.join(
                    temp_dir,
                    f"filaments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                )
                try:
                    df_to_save.to_csv(temp_filament_csv_path, index=False)
                    gr.Info("Filaments prepared for download.")
                    return gr.File(
                        value=temp_filament_csv_path,
                        label="Download Filament CSV",
                        interactive=True,
                        visible=True,
                    )
                except Exception as e:
                    gr.Error(f"Error saving CSV for download: {e}")
                    return None

            filament_table.change(
                update_filament_df_state_from_table,
                inputs=[filament_table],
                outputs=None,
                queue=False,
            )
            add_filament_button.click(
                add_filament_to_table,
                inputs=[filament_table, new_brand, new_name, new_td, new_color_hex],
                outputs=[filament_table],
            )
            load_csv_button.upload(
                load_filaments_from_csv_upload,
                inputs=[load_csv_button],
                outputs=[filament_table],
            )
            save_csv_button.click(
                save_filaments_to_file_for_download,
                inputs=[filament_df_state],
                outputs=[download_csv_trigger],
            )

        with gr.TabItem("Run Autoforge"):
            accordion_params_dict = {}
            accordion_params_ordered_names = []

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Image (Required)")
                    input_image_component = gr.Image(
                        type="filepath",
                        label="Upload Image",
                        sources=["upload"],
                        interactive=True,
                    )
                with gr.Column(scale=2):
                    gr.Markdown("### Autoforge Parameters")
                    with gr.Accordion("Progress & Output", open=True):
                        final_image_preview = gr.Image(
                            label="Final Model Preview",
                            type="filepath",
                            interactive=False,
                        )
            with gr.Row():
                download_zip = gr.File(  # was visible=True
                    label="Download all results (.zip)",
                    interactive=True,
                    visible=False,
                )
            with gr.Row():
                with gr.Accordion("Adjust Parameters", open=False):
                    args_for_accordion = get_script_args_info(
                        exclude_args=["--input_image"]
                    )

                    for arg in args_for_accordion:
                        label, info, default_val = (
                            f"{arg['name']}",
                            arg["help"],
                            arg.get("default"),
                        )
                        if arg["type"] == "number":
                            accordion_params_dict[arg["name"]] = gr.Number(
                                label=label,
                                value=default_val,
                                info=info,
                                minimum=arg.get("min"),
                                maximum=arg.get("max"),
                                step=arg.get(
                                    "step",
                                    0.001 if isinstance(default_val, float) else 1,
                                ),
                                precision=arg.get("precision", None),
                            )
                        elif arg["type"] == "slider":
                            accordion_params_dict[arg["name"]] = gr.Slider(
                                label=label,
                                value=default_val,
                                info=info,
                                minimum=arg.get("min", 0),
                                maximum=arg.get("max", 1),
                                step=arg.get("step", 0.01),
                            )
                        elif arg["type"] == "checkbox":
                            accordion_params_dict[arg["name"]] = gr.Checkbox(
                                label=label, value=default_val, info=info
                            )
                        elif arg["type"] == "colorpicker":
                            accordion_params_dict[arg["name"]] = gr.ColorPicker(
                                label=label, value=default_val, info=info
                            )
                        else:
                            accordion_params_dict[arg["name"]] = gr.Textbox(
                                label=label, value=str(default_val), info=info
                            )
                        accordion_params_ordered_names.append(arg["name"])

            run_button = gr.Button(
                "Run Autoforge Process",
                variant="primary",
                elem_id="run_button_full_width",
            )


            progress_output = gr.Textbox(
                label="Console Output",
                lines=15,
                autoscroll=True,
                show_copy_button=False,
            )

    # --- Backend Function for Running the Script ---
    def execute_autoforge_script(
        current_filaments_df_state_val, input_image_path, *accordion_param_values
    ):
        # 0. Validate Inputs
        if (
            not input_image_path
        ):  # Covers None and empty string from gr.Image(type="filepath")
            gr.Error("Input Image is required! Please upload an image.")
            return create_empty_error_outputs("Error: Input Image is required!")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir_val = os.path.join(GRADIO_OUTPUT_BASE_DIR, f"run_{timestamp}")
        os.makedirs(run_output_dir_val, exist_ok=True)
        current_run_output_dir.value = run_output_dir_val

        # 1. Save current filaments
        if (
            current_filaments_df_state_val is None
            or current_filaments_df_state_val.empty
        ):
            gr.Error("Filament table is empty. Please add filaments.")
            return create_empty_error_outputs("Error: Filament table is empty.")

        temp_filament_csv = os.path.join(run_output_dir_val, "materials.csv")
        df_to_save = current_filaments_df_state_val.copy()
        required_cols = ["Brand", " Name", " TD", " Color"]
        missing_cols = [col for col in required_cols if col not in df_to_save.columns]
        if missing_cols:
            err_msg = (
                f"Error: Filament data is missing columns: {', '.join(missing_cols)}."
            )
            gr.Error(err_msg)
            return create_empty_error_outputs(err_msg)
        try:
            df_to_save.to_csv(temp_filament_csv, index=False)
        except Exception as e:
            err_msg = f"Error saving temporary filament CSV: {e}"
            gr.Error(err_msg)
            return create_empty_error_outputs(err_msg)

        # 2. Construct command
        python_executable = sys.executable or "python"
        command = ["autoforge",]
        command.extend(["--csv_file", temp_filament_csv])
        command.extend(["--output_folder", run_output_dir_val])
        command.extend(["--disable_visualization_for_gradio","1"])

        base_filename = os.path.basename(input_image_path)
        script_input_image_path = os.path.join(run_output_dir_val, base_filename)
        try:
            shutil.copy(input_image_path, script_input_image_path)
            command.extend(["--input_image", script_input_image_path])
        except Exception as e:
            err_msg = f"Error copying input image: {e}"
            gr.Error(err_msg)
            return create_empty_error_outputs(err_msg)

        param_dict = dict(zip(accordion_params_ordered_names, accordion_param_values))
        for arg_name, arg_widget_val in param_dict.items():
            if arg_widget_val is None or arg_widget_val == "":
                arg_info_list = [
                    item for item in get_script_args_info() if item["name"] == arg_name
                ]  # get full list to check type
                if (
                    arg_info_list
                    and arg_info_list[0]["type"] == "checkbox"
                    and arg_widget_val is False
                ):
                    continue
                else:
                    continue
            if isinstance(arg_widget_val, bool):
                if arg_widget_val:
                    command.append(arg_name)
            else:
                command.extend([arg_name, str(arg_widget_val)])

        # 3. Run script
        log_output = (
            f"Starting Autoforge process at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Output directory: {run_output_dir_val}\n"
            f"Command: {' '.join(command)}\n\n"
        )

        yield create_empty_error_outputs(log_output)  # clear UI and show header

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # ---- helper: read stdout in a background thread -------------------
        from threading import Thread
        from queue import Queue, Empty

        def _enqueue(pipe, q):
            """Forward stdout/stderr to a queue, emitting on both '\n' and '\r'."""
            buf = ""
            while True:
                ch = pipe.read(1)  # read a single character
                if ch == "":  # EOF
                    if buf:
                        q.put(buf)  # flush whatever is left
                    break
                buf += ch
                if ch in ("\n", "\r"):  # tqdm uses '\r'
                    q.put(buf)
                    buf = ""
            pipe.close()

        q_out = Queue()
        Thread(target=_enqueue, args=(process.stdout, q_out), daemon=True).start()
        Thread(target=_enqueue, args=(process.stderr, q_out), daemon=True).start()

        preview_mtime = 0
        last_push = 0

        def _maybe_new_preview():
            """
            If vis_temp.png has a newer mtime than last time, copy it to a
            stamped name (to defeat browser cache) and return that path.
            Otherwise return gr.update() so the image stays as-is.
            """
            from gradio import update  # local import for clarity

            nonlocal preview_mtime

            src = os.path.join(run_output_dir_val, "vis_temp.png")
            if not os.path.exists(src):
                return update()  # nothing new, keep old

            mtime = os.path.getmtime(src)
            if mtime <= preview_mtime:  # unchanged
                return update()  # → no UI update

            return src  # → refresh image

        # ---- main loop: poll every 0.5 s ----------------------------------
        while process.poll() is None or not q_out.empty():
            # drain whatever is waiting in stdout
            try:
                while True:
                    log_output += q_out.get_nowait()
            except Empty:
                pass

            now = time.time()
            if now - last_push >= 1.0:  # 500 ms tick
                current_preview = _maybe_new_preview()
                yield (
                    log_output,
                    current_preview,
                    gr.update(),  # ### ZIP PATCH: placeholder for zip widget
                )
                last_push = now

            time.sleep(0.05)  # keep CPU load low

        return_code = process.wait()
        log_output += (
            "\nAutoforge process completed successfully!"
            if return_code == 0
            else f"\nAutoforge process failed with exit code {return_code}."
        )

        # make sure we show the final preview (if any)
        final_preview = _maybe_new_preview() or os.path.join(
            run_output_dir_val, "final_model.png"
        )

        zip_base = os.path.join(
            run_output_dir_val, "autoforge_results"
        )  # ### ZIP PATCH
        zip_path = shutil.make_archive(zip_base, "zip", run_output_dir_val)

        # 4. Prepare output file paths
        png_path = os.path.join(run_output_dir_val, "final_model.png")
        stl_path = os.path.join(run_output_dir_val, "final_model.stl")
        txt_path = os.path.join(run_output_dir_val, "swap_instructions.txt")
        hfp_path = os.path.join(run_output_dir_val, "project_file.hfp")

        out_png = png_path if os.path.exists(png_path) else None
        out_stl = stl_path if os.path.exists(stl_path) else None
        out_txt = txt_path if os.path.exists(txt_path) else None
        out_hfp = hfp_path if os.path.exists(hfp_path) else None

        if out_png is None:
            log_output += "\nWarning: final_model.png not found in output."

        yield (
            log_output,  # progress_output
            out_png,  # final_image_preview
            gr.update(
                value=zip_path, visible=True, interactive=True
            ),  # ### ZIP PATCH: download_zip
        )

    run_inputs = [filament_df_state, input_image_component] + [
        accordion_params_dict[name] for name in accordion_params_ordered_names
    ]
    run_outputs = [
        progress_output,
        final_image_preview,
        download_zip,  # ### ZIP PATCH: only three outputs now
    ]

    run_button.click(execute_autoforge_script, inputs=run_inputs, outputs=run_outputs)

css = """ #run_button_full_width { width: 100%; } """
if __name__ == "__main__":
    if not os.path.exists(DEFAULT_MATERIALS_CSV):
        print(f"Creating default filament file: {DEFAULT_MATERIALS_CSV}")
        try:
            initial_df.to_csv(DEFAULT_MATERIALS_CSV, index=False)
        except Exception as e:
            print(f"Could not write default {DEFAULT_MATERIALS_CSV}: {e}")
    print("To run the UI, execute: python app.py")  # Corrected to python app.py
    demo.queue().launch(share=False)
