import streamlit as st
import os
import pandas as pd
from PIL import Image

from autoforge.Helper.filamentcolors_library import download_filament_info

os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from autoforge.Gui.Views.backend_database import (
    get_owned_filaments,
    init_db,
    import_json_data,
)
from autoforge.Gui.Views.filament_editor import hex_renderer, show_filament_editor


def streamlit_visualize(image: Image.Image):
    """
    Update the autoforge preview image in the UI with the provided PIL image.
    Call this function from your image-generation script to update the preview.
    """
    if "preview_placeholder" in st.session_state:
        st.session_state.preview_placeholder.image(image, use_container_width=True)
    else:
        st.error(
            "Preview placeholder not initialized. Make sure the UI is set up correctly."
        )


# -------------------------------
# Database and Initialization
# -------------------------------


def show_autoforge_view():
    st.title("Autoforge")

    # Left: Owned Filaments, Middle: Preview, Right: Multi-Segment Slider
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        cols_order = ["Brand", "Name", "TD", "HexColor", "Uuid"]
        st.subheader("Owned Filaments")
        owned_data = pd.DataFrame(get_owned_filaments())[cols_order]
        if not owned_data.empty:
            gb_owned = GridOptionsBuilder.from_dataframe(owned_data)
            gb_owned.configure_selection("multiple", use_checkbox=True)
            gb_owned.configure_column("Uuid", hide=True)

            gb_owned.configure_column(
                "HexColor", cellRenderer=hex_renderer, cellEditor=hex_renderer
            )
            gridOptions_owned = gb_owned.build()

            owned_grid = AgGrid(
                owned_data,
                gridOptions=gridOptions_owned,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                reload_data=True,
                allow_unsafe_jscode=True,
                height=400,
                fit_columns_on_grid_load=True,
            )

            selected_owned = owned_grid.get("selected_rows", [])
            if selected_owned is not None and len(selected_owned) > 0:
                st.write(f"Selected {len(selected_owned)} filament(s).")
            else:
                st.write("No filaments selected.")
        else:
            st.write("No owned filaments yet.")

    with col2:
        st.subheader("Autoforge Preview")
        # Create (or reuse) a placeholder for the preview image and store it in session state.
        if "preview_placeholder" not in st.session_state:
            st.session_state.preview_placeholder = st.empty()
        # Initially display a default placeholder image.
        st.session_state.preview_placeholder.image(
            "https://via.placeholder.com/600x400?text=Autoforge+Preview",
            use_container_width=True,
        )

    with col3:
        st.subheader("Multi-Segment Slider (Concept)")
        if "segment_handles" not in st.session_state:
            st.session_state.segment_handles = [0.2, 0.8]

        new_handles = []
        for i, val in enumerate(st.session_state.segment_handles):
            new_val = st.slider(
                f"Handle {i + 1}",
                min_value=0.0,
                max_value=1.0,
                value=val,
                step=0.01,
            )
            new_handles.append(new_val)
        new_handles.sort()
        st.session_state.segment_handles = new_handles
        st.write("Handles:", st.session_state.segment_handles)

    # ----------------------------
    # Configuration Settings Drawer
    # ----------------------------
    with st.expander("Configuration Settings", expanded=False):
        st.number_input(
            "Number of optimization iterations", value=5000, key="iterations"
        )
        st.number_input("Learning rate", value=1e-2, format="%.5f", key="learning_rate")
        st.number_input(
            "Layer thickness (mm)", value=0.04, format="%.5f", key="layer_height"
        )
        st.number_input("Maximum number of layers", value=75, key="max_layers")
        st.number_input(
            "Minimum number of layers (for pruning)", value=0, key="min_layers"
        )
        st.number_input(
            "Background height (mm)", value=0.4, format="%.5f", key="background_height"
        )
        st.color_picker("Background color", "#000000", key="background_color")
        st.number_input("Max dimension for target image", value=1024, key="output_size")
        st.number_input("Max dimension for solver image", value=128, key="solver_size")
        st.number_input(
            "Final tau value for Gumbel-Softmax", value=0.01, format="%.5f", key="decay"
        )
        st.checkbox(
            "Perform pruning after optimization", value=True, key="perform_pruning"
        )
        st.number_input(
            "Max colors allowed after pruning", value=100, key="pruning_max_colors"
        )
        st.number_input(
            "Max swaps allowed after pruning", value=100, key="pruning_max_swaps"
        )
        st.number_input(
            "Random seed (0 for automatic generation)", value=0, key="random_seed"
        )

    # st.write(
    #     "Current settings:",
    #     {
    #         "iterations": st.session_state.iterations,
    #         "learning_rate": st.session_state.learning_rate,
    #         "layer_height": st.session_state.layer_height,
    #         "max_layers": st.session_state.max_layers,
    #         "min_layers": st.session_state.min_layers,
    #         "background_height": st.session_state.background_height,
    #         "background_color": st.session_state.background_color,
    #         "output_size": st.session_state.output_size,
    #         "solver_size": st.session_state.solver_size,
    #         "decay": st.session_state.decay,
    #         "visualize": st.session_state.visualize,
    #         "perform_pruning": st.session_state.perform_pruning,
    #         "pruning_max_colors": st.session_state.pruning_max_colors,
    #         "pruning_max_swaps": st.session_state.pruning_max_swaps,
    #         "random_seed": st.session_state.random_seed,
    #     },
    # )

    # Control buttons for process management.
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Pause/Resume"):
            st.write("Pause/Resume clicked. (Demo placeholder)")
    with colB:
        if st.button("Stop"):
            st.write("Stop clicked. (Demo placeholder)")
    with colC:
        if st.button("Pruning"):
            st.write("Pruning clicked. (Demo placeholder)")


# -------------------------------------------------
# Main
# -------------------------------------------------


def main():
    print("Downloading filament info...")
    print("If this is your first time running this script, it may take a few minutes.")
    download_filament_info()
    # Initialize DB and import JSON data
    init_db()
    import_json_data()

    st.set_page_config(layout="wide")

    # Side drawer (Streamlit's sidebar):
    view_choice = st.sidebar.radio("Choose a view:", ("Autoforge", "Filament Editor"))

    if view_choice == "Filament Editor":
        show_filament_editor()
    else:
        show_autoforge_view()


if __name__ == "__main__":
    main()
