import ast

import streamlit as st
import pandas as pd
import hashlib

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from autoforge.Gui.Views.backend_database import (
    get_items,
    get_owned_filaments,
    add_owned_filament,
    delete_owned_filament,
    update_item,
    add_item,
)


def get_grid_key(df):
    # Create a hash from the DataFrame contents
    hash_obj = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    )
    return hash_obj.hexdigest()


hex_renderer = JsCode("""
class ColorCellEditor {
  init(params) {
    this.eInput = document.createElement('input');
    this.eInput.type = 'color';
    this.eInput.value = params.value;
    this.eInput.style.width = '60px';
    this.eInput.style.height = '25px';
    this.eInput.style.padding = '0';
    this.eInput.style.margin = '0';
    this.eInput.style.border = 'none';
    this.eInput.style.outline = 'none';
    this.eInput.style.borderRadius = '0';
    this.eInput.style.boxShadow = 'none';
    this.eInput.style.appearance = 'none';
    this.eInput.style.webkitAppearance = 'none';
    this.eInput.style.MozAppearance = 'none';
  }
  getGui() {
    return this.eInput;
  }
  afterGuiAttached() {
    this.eInput.focus();
  }
  getValue() {
    return this.eInput.value;
  }
  destroy() {}
  isPopup() {
    return false;
  }
}
""")


def show_filament_editor():
    st.title("Filament Editor")

    cols_order = ["Brand", "Name", "TD", "HexColor", "Uuid"]
    # Always load fresh data from the database
    swatches_data = pd.DataFrame(get_items())[cols_order]
    owned_data = pd.DataFrame(get_owned_filaments())[cols_order]

    # Configure AgGrid for swatches
    gb_main = GridOptionsBuilder.from_dataframe(swatches_data)
    gb_main.configure_default_column(editable=True)
    gb_main.configure_column("Uuid", hide=True)
    gb_main.configure_selection("multiple", use_checkbox=True)
    gb_main.configure_column(
        "HexColor", cellRenderer=hex_renderer, cellEditor=hex_renderer
    )
    gridOptions_main = gb_main.build()

    # Configure AgGrid for owned filaments
    if not owned_data.empty:
        gb_owned = GridOptionsBuilder.from_dataframe(owned_data)
    else:
        gb_owned = GridOptionsBuilder.from_dataframe(
            pd.DataFrame(columns=swatches_data.columns)
        )
    gb_owned.configure_selection("multiple", use_checkbox=True)
    gb_owned.configure_column("Uuid", hide=True)
    gb_owned.configure_column(
        "HexColor", cellRenderer=hex_renderer, cellEditor=hex_renderer
    )
    gridOptions_owned = gb_owned.build()

    # Generate a dynamic key for the AgGrid based on current swatches_data
    grid_key = get_grid_key(swatches_data)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("All Filaments")
        grid_response = AgGrid(
            swatches_data,
            gridOptions=gridOptions_main,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            reload_data=True,
            allow_unsafe_jscode=True,
            height=400,
            key=grid_key,  # dynamic key forces re-render on data change
        )
        if st.button("Add Selected to Owned"):
            selected_rows = grid_response.get("selected_rows", [])
            current_owned = pd.DataFrame(get_owned_filaments())
            added_count = 0
            for row in selected_rows.to_dict("records"):
                exists = False
                if not current_owned.empty:
                    dup = current_owned[
                        (current_owned["Brand"] == row["Brand"])
                        & (current_owned["Name"] == row["Name"])
                        & (current_owned["TD"] == row["TD"])
                        & (current_owned["HexColor"] == row["HexColor"])
                    ]
                    if not dup.empty:
                        exists = True
                if not exists:
                    add_owned_filament(
                        {
                            "Brand": row["Brand"],
                            "Name": row["Name"],
                            "TD": row["TD"],
                            "HexColor": row["HexColor"],
                        }
                    )
                    added_count += 1
            if added_count > 0:
                st.success(f"Added {added_count} filament(s) to owned filaments.")
                st.rerun()
            else:
                st.warning("No new filaments were added to owned.")

    with col2:
        st.subheader("Owned Filaments")
        if not owned_data.empty:
            owned_grid = AgGrid(
                owned_data,
                gridOptions=gridOptions_owned,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                reload_data=True,
                allow_unsafe_jscode=True,
                height=400,
            )
            if st.button("Remove Selected from Owned"):
                selected_owned = owned_grid.get("selected_rows", [])
                removed_count = 0
                for row in selected_owned.to_dict("records"):
                    if delete_owned_filament(row["Uuid"]):
                        removed_count += 1
                if removed_count > 0:
                    st.success(
                        f"Removed {removed_count} filament(s) from owned filaments."
                    )
                    st.rerun()
                else:
                    st.warning("No filaments selected for removal.")
        else:
            st.write("No owned filaments yet.")

    st.markdown("### Save Changes")
    if st.button("Save Changes"):
        # The "data" returned by AgGrid can sometimes be a list of strings or a list of dicts
        updated_data = grid_response["data"]
        if (
            isinstance(updated_data, list)
            and updated_data
            and isinstance(updated_data[0], str)
        ):
            try:
                updated_data = [ast.literal_eval(row) for row in updated_data]
            except Exception as e:
                st.error(f"Error converting row data: {e}")
        updated_df = pd.DataFrame(updated_data)
        for index, row in updated_df.iterrows():
            uuid_val = row["Uuid"]
            update_data = {
                "Brand": row["Brand"],
                "Name": row["Name"],
                "TD": row["TD"],
                "HexColor": row["HexColor"],
            }
            updated = update_item(uuid_val, update_data)
            if not updated:
                st.error(f"Error updating row {uuid_val}")
        st.success("Changes saved.")
        st.rerun()

    st.markdown("### Add New Filament")
    with st.form("add_swatch_form"):
        new_brand = st.text_input("Brand")
        new_name = st.text_input("Name")
        new_td = st.number_input("TD", value=0.0, step=0.1)
        new_hex = st.color_picker("Hex Color", "#ffffff")
        submitted = st.form_submit_button("Add filament")
        if submitted:
            new_data = {
                "Brand": new_brand,
                "Name": new_name,
                "TD": new_td,
                "HexColor": new_hex,
            }
            added = add_item(new_data)
            add_owned_filament(new_data)
            if added:
                st.success("Filament added.")
                st.rerun()
            else:
                st.error("Error adding filament.")
