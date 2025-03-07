import streamlit as st
import sqlite3
import json
import uuid
import os
import pandas as pd
import ast
import hashlib

# -------------------------------
# Database and Initialization
# -------------------------------

DATABASE = "swatches.db"
JSON_FILE = "../swatches.json"


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS swatches (
            Uuid TEXT PRIMARY KEY,
            Brand TEXT,
            Name TEXT,
            TD REAL,
            HexColor TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS owned_filaments (
            Uuid TEXT PRIMARY KEY,
            Brand TEXT,
            Name TEXT,
            TD REAL,
            HexColor TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()


def import_json_data():
    if not os.path.exists(JSON_FILE):
        return
    mtime = os.path.getmtime(JSON_FILE)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM meta WHERE key = ?", ("json_mtime",))
    row = cursor.fetchone()
    stored_mtime = float(row["value"]) if row else 0

    if mtime > stored_mtime:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            cursor.execute("SELECT 1 FROM swatches WHERE Uuid = ?", (item["Uuid"],))
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO swatches (Uuid, Brand, Name, TD, HexColor) VALUES (?, ?, ?, ?, ?)",
                    (
                        item["Uuid"],
                        item["Brand"],
                        item["Name"],
                        item["TD"],
                        item["HexColor"],
                    ),
                )
        cursor.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("json_mtime", str(mtime)),
        )
        conn.commit()
    conn.close()


# Initialize DB and import JSON data
init_db()
import_json_data()

# -------------------------------
# Backend CRUD Functions
# -------------------------------


def get_items():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM swatches")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_item(item):
    new_uuid = str(uuid.uuid4())
    hex_color = (
        item["HexColor"] if item["HexColor"].startswith("#") else "#" + item["HexColor"]
    )
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO swatches (Uuid, Brand, Name, TD, HexColor) VALUES (?, ?, ?, ?, ?)",
        (new_uuid, item["Brand"], item["Name"], item["TD"], hex_color),
    )
    conn.commit()
    conn.close()
    return {
        "Uuid": new_uuid,
        "Brand": item["Brand"],
        "Name": item["Name"],
        "TD": item["TD"],
        "HexColor": hex_color,
    }


def update_item(uuid_val, update_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM swatches WHERE Uuid = ?", (uuid_val,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        st.error("Item not found")
        return None

    updated_brand = update_data.get("Brand") or row["Brand"]
    updated_name = update_data.get("Name") or row["Name"]
    updated_td = (
        update_data.get("TD") if update_data.get("TD") is not None else row["TD"]
    )
    updated_hex = update_data.get("HexColor") or row["HexColor"]
    if not updated_hex.startswith("#"):
        updated_hex = "#" + updated_hex

    cursor.execute(
        "UPDATE swatches SET Brand = ?, Name = ?, TD = ?, HexColor = ? WHERE Uuid = ?",
        (updated_brand, updated_name, updated_td, updated_hex, uuid_val),
    )
    conn.commit()
    cursor.execute("SELECT * FROM swatches WHERE Uuid = ?", (uuid_val,))
    updated_row = cursor.fetchone()
    conn.close()
    return dict(updated_row)


def delete_item(uuid_val):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM swatches WHERE Uuid = ?", (uuid_val,))
    if cursor.rowcount == 0:
        conn.close()
        st.error("Item not found")
        return False
    conn.commit()
    conn.close()
    return True


def get_owned_filaments():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM owned_filaments")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_owned_filament(item):
    new_uuid = str(uuid.uuid4())
    hex_color = (
        item["HexColor"] if item["HexColor"].startswith("#") else "#" + item["HexColor"]
    )
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO owned_filaments (Uuid, Brand, Name, TD, HexColor) VALUES (?, ?, ?, ?, ?)",
        (new_uuid, item["Brand"], item["Name"], item["TD"], hex_color),
    )
    conn.commit()
    conn.close()
    return {
        "Uuid": new_uuid,
        "Brand": item["Brand"],
        "Name": item["Name"],
        "TD": item["TD"],
        "HexColor": hex_color,
    }


def delete_owned_filament(uuid_val):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM owned_filaments WHERE Uuid = ?", (uuid_val,))
    if cursor.rowcount == 0:
        conn.close()
        st.error("Owned filament not found")
        return False
    conn.commit()
    conn.close()
    return True


# -------------------------------
# Helper for Dynamic AgGrid Key
# -------------------------------


def get_grid_key(df):
    # Create a hash from the DataFrame contents
    hash_obj = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    )
    return hash_obj.hexdigest()


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(layout="wide")
st.title("Swatch Editor with Owned Filaments (Persistent)")

# Always load fresh data from the database
swatches_data = pd.DataFrame(get_items())
owned_data = pd.DataFrame(get_owned_filaments())

# Configure AgGrid grids (same configuration as before)
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

gb_main = GridOptionsBuilder.from_dataframe(swatches_data)
gb_main.configure_default_column(editable=True)
gb_main.configure_selection("multiple", use_checkbox=True)

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
gb_main.configure_column("HexColor", cellRenderer=hex_renderer, cellEditor=hex_renderer)
gridOptions_main = gb_main.build()

if not owned_data.empty:
    gb_owned = GridOptionsBuilder.from_dataframe(owned_data)
else:
    gb_owned = GridOptionsBuilder.from_dataframe(
        pd.DataFrame(columns=swatches_data.columns)
    )
gb_owned.configure_selection("multiple", use_checkbox=True)
gb_owned.configure_column(
    "HexColor", cellRenderer=hex_renderer, cellEditor=hex_renderer
)
gridOptions_owned = gb_owned.build()

# Generate a dynamic key for the AgGrid based on current swatches_data
grid_key = get_grid_key(swatches_data)

col1, col2 = st.columns(2)

with col1:
    st.subheader("All Swatches")
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
        for row in selected_rows:
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
            st.success(f"Added {added_count} swatch(es) to owned filaments.")
            st.rerun()
        else:
            st.warning("No new swatches were added to owned.")

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
            for row in selected_owned:
                if delete_owned_filament(row["Uuid"]):
                    removed_count += 1
            if removed_count > 0:
                st.success(f"Removed {removed_count} swatch(es) from owned filaments.")
                st.rerun()
            else:
                st.warning("No swatches selected for removal.")
    else:
        st.write("No owned filaments yet.")

st.markdown("### Save Changes")
if st.button("Save Changes"):
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

st.markdown("### Add New Swatch")
with st.form("add_swatch_form"):
    new_brand = st.text_input("Brand")
    new_name = st.text_input("Name")
    new_td = st.number_input("TD", value=0.0, step=0.1)
    new_hex = st.color_picker("Hex Color", "#ffffff")
    submitted = st.form_submit_button("Add Swatch")
    if submitted:
        new_data = {
            "Brand": new_brand,
            "Name": new_name,
            "TD": new_td,
            "HexColor": new_hex,
        }
        added = add_item(new_data)
        if added:
            st.success("Swatch added.")
            # Force a re-run so the grid is refreshed with new data.
            st.rerun()
        else:
            st.error("Error adding swatch.")
