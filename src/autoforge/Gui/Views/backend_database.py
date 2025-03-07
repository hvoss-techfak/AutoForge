import json
import os
import uuid

import streamlit as st
import sqlite3


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


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


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
