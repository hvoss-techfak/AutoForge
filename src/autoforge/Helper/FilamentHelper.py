import uuid

import numpy as np
import pandas as pd
import torch


def load_materials(csv_filename):
    """
    Load material data from a CSV file.

    Args:
        csv_filename (str): Path to the hueforge CSV file containing material data.

    Returns:
        tuple: A tuple containing:
            - material_colors (jnp.ndarray): Array of material colors in float64.
            - material_TDs (jnp.ndarray): Array of material transmission/opacity parameters in float64.
            - material_names (list): List of material names.
            - colors_list (list): List of color hex strings.
    """
    df = pd.read_csv(csv_filename)
    material_names = [
        brand + " - " + name
        for brand, name in zip(df["Brand"].tolist(), df[" Name"].tolist())
    ]
    material_TDs = (df[" TD"].astype(float)).to_numpy()
    colors_list = df[" Color"].tolist()
    # Use float64 for material colors.
    material_colors = np.array(
        [hex_to_rgb(color) for color in colors_list], dtype=np.float64
    )
    material_TDs = np.array(material_TDs, dtype=np.float64)
    return material_colors, material_TDs, material_names, colors_list


def load_materials_data(csv_filename):
    """
    Load the full material data from the CSV file.

    Args:
        csv_filename (str): Path to the CSV file containing material data.

    Returns:
        list: A list of dictionaries (one per material) with keys such as
              "Brand", "Type", "Color", "Name", "TD", "Owned", and "Uuid".
    """
    df = pd.read_csv(csv_filename)
    # Use a consistent key naming. For example, convert 'TD' to 'Transmissivity' and 'Uuid' to 'uuid'
    records = df.to_dict(orient="records")
    return records


def hex_to_rgb(hex_str):
    """
    Convert a hex color string to a normalized RGB list.

    Args:
        hex_str (str): The hex color string (e.g., '#RRGGBB').

    Returns:
        list: A list of three floats representing the RGB values normalized to [0, 1].
    """
    hex_str = hex_str.lstrip("#")
    return [int(hex_str[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def extract_colors_from_swatches(swatch_data):
    # we keep only data with transmission distance
    swatch_data = [swatch for swatch in swatch_data if swatch["td"]]

    # For now we load it and convert it in the same way as the hueforge csv files
    out = {}
    for swatch in swatch_data:
        brand = swatch["manufacturer"]["name"]
        name = swatch["color_name"]
        color = swatch["hex_color"]
        td = swatch["td"]
        out[(brand, name)] = (color, td)

    # convert to the same format as the hueforge csv files
    material_names = [brand + " - " + name for (brand, name) in out.keys()]
    material_colors = np.array(
        [hex_to_rgb("#" + color) for color, _ in out.values()], dtype=np.float64
    )
    material_TDs = np.array([td for _, td in out.values()], dtype=np.float64)
    colors_list = [color for color, _ in out.values()]

    return material_colors, material_TDs, material_names, colors_list


def swatch_data_to_table(swatch_data):
    """
    Converts swatch JSON data into a table (list of dicts) with columns:
    "Brand", "Name", "Transmission Distance", "Hex Color".
    """
    table = []
    for swatch in swatch_data:
        if not swatch["td"]:
            continue
        brand = swatch["manufacturer"]["name"]
        name = swatch["color_name"]
        hex_color = swatch["hex_color"]
        td = swatch["td"]
        table.append(
            {
                "Brand": brand,
                "Name": name,
                "TD": td,
                "HexColor": f"#{hex_color}",
                "Uuid": str(uuid.uuid4()),
            }
        )
    return table


def count_distinct_colors(dg: torch.Tensor) -> int:
    """
    Count how many distinct color/material IDs appear in dg.
    """
    unique_mats = torch.unique(dg)
    return len(unique_mats)


def count_swaps(dg: torch.Tensor) -> int:
    """
    Count how many color changes (swaps) occur between adjacent layers.
    """
    # A 'swap' is whenever dg[i] != dg[i+1].
    return int((dg[:-1] != dg[1:]).sum().item())
