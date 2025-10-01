import numpy as np
import torch
import os

from autoforge.Helper.FilamentHelper import (
    load_materials,
    hex_to_rgb,
    extract_colors_from_swatches,
    swatch_data_to_table,
    count_distinct_colors,
    count_swaps,
)
from autoforge.Helper.OutputHelper import (
    extract_filament_swaps,
    generate_swap_instructions,
    generate_stl,
)


def test_load_materials(dummy_args):
    colors, tds, names, hexes = load_materials(dummy_args)
    assert colors.shape[0] == 3
    assert tds.shape[0] == 3
    assert len(names) == 3
    assert len(hexes) == 3


def test_hex_to_rgb():
    rgb = hex_to_rgb("#FFFFFF")
    assert rgb == [1.0, 1.0, 1.0]


def test_extract_colors_from_swatches():
    swatches = [
        {
            "td": 0.5,
            "manufacturer": {"name": "BrandX"},
            "color_name": "Redish",
            "hex_color": "ff0000",
        },
        {
            "td": 0.7,
            "manufacturer": {"name": "BrandY"},
            "color_name": "Greenish",
            "hex_color": "00ff00",
        },
    ]
    colors, tds, names, hexes = extract_colors_from_swatches(swatches)
    assert colors.shape[0] == 2
    assert tds.tolist() == [0.5, 0.7]


def test_swatch_data_to_table():
    swatches = [
        {
            "td": 0.4,
            "manufacturer": {"name": "B"},
            "color_name": "Blue",
            "hex_color": "0000ff",
        }
    ]
    table = swatch_data_to_table(swatches)
    assert table[0]["HexColor"] == "#0000ff"


def test_count_functions():
    dg = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    assert count_distinct_colors(dg) == 3
    assert count_swaps(dg) == 2


def test_extract_filament_swaps():
    disc_global = np.array([0, 0, 1, 1, 2, 2, 2, 1, 1])
    height_map = np.zeros((4, 4), dtype=int)
    height_map[0, 0] = 5  # max layer L=5
    filament_indices, slider_values = extract_filament_swaps(
        disc_global, height_map, background_layers=1
    )
    assert filament_indices[0] == disc_global[0]
    assert len(slider_values) == len(filament_indices)


def test_generate_swap_instructions():
    disc_global = np.array([0, 0, 1, 1, 2, 2, 2])
    height_map = np.zeros((4, 4), dtype=int)
    height_map[0, 0] = 5
    material_names = ["A", "B", "C"]
    inst = generate_swap_instructions(
        disc_global,
        height_map,
        h=0.2,
        background_layers=1,
        background_height=0.6,
        material_names=material_names,
    )
    assert any("swap to" in s.lower() for s in inst)


def test_generate_stl(tmp_path):
    hm = np.zeros((4, 4), dtype=float)
    hm[1:3, 1:3] = 1.0
    filename = tmp_path / "test.stl"
    generate_stl(hm, str(filename), background_height=0.2, maximum_x_y_size=10.0)
    assert os.path.exists(filename)
    assert os.path.getsize(filename) > 100  # some bytes
