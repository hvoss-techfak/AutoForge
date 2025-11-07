import json
import numpy as np
import os
import argparse

from autoforge.Helper.OutputHelper import (
    extract_filament_swaps,
    generate_swap_instructions,
    generate_project_file,
    generate_stl,
)


def _args():
    ns = argparse.Namespace()
    ns.background_height = 0.5
    ns.layer_height = 0.2
    ns.max_layers = 5
    ns.background_color = "#000000"
    return ns


def test_extract_filament_swaps_and_instructions():
    disc_global = np.array([0, 0, 1, 1, 2, 2])
    disc_height = np.array([0, 1, 2, 3, 4, 5])
    filament_indices, slider_values = extract_filament_swaps(
        disc_global, disc_height, background_layers=2
    )
    assert filament_indices[0] == 0
    assert slider_values[0] == 1
    assert filament_indices[-1] == filament_indices[-2]
    names = ["Mat0", "Mat1", "Mat2"]
    instr = generate_swap_instructions(
        disc_global,
        disc_height,
        h=0.2,
        background_layers=2,
        background_height=0.5,
        material_names=names,
    )
    assert any("swap" in s for s in instr)
    assert instr[-1].startswith("For the rest")


def test_generate_project_file_and_stl(tmp_path):
    # Setup simple material CSV
    csv_path = tmp_path / "materials.csv"
    import pandas as pd

    df = pd.DataFrame(
        {
            "Brand": ["B0", "B1", "B2"],
            "Name": ["N0", "N1", "N2"],
            "Transmissivity": [1.0, 2.0, 3.0],
            "Color": ["#000000", "#111111", "#222222"],
        }
    )
    df.to_csv(csv_path, index=False)

    args = _args()
    args.csv_file = str(csv_path)
    args.json_file = ""

    disc_global = np.array([0, 1, 1, 2, 2, 2])
    disc_height = np.array([0, 1, 2, 3, 4, 5])

    proj_path = tmp_path / "proj.json"
    stl_path = tmp_path / "mesh.stl"
    generate_project_file(
        str(proj_path),
        args,
        disc_global,
        disc_height,
        width_mm=10.0,
        height_mm=5.0,
        stl_filename=str(stl_path),
        csv_filename=str(csv_path),
    )
    data = json.loads(proj_path.read_text())
    assert "filament_set" in data and len(data["filament_set"]) >= 2
    assert data["stl"] == os.path.basename(str(stl_path))

    # Simple height map for STL
    height_map = np.array([[0, 1], [2, 3]], dtype=np.float32)
    generate_stl(
        height_map, str(stl_path), background_height=0.5, maximum_x_y_size=10.0
    )
    assert stl_path.exists() and stl_path.stat().st_size > 0
