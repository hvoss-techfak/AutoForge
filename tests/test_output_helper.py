import json
import types
import pytest

np = pytest.importorskip("numpy")
pandas = pytest.importorskip("pandas")
# trimesh is used inside generate_stl; if missing, test will be skipped by importorskip
trimesh = pytest.importorskip("trimesh")

from autoforge.Helper.OutputHelper import (
    extract_filament_swaps,
    generate_swap_instructions,
    generate_project_file,
    generate_stl,
)


def make_args(tmp_path):
    csv_path = tmp_path / "materials.csv"
    df = pandas.DataFrame(
        [
            {
                "Brand": "BrandA",
                "Name": "Mat1",
                "Transmissivity": 0.5,
                "Color": "#112233",
            },
            {
                "Brand": "BrandB",
                "Name": "Mat2",
                "Transmissivity": 1.0,
                "Color": "#445566",
            },
            {
                "Brand": "BrandC",
                "Name": "Mat3",
                "Transmissivity": 0.1,
                "Color": "#778899",
            },
        ]
    )
    df.to_csv(csv_path, index=False)
    return types.SimpleNamespace(
        csv_file=str(csv_path),
        json_file="",
        layer_height=0.04,
        background_height=0.24,
        max_layers=10,
        background_color="#000000",
    )


def test_extract_filament_swaps_simple():
    disc_global = np.array([0, 0, 1, 1, 2, 2])
    disc_height = np.array([[0, 1], [1, 2]])
    filament_indices, slider_values = extract_filament_swaps(
        disc_global, disc_height, background_layers=6
    )
    assert filament_indices[0] == 0
    assert filament_indices[-1] == filament_indices[-2]
    assert slider_values[0] == 1
    assert slider_values[-1] == slider_values[-2] + 1


def test_generate_swap_instructions():
    disc_global = np.array([0, 0, 1, 1, 2, 2])
    disc_height = np.array([[0, 1], [1, 2]])
    names = ["A - Mat1", "B - Mat2", "C - Mat3"]
    instr = generate_swap_instructions(disc_global, disc_height, 0.04, 6, 0.24, names)
    assert any("swap" in line.lower() for line in instr)
    assert instr[-1].startswith("For the rest")


def test_generate_project_file(tmp_path):
    args = make_args(tmp_path)
    disc_global = np.array([0, 1, 1, 2, 2, 2])
    disc_height = np.array([[0, 1], [2, 3]])
    project_path = tmp_path / "project.hfp"
    stl_path = tmp_path / "model.stl"
    # create a small valid STL via our generator
    dummy = np.zeros((2, 2), dtype=np.float32)
    generate_stl(dummy, stl_path, args.background_height, maximum_x_y_size=10)
    generate_project_file(
        str(project_path),
        args,
        disc_global,
        disc_height,
        50,
        40,
        str(stl_path),
        args.csv_file,
    )
    data = json.loads(project_path.read_text())
    assert "filament_set" in data
    assert data["layer_height"] == args.layer_height
    assert data["slider_values"]


def test_generate_stl_basic(tmp_path):
    height_map = np.random.rand(5, 7).astype(np.float32)
    out_path = tmp_path / "out.stl"
    generate_stl(height_map, out_path, background_height=0.2, maximum_x_y_size=50)
    assert out_path.exists()
    # basic size check (header 80 + uint32 count + triangles)
    assert out_path.stat().st_size > 84
