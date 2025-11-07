import types
import pytest

# Optional heavy deps
pandas = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from autoforge.Helper.FilamentHelper import (
    hex_to_rgb,
    count_distinct_colors,
    count_swaps,
    load_materials,
    load_materials_data,
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
    return types.SimpleNamespace(csv_file=str(csv_path), json_file="")


def test_hex_to_rgb():
    rgb = hex_to_rgb("#8040ff")
    assert len(rgb) == 3
    assert all(0.0 <= c <= 1.0 for c in rgb)
    # Exact conversion check
    assert rgb[0] == 128 / 255 and rgb[1] == 64 / 255 and rgb[2] == 255 / 255


def test_count_distinct_colors():
    t = torch.tensor([0, 0, 1, 2, 2, 2, 5])
    assert count_distinct_colors(t) == 4


def test_count_swaps():
    # swaps when value changes from previous
    t = torch.tensor([1, 1, 2, 2, 3, 3, 3, 1])
    # Changes at indices 1->2, 3->4, 6->7 = 3 swaps
    assert count_swaps(t) == 3


def test_load_materials_and_data(tmp_path):
    args = make_args(tmp_path)
    colors, tds, names, colors_list = load_materials(args)
    assert colors.shape == (3, 3)
    assert tds.shape == (3,)
    assert len(names) == 3
    assert len(colors_list) == 3
    data_records = load_materials_data(args)
    assert isinstance(data_records, list)
    assert len(data_records) == 3
    assert {r["Brand"] for r in data_records} == {"BrandA", "BrandB", "BrandC"}
