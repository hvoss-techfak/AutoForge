import os
import sys
import types
import numpy as np
import pytest

# Ensure src is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_image():
    # 32x32 random uint8 image
    return np.random.default_rng(0).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


@pytest.fixture
def mock_depth_pipeline(monkeypatch):
    # Ensure both the transformers.pipeline function and the module-local imported symbol are patched
    def dummy_pipeline(*args, **kwargs):
        class Dummy:
            def __call__(self, image):
                import numpy as _np
                from PIL import Image as _Image

                arr = _np.array(image)
                H, W = arr.shape[:2]
                grad = _np.linspace(0, 1, H, dtype=_np.float32).reshape(H, 1)
                depth = (_np.repeat(grad, W, axis=1) * 255).astype("uint8")
                return {"depth": _Image.fromarray(depth)}

        return Dummy()

    monkeypatch.setattr("transformers.pipeline", dummy_pipeline, raising=True)
    # Patch module-local pipeline if module already imported
    try:
        import autoforge.Helper.Heightmaps.DepthEstimateHeightMap as dehm

        monkeypatch.setattr(dehm, "pipeline", dummy_pipeline, raising=True)
    except Exception:
        pass
    return True


@pytest.fixture
def dummy_args(tmp_path):
    # Minimal args namespace for functions expecting many attributes
    ns = types.SimpleNamespace()
    ns.background_height = 0.6
    ns.layer_height = 0.2
    ns.max_layers = 8
    ns.background_color = "#000000"
    ns.csv_file = str(tmp_path / "materials.csv")
    ns.json_file = ""
    # create a minimal materials CSV
    with open(ns.csv_file, "w") as f:
        f.write("Brand,Name,Color,Transmissivity\n")
        f.write("A,MatA,#FF0000,0.5\n")
        f.write("B,MatB,#00FF00,0.7\n")
        f.write("C,MatC,#0000FF,0.9\n")
    return ns


@pytest.fixture
def cpu_device():
    import torch

    return torch.device("cpu")


@pytest.fixture
def material_data_tensors():
    import torch

    material_colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    material_TDs = torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32)
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    return material_colors, material_TDs, background
