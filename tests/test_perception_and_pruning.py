import torch
import pytest

from autoforge.Loss.PerceptionLoss import MultiLayerVGGPerceptualLoss
from autoforge.Helper.PruningHelper import (
    disc_to_logits,
    merge_color,
    find_color_bands,
    merge_bands,
    remove_outlier_pixels,
    smooth_coplanar_faces,
)


@pytest.fixture(autouse=True)
def mock_vgg16(monkeypatch):
    # Replace torchvision.models.vgg16 to avoid weight download.
    import torchvision.models as models

    class DummyFeatures(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.Conv2d(3, 3, 3, padding=1),  # idx 0
                    torch.nn.ReLU(),  # 1
                    torch.nn.Conv2d(3, 3, 3, padding=1),  # 2
                    torch.nn.ReLU(),  # 3
                    torch.nn.Conv2d(3, 3, 3, padding=1),  # 4
                    torch.nn.ReLU(),  # 5
                    torch.nn.Conv2d(3, 3, 3, padding=1),  # 6
                    torch.nn.ReLU(),  # 7
                    torch.nn.Conv2d(3, 3, 3, padding=1),  # 8
                ]
            )

        def __getitem__(self, idx):
            return self.layers[idx]

        def __iter__(self):
            return iter(self.layers)

        def __len__(self):
            return len(self.layers)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = DummyFeatures()

    def dummy_vgg16(weights=None):
        return DummyModel()

    monkeypatch.setattr(models, "vgg16", dummy_vgg16)
    yield


def test_perceptual_loss_forward():
    loss_mod = MultiLayerVGGPerceptualLoss(layers=[8])
    x = torch.rand(1, 3, 32, 32) * 255.0
    y = x.clone()
    out = loss_mod(x, y)
    assert torch.isfinite(out)
    assert out.item() >= 0.0


def test_disc_to_logits():
    dg = torch.tensor([0, 1, 0, 2])
    logits = disc_to_logits(dg, num_materials=3)
    assert logits.shape == (4, 3)
    # Chosen material should have max
    for i, row in enumerate(logits):
        assert torch.argmax(row).item() == dg[i].item()


def test_merge_color():
    dg = torch.tensor([0, 1, 2, 1])
    merged = merge_color(dg, 1, 0)
    assert (merged == torch.tensor([0, 0, 2, 0])).all()


def test_find_color_bands_and_merge_bands():
    dg = torch.tensor([0, 0, 1, 1, 2, 2, 1])
    bands = find_color_bands(dg)
    # Expect 4 bands
    assert len(bands) == 4
    # Merge first two bands forward
    merged = merge_bands(dg, bands[0], bands[1], direction="forward")
    assert merged[0:4].tolist() == [0, 0, 0, 0]


def test_remove_outlier_pixels():
    h = torch.zeros(8, 8)
    h[4, 4] = 10.0
    cleaned = remove_outlier_pixels(h, threshold=5.0)
    assert cleaned[4, 4] != 10.0  # outlier corrected


def test_smooth_coplanar_faces():
    base = torch.ones(8, 8)
    base[4, 4] = 1.5  # small bump
    smoothed = smooth_coplanar_faces(base, angle_threshold=10)
    assert smoothed[4, 4] < base[4, 4]
