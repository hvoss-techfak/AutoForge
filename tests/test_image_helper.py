import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from autoforge.Helper.ImageHelper import (
    increase_saturation,
    srgb_to_lab,
    resize_image_exact,
)


def test_increase_saturation_channel_last():
    img = torch.rand(16, 16, 3)
    out = increase_saturation(img, 0.5)
    assert out.shape == img.shape
    # Saturation increase should move values away from gray; variance of (img-gray) scaled
    gray = (img * torch.tensor([0.2989, 0.5870, 0.1140])).sum(-1, keepdim=True)
    orig_diff = (img - gray).abs().mean().item()
    new_diff = (out - gray).abs().mean().item()
    assert new_diff > orig_diff


def test_increase_saturation_channel_first():
    img = torch.rand(3, 8, 8)
    out = increase_saturation(img, 0.2)
    assert out.shape == img.shape


def test_srgb_to_lab_round_trip_shape():
    img = torch.randint(0, 255, (10, 10, 3), dtype=torch.uint8).to(torch.float32)
    lab = srgb_to_lab(img)
    assert lab.shape == img.shape
    # L channel roughly within [0,100]
    assert lab[..., 0].min() >= -1 and lab[..., 0].max() <= 110


def test_resize_image_exact():
    arr = (np.random.rand(20, 30, 3) * 255).astype(np.uint8)
    out = resize_image_exact(arr, 10, 15)
    assert out.shape == (15, 10, 3)
