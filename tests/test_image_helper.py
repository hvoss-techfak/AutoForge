import torch
import numpy as np
from autoforge.Helper.ImageHelper import (
    srgb_to_lab,
    increase_saturation,
    resize_image_exact,
)


def test_srgb_to_lab_shape():
    img = torch.randint(0, 256, (16, 16, 3), dtype=torch.float32)
    lab = srgb_to_lab(img)
    assert lab.shape == img.shape
    # L channel range plausibility
    assert lab[..., 0].min() >= -5 and lab[..., 0].max() <= 110


def test_increase_saturation_channel_last():
    img = torch.rand(8, 8, 3)
    out = increase_saturation(img, 0.5)
    assert out.shape == img.shape
    # not identical (unless random unlucky)
    assert torch.mean(torch.abs(out - img)) > 0


def test_increase_saturation_channel_first():
    img = torch.rand(3, 8, 8)
    out = increase_saturation(img, 0.2)
    assert out.shape == img.shape


def test_resize_image_exact():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    out = resize_image_exact(img, 5, 5)
    assert out.shape == (5, 5, 3)
