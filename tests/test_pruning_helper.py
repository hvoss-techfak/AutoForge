import torch

from autoforge.Helper.PruningHelper import (
    disc_to_logits,
    merge_color,
    find_color_bands,
    merge_bands,
    remove_outlier_pixels,
    smooth_coplanar_faces,
)


class DummyOptimizer:
    def __init__(
        self, target, material_colors, material_TDs, background, H=4, W=4, L=4
    ):
        self.target = target
        self.material_colors = material_colors
        self.material_TDs = material_TDs
        self.background = background
        self.h = 0.2
        self.vis_tau = 0.5
        self.best_seed = 0
        self.max_layers = L
        self.best_params = {
            "pixel_height_logits": torch.zeros(H, W),
            "global_logits": torch.zeros(L, material_colors.shape[0]),
            "height_offsets": torch.zeros(L, 1),
        }

    def _apply_height_offset(self, logits, offsets):
        return logits

    def discretize_solution(self, params, tau_g, h, max_layers, rng_seed):
        # Simple: return sequential global ids and heights as zeros
        L = params["global_logits"].shape[0]
        dg = torch.arange(L)
        dh = torch.zeros(4, 4, dtype=torch.int32)
        return dg, dh

    def get_best_discretized_image(
        self, custom_height_logits=None, custom_global_logits=None
    ):
        # Return a constant image depending on chosen material indices
        return torch.zeros(4, 4, 3)


def _basic_dummy():
    target = torch.zeros(4, 4, 3)
    material_colors = torch.rand(3, 3)
    material_TDs = torch.ones(3)
    background = torch.zeros(3)
    return DummyOptimizer(target, material_colors, material_TDs, background)


def test_disc_to_logits_and_merge_color():
    dg = torch.tensor([0, 1, 2, 1])
    logits = disc_to_logits(dg, num_materials=3)
    assert logits.shape == (4, 3)
    # large positive at chosen indices
    for i, c in enumerate(dg):
        assert logits[i, c] > 1e4
        assert (logits[i, torch.arange(3) != c] < 0).all()

    merged = merge_color(dg, 1, 0)
    assert torch.equal(merged, torch.tensor([0, 0, 2, 0]))


def test_find_color_bands_and_merge_bands():
    dg = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    bands = find_color_bands(dg)
    assert bands == [(0, 1, 0), (2, 4, 1), (5, 6, 2)]
    # merge forward
    dg2 = merge_bands(dg, bands[0], bands[1], direction="forward")
    assert torch.equal(dg2, torch.tensor([0, 0, 0, 0, 0, 2, 2]))
    # merge backward
    dg3 = merge_bands(dg, bands[0], bands[1], direction="backward")
    assert torch.equal(dg3, torch.tensor([1, 1, 1, 1, 1, 2, 2]))


def test_remove_outlier_pixels_and_smooth():
    h = torch.zeros(5, 5)
    h[2, 2] = 10.0
    cleaned = remove_outlier_pixels(h, threshold=1.0)
    assert cleaned[2, 2] != h[2, 2]

    # smoothing shouldn't explode and keeps shape
    sm = smooth_coplanar_faces(cleaned, angle_threshold=45.0)
    assert sm.shape == h.shape
