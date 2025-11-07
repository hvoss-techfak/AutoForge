import numpy as np
import pytest
import sys
import types

from autoforge.Helper.Heightmaps.DepthEstimateHeightMap import (
    init_height_map_depth_color_adjusted,
    choose_optimal_num_bands,
)


class DummyDepthPipe:
    def __init__(self, depth_map: np.ndarray):
        self._depth_map = depth_map

    def __call__(self, image):  # image is a PIL Image
        # Return a dict mimicking transformers depth-estimation pipeline output
        return {"depth": self._depth_map}


@pytest.fixture
def two_color_image():
    # 4x4 image: left half red, right half blue
    left = np.tile(np.array([255, 0, 0], dtype=np.uint8), (4, 2, 1))
    right = np.tile(np.array([0, 0, 255], dtype=np.uint8), (4, 2, 1))
    img = np.concatenate([left, right], axis=1)
    return img.astype(np.float32)


def _install_fake_transformers(monkeypatch, depth_map: np.ndarray):
    fake_mod = types.ModuleType("transformers")

    def dummy_pipeline(*, task, model):
        assert task == "depth-estimation"
        return DummyDepthPipe(depth_map)

    fake_mod.pipeline = dummy_pipeline
    # ensure subattributes potentially accessed exist (not used here but safe)
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)


def test_init_height_map_no_split(monkeypatch, two_color_image):
    # Depth map with small variation so depth_range < threshold => no splits
    depth_map = np.concatenate(
        [
            np.full((4, 2), 0.2, dtype=np.float32),
            np.full((4, 2), 0.8, dtype=np.float32),
        ],
        axis=1,
    )

    _install_fake_transformers(monkeypatch, depth_map)

    logits, labels = init_height_map_depth_color_adjusted(
        two_color_image,
        max_layers=4,
        depth_threshold=0.05,  # range inside cluster is 0, so no split
        random_seed=0,
        depth_strength=0.0,  # force even spacing only
    )

    assert logits.shape == labels.shape == two_color_image.shape[:2]
    assert np.isfinite(logits).all()
    # Reconstruct normalized values via sigmoid
    recon = 1 / (1 + np.exp(-logits))
    unique_vals = sorted(np.unique(np.round(recon, 2)))
    # Expect two clusters => two even-spaced values from 0.1 to 1.0
    expected = [0.1, 1.0]
    assert len(unique_vals) == 2
    assert np.allclose(unique_vals, expected, atol=0.05)


def test_init_height_map_with_split(monkeypatch):
    # Image with two colors arranged so each initial cluster spans gradient depths triggering split
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[:, :2] = np.array([255, 0, 0], dtype=np.float32)  # red
    img[:, 2:] = np.array([0, 255, 0], dtype=np.float32)  # green

    # Depth gradient across rows ensures depth_range within each color region is large.
    row_grad = np.linspace(0.0, 1.0, 4, dtype=np.float32).reshape(4, 1)
    depth_map = np.concatenate(
        [
            np.repeat(row_grad, 2, axis=1),  # red region gradient
            np.repeat(1 - row_grad, 2, axis=1),  # green region inverse gradient
        ],
        axis=1,
    )

    _install_fake_transformers(monkeypatch, depth_map)

    logits, labels = init_height_map_depth_color_adjusted(
        img,
        max_layers=4,
        depth_threshold=0.2,  # low threshold so each initial cluster splits (range ~1)
        random_seed=0,
        depth_strength=0.25,
        order_blend=0.5,
    )

    assert logits.shape == (4, 4)
    assert labels.shape == (4, 4)
    assert np.isfinite(logits).all()
    # Expect 4 final clusters due to splitting both initial clusters
    assert len(np.unique(labels)) == 4
    # Check that reconstructed values cover >2 distinct spacings
    recon = 1 / (1 + np.exp(-logits))
    assert len(np.unique(np.round(recon, 2))) >= 3


def test_init_height_map_depth_strength_one(monkeypatch, two_color_image):
    # Depth map extreme values cause clipping of lower cluster to min_cluster_value when depth_strength=1
    depth_map = np.concatenate(
        [
            np.full((4, 2), 0.0, dtype=np.float32),
            np.full((4, 2), 1.0, dtype=np.float32),
        ],
        axis=1,
    )

    _install_fake_transformers(monkeypatch, depth_map)

    logits, labels = init_height_map_depth_color_adjusted(
        two_color_image,
        max_layers=4,
        depth_threshold=0.05,
        random_seed=0,
        depth_strength=1.0,  # pure depth
    )

    recon = 1 / (1 + np.exp(-logits))
    unique_vals = sorted(np.unique(np.round(recon, 2)))
    # Expect clipping of low-depth cluster to 0.1 and high cluster to 1.0
    assert unique_vals[0] == pytest.approx(0.1, abs=0.02)
    assert unique_vals[-1] == pytest.approx(1.0, abs=0.02)


def test_choose_optimal_num_bands_all_identical():
    # Centroids identical => silhouette score skipped; should return min_bands
    centroids = np.zeros((10, 3), dtype=np.float32)
    k = choose_optimal_num_bands(centroids, min_bands=2, max_bands=4, random_seed=0)
    assert k == 2
