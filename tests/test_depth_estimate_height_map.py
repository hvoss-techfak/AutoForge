import numpy as np

from autoforge.Helper.Heightmaps.DepthEstimateHeightMap import (
    initialize_pixel_height_logits,
    tsp_simulated_annealing,
    choose_optimal_num_bands,
)


def test_initialize_pixel_height_logits_range(small_image):
    logits = initialize_pixel_height_logits(small_image)
    assert logits.shape == small_image.shape[:2]
    # Inverse sigmoid -> sigmoid(logits) should approx luminance
    lum = (
        0.299 * small_image[..., 0]
        + 0.587 * small_image[..., 1]
        + 0.114 * small_image[..., 2]
    ) / 255.0
    recon = 1 / (1 + np.exp(-logits))
    assert np.mean(np.abs(recon - lum)) < 1e-5


def test_tsp_simulated_annealing_basic():
    band_reps = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    order = tsp_simulated_annealing(band_reps, 0, 2, num_iter=200, initial_temp=10)
    assert order[0] == 0 and order[-1] == 2
    assert set(order) == {0, 1, 2, 3}


def test_choose_optimal_num_bands():
    centroids = np.concatenate(
        [
            np.random.randn(10, 3) * 0.01 + 0.0,
            np.random.randn(10, 3) * 0.01 + 1.0,
            np.random.randn(10, 3) * 0.01 + 2.0,
        ],
        axis=0,
    )
    k = choose_optimal_num_bands(centroids, min_bands=2, max_bands=4, random_seed=0)
    assert k in (2, 3, 4)
