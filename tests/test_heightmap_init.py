import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("skimage")

from autoforge.Helper.Heightmaps.ChristofidesHeightMap import (
    two_stage_weighted_kmeans,
    compute_ordering_metric,
    tsp_order_christofides_path,
    create_mapping,
    interpolate_arrays,
)


def test_two_stage_weighted_kmeans_small():
    # simple two-color image in Lab-like space
    H, W = 20, 20
    lab = np.zeros((H * W, 3), dtype=np.float32)
    lab[: (H * W) // 2] = np.array([50, 0, 0])
    lab[(H * W) // 2 :] = np.array([80, 20, -10])
    centroids, labels = two_stage_weighted_kmeans(
        lab, H, W, overcluster_k=10, final_k=2, random_state=42
    )
    assert centroids.shape == (2, 3)
    assert set(np.unique(labels)) <= {0, 1}


def test_ordering_metric_and_tsp():
    labs = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    nodes = [0, 1, 2, 3]
    order = tsp_order_christofides_path(nodes, labs, bg=0, fg=3)
    # should start at bg and end at fg
    assert order[0] == 0 and order[-1] == 3
    metric = compute_ordering_metric(order, labs)
    assert metric >= 3.0


def test_create_mapping_and_interpolate_arrays():
    values = [0.0, 0.5, 1.0]
    arrs = [
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0.5, 0.5, 0.1], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.float32),
    ]
    pairs = list(zip(values, arrs))
    out = interpolate_arrays(pairs, num_points=5)
    assert out.shape == (5, 3)
    # mapping
    labs = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    final_order = [0, 1, 2]
    all_labels = [0, 1, 2]
    mapping = create_mapping(final_order, labs, all_labels)
    assert mapping[0] == 0.0 and mapping[2] == 1.0
