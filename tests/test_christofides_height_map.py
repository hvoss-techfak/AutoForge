import numpy as np

from autoforge.Helper.Heightmaps.ChristofidesHeightMap import (
    two_stage_weighted_kmeans,
    build_distance_matrix,
    matrix_to_graph,
    christofides_tsp,
    compute_ordering_metric,
    create_mapping,
    tsp_order_christofides_path,
)
from skimage.color import rgb2lab


def test_two_stage_weighted_kmeans_small():
    rng = np.random.default_rng(0)
    img = (
        rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8).astype(np.float32)
        / 255.0
    )
    lab = rgb2lab(img)
    labs, labels = two_stage_weighted_kmeans(
        lab.reshape(-1, 3), 16, 16, overcluster_k=20, final_k=4, random_state=0
    )
    assert labs.shape == (4, 3)
    assert labels.shape == (16, 16)
    assert set(np.unique(labels)).issubset(set(range(4)))


def test_christofides_cycle_basic():
    # Simple square in 2D lab space
    labs = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    nodes = [0, 1, 2, 3]
    D = build_distance_matrix(labs, nodes)
    G = matrix_to_graph(D, nodes)
    cycle = christofides_tsp(G)
    assert cycle[0] == cycle[-1]
    assert set(cycle[:-1]) == set(nodes)


def test_ordering_metric_monotonic():
    labs = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    ord1 = [0, 1, 2]
    ord2 = [0, 2, 1]
    m1 = compute_ordering_metric(ord1, labs)
    m2 = compute_ordering_metric(ord2, labs)
    assert m1 < m2  # straight path shorter than zigzag


def test_create_mapping_even_spacing():
    labs = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    ordering = [0, 1, 2]
    all_labels = [0, 1, 2]
    mapping = create_mapping(ordering, labs, all_labels)
    assert mapping[0] == 0.0 and mapping[2] == 1.0


def test_tsp_order_christofides_path_includes_bg_fg():
    labs = np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0]], dtype=float)
    nodes = [0, 1, 2]
    ordering = tsp_order_christofides_path(nodes, labs, bg=0, fg=1)
    assert ordering[0] == 0 and ordering[-1] == 1
    assert set(ordering) == set(nodes)
