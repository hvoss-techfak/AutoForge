import numpy as np

from autoforge.Helper.Heightmaps.ChristofidesHeightMap import (
    _compute_distinctiveness,
    build_distance_matrix,
    sample_pixels_for_silhouette,
    segmentation_quality,
    matrix_to_graph,
    minimum_spanning_tree,
    find_odd_vertexes,
    minimum_weight_matching,
    find_eulerian_tour,
    christofides_tsp,
    prune_ordering,
    create_mapping,
    tsp_order_christofides_path,
    compute_ordering_metric,
)


def test_compute_distinctiveness_and_distance_matrix():
    centroids = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    d = _compute_distinctiveness(centroids)
    assert d.shape == (3,)

    nodes = [0, 1, 2]
    D = build_distance_matrix(centroids, nodes)
    assert D.shape == (3, 3)
    assert np.isclose(D[0, 1], 1.0)


def test_sample_and_segmentation_quality():
    labels = np.tile(np.array([[0, 1], [1, 0]], dtype=int), (5, 5))
    idx, subset = sample_pixels_for_silhouette(labels, sample_size=10, random_state=0)
    assert len(idx) == len(subset)

    X = np.random.rand(labels.size, 3).astype(float)
    score = segmentation_quality(X, labels, sample_size=50, random_state=0)
    assert isinstance(score, float)


def test_graph_and_christofides_components():
    labs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    nodes = [0, 1, 2, 3]
    D = build_distance_matrix(labs, nodes)
    G = matrix_to_graph(D, nodes)
    MST = minimum_spanning_tree(G)
    odd = find_odd_vertexes(MST)
    minimum_weight_matching(MST, G, odd)
    tour = find_eulerian_tour(MST, G)
    path = christofides_tsp(G)
    assert len(path) >= 4


def test_prune_ordering_and_mapping_and_tsp_path():
    labs = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    ordering = [0, 1, 2, 3]
    pruned = prune_ordering(
        ordering, labs, bg=0, fg=3, min_length=3, improvement_factor=0.1
    )
    assert len(pruned) >= 3

    mapping = create_mapping(pruned, labs, all_labels=[0, 1, 2, 3])
    assert set(mapping.keys()) == {0, 1, 2, 3}

    cycle = tsp_order_christofides_path([0, 1, 2, 3], labs, bg=0, fg=3)
    assert cycle[0] == 0 and cycle[-1] == 3

    metric = compute_ordering_metric([0, 2, 3], labs)
    assert metric >= 0.0
