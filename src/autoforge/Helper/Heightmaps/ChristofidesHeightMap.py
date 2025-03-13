import concurrent.futures
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab
from tqdm import tqdm


def build_distance_matrix(labs, nodes):
    """
    Given an array labs (with shape (N, dims)) and a list of node indices,
    return a distance matrix (NumPy array) of shape (len(nodes), len(nodes)).
    """
    pts = labs[nodes]  # extract only the points corresponding to nodes
    # Use cdist for fast vectorized distance computation.
    return cdist(pts, pts, metric="euclidean")


def matrix_to_graph(matrix, nodes):
    """
    Convert a 2D NumPy array (matrix) into a dictionary-of-dicts graph,
    where graph[u][v] = matrix[i][j] for u = nodes[i], v = nodes[j].
    """
    graph = {}
    n = len(nodes)
    for i in range(n):
        u = nodes[i]
        graph[u] = {}
        for j in range(n):
            v = nodes[j]
            if u != v:
                graph[u][v] = matrix[i, j]
    return graph


# --- Christofides Helpers (same as before) ---


class UnionFind:
    def __init__(self):
        self.parents = {}
        self.weights = {}

    def __getitem__(self, obj):
        if obj not in self.parents:
            self.parents[obj] = obj
            self.weights[obj] = 1
            return obj
        path = [obj]
        root = self.parents[obj]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max(((self.weights[r], r) for r in roots))[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def minimum_spanning_tree(G):
    tree = []
    subtrees = UnionFind()
    # Build list of edges from graph dictionary.
    edges = sorted((G[u][v], u, v) for u in G for v in G[u])
    for W, u, v in edges:
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, W))
            subtrees.union(u, v)
    return tree


def find_odd_vertexes(MST):
    degree = {}
    for u, v, _ in MST:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    return [v for v in degree if degree[v] % 2 == 1]


def minimum_weight_matching(MST, G, odd_vert):
    odd_vertices = odd_vert.copy()
    random.shuffle(odd_vertices)
    while odd_vertices:
        v = odd_vertices.pop()
        best_u = None
        best_dist = float("inf")
        for u in odd_vertices:
            if G[v][u] < best_dist:
                best_dist = G[v][u]
                best_u = u
        MST.append((v, best_u, G[v][best_u]))
        odd_vertices.remove(best_u)


def find_eulerian_tour(MST, G):
    graph = {}
    for u, v, _ in MST:
        graph.setdefault(u, []).append(v)
        graph.setdefault(v, []).append(u)
    start = next(iter(graph))
    tour = []
    stack = [start]
    while stack:
        v = stack[-1]
        if graph[v]:
            w = graph[v].pop()
            graph[w].remove(v)
            stack.append(w)
        else:
            tour.append(stack.pop())
    return tour


def christofides_tsp(graph):
    MST = minimum_spanning_tree(graph)
    odd_vertices = find_odd_vertexes(MST)
    minimum_weight_matching(MST, graph, odd_vertices)
    eulerian_tour = find_eulerian_tour(MST, graph)
    seen = set()
    path = []
    for v in eulerian_tour:
        if v not in seen:
            seen.add(v)
            path.append(v)
    path.append(path[0])
    return path


def prune_ordering(ordering, labs, bg, fg, min_length=3, improvement_factor=1.5):
    """
    Iteratively remove clusters from the ordering if doing so significantly reduces
    the total Lab-space distance. Only clusters that produce an improvement greater
    than improvement_factor * (median gap) are removed.

    Parameters:
      ordering: list of cluster indices (the current ordering)
      labs: Lab-space coordinates (indexed by cluster index)
      bg: background anchor (never removed)
      fg: foreground anchor (never removed)
      min_length: minimum allowed length of ordering
      improvement_factor: factor multiplied by the median gap to decide if a cluster is an outlier

    Returns:
      A pruned ordering that hopefully removes only extreme outliers.
    """
    current_order = ordering.copy()
    improved = True
    # print(f"Height map pruning pass: initial number of clusters = {len(ordering)}")
    while improved:
        improved = False
        # Compute gaps between consecutive clusters.
        diffs = [
            np.linalg.norm(labs[current_order[i]] - labs[current_order[i - 1]])
            for i in range(1, len(current_order))
        ]
        if len(diffs) == 0:
            break
        median_diff = np.median(diffs)
        # Try each candidate (skip first and last if they are bg and fg)
        for i in range(1, len(current_order) - 1):
            # Optionally, preserve the fg anchor.
            if current_order[i] == fg:
                continue
            # Compute the improvement if we remove current_order[i]:
            d1 = np.linalg.norm(labs[current_order[i]] - labs[current_order[i - 1]])
            d2 = np.linalg.norm(labs[current_order[i + 1]] - labs[current_order[i]])
            direct = np.linalg.norm(
                labs[current_order[i + 1]] - labs[current_order[i - 1]]
            )
            improvement = (d1 + d2) - direct
            # Remove the cluster if the improvement is large compared to median gap.
            if improvement > improvement_factor * median_diff:
                new_order = current_order[:i] + current_order[i + 1 :]
                if len(new_order) >= min_length:
                    # print(
                    #     f"Pruning outlier: removing cluster {current_order[i]} improved local gap by {improvement:.2f}"
                    # )
                    current_order = new_order
                    improved = True
                    break  # restart the scan after removal
    # print(f"Height map pruning pass: final number of clusters = {len(current_order)}")
    return current_order


def create_mapping(final_ordering, labs, all_labels):
    """
    Creates a mapping from each cluster (from all_labels) to a value in [0,1].
    Clusters in final_ordering get evenly spaced values.
    For clusters that were pruned (i.e. not in final_ordering), assign the value
    of the nearest cluster in final_ordering (based on Lab-space distance).

    Parameters:
      final_ordering: list of cluster indices (after pruning)
      labs: array of Lab-space coordinates (indexed by cluster index)
      all_labels: sorted list of all unique clusters produced by KMeans

    Returns:
      mapping: a dict mapping each cluster label in all_labels to a float in [0,1].
    """
    mapping = {}
    n_order = len(final_ordering)
    # If there's only one cluster in final_ordering, assign 0.5
    if n_order == 1:
        for label in all_labels:
            mapping[label] = 0.5
        return mapping

    # Assign evenly spaced values for clusters in final_ordering.
    for i, cluster in enumerate(final_ordering):
        mapping[cluster] = i / (n_order - 1)

    # For clusters not in final_ordering, find the nearest cluster (in Lab space)
    # from final_ordering and use its mapping value.
    for label in all_labels:
        if label not in mapping:
            lab_val = labs[label]
            best_cluster = None
            best_dist = float("inf")
            for cl in final_ordering:
                d = np.linalg.norm(labs[cl] - lab_val)
                if d < best_dist:
                    best_dist = d
                    best_cluster = cl
            mapping[label] = mapping[best_cluster]
    return mapping


def tsp_order_christofides_path(nodes, labs, bg, fg):
    """
    nodes: list of cluster indices (must include bg and fg)
    labs: Lab-space coordinates (indexed by cluster index)
    bg, fg: background and foreground cluster indices

    Returns an ordering (list of cluster indices) that contains all nodes,
    starts with bg and ends with fg.
    """
    artificial = -1
    augmented_nodes = nodes + [artificial]
    LARGE = 1e6

    # Precompute the distance matrix for original nodes.
    D = build_distance_matrix(labs, nodes)
    n = len(nodes)

    # Build an augmented matrix of size (n+1)x(n+1).
    aug_mat = np.zeros((n + 1, n + 1))
    aug_mat[:n, :n] = D
    # For the artificial node, set cost = 0 for bg/fg, LARGE otherwise.
    for i, u in enumerate(nodes):
        if u in {bg, fg}:
            aug_mat[i, n] = 0.0
            aug_mat[n, i] = 0.0
        else:
            aug_mat[i, n] = LARGE
            aug_mat[n, i] = LARGE
    aug_mat[n, n] = 0.0

    # Convert the augmented matrix into a graph dictionary.
    aug_nodes = nodes + [artificial]
    graph = matrix_to_graph(aug_mat, aug_nodes)

    # Run Christofides on the augmented graph.
    cycle = christofides_tsp(graph)
    # Remove the artificial node.
    cycle = [node for node in cycle if node != artificial]

    # Rotate the cycle so that bg is first.
    if bg in cycle:
        idx = cycle.index(bg)
        cycle = cycle[idx:] + cycle[:idx]
    else:
        raise ValueError("Background node missing from cycle")

    # Force fg to be the last element.
    if fg not in cycle:
        raise ValueError("Foreground node missing from cycle")
    if cycle[-1] != fg:
        cycle.remove(fg)
        cycle.append(fg)

    # --- NEW: Check if reversing the internal order yields a lower ordering metric ---
    if len(cycle) > 2:
        # Reverse the ordering of the nodes between bg and fg.
        reversed_cycle = [cycle[0]] + cycle[1:-1][::-1] + [cycle[-1]]
        if compute_ordering_metric(reversed_cycle, labs) < compute_ordering_metric(
            cycle, labs
        ):
            cycle = reversed_cycle

    return cycle


# --- Optimized Ordering Metric (Vectorized) ---


def compute_ordering_metric(ordering, labs):
    """
    Computes a metric for the ordering as the total Lab-space distance between consecutive clusters.
    Uses vectorized operations for speed.
    """
    pts = labs[ordering]
    # Compute differences between consecutive rows.
    diffs = np.diff(pts, axis=0)
    # Compute Euclidean norms along rows and sum.
    return np.sum(np.linalg.norm(diffs, axis=1))


# --- Revised init_height_map with Optimizations ---


def init_height_map(
    target,
    max_layers,
    h,  # unused here but preserved for API compatibility
    background_tuple,
    eps=1e-6,
    random_seed=None,
    lab_weights=(2.0, 1.0, 1.0),
):
    """
    Initializes pixel height logits by:
      1. Clustering the image into max_layers clusters (via KMeans).
      2. Converting cluster centers to Lab space.
      3. Determining two anchor clusters:
         - The background cluster (closest to background_tuple) as the bottom.
         - The foreground cluster (farthest from the background) as the top.
      4. Using a Christofides TSP solution (with an artificial node) to order the clusters.
      5. Mapping the clusters to evenly spaced height values in [0, 1] and converting to logits.
    """
    import random

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    H, W, _ = target.shape

    # Convert target (assumed in [0,255]) to a PIL Image.
    pil_im = Image.fromarray(target.astype(np.uint8))
    # Quantize image into max_layers colors.
    quantized_im = pil_im.quantize(colors=max_layers)
    # Retrieve per-pixel labels (indices into the palette).
    labels = np.array(quantized_im)

    # Get the full palette and reshape it into (-1, 3)
    full_palette = quantized_im.getpalette()
    palette_arr = np.array(full_palette, dtype=np.uint8).reshape(-1, 3)

    # Get only the palette indices actually used in the quantized image.
    unique_palette_indices = sorted(np.unique(labels))
    # Create the labs array from the used palette colors.
    labs_rgb = palette_arr[unique_palette_indices].astype(np.float32)

    # Convert RGB values (0-255) to Lab space.
    labs = rgb2lab(labs_rgb / 255.0)
    # Apply the weighting to each Lab channel.
    labs[:, 0] *= lab_weights[0]
    labs[:, 1] *= lab_weights[1]
    labs[:, 2] *= lab_weights[2]

    # Remap the labels in the image to a compact index range.
    palette_map = {old: new for new, old in enumerate(unique_palette_indices)}
    labels = np.vectorize(lambda x: palette_map[x])(labels)

    # Convert the background color to Lab (with same weighting).
    bg_rgb = np.array(background_tuple).astype(np.float32) / 255.0
    bg_lab = rgb2lab(np.array([[bg_rgb]]))[0, 0, :]
    bg_lab[0] *= lab_weights[0]
    bg_lab[1] *= lab_weights[1]
    bg_lab[2] *= lab_weights[2]

    # Identify the best background and foreground clusters.
    distances = np.linalg.norm(labs - bg_lab, axis=1)
    bg_cluster = int(np.argmin(distances))
    fg_cluster = int(np.argmax(distances))

    # Use all the new compact cluster indices.
    unique_clusters = list(range(len(unique_palette_indices)))
    nodes = unique_clusters

    # Get the ordering via TSP ordering function.
    final_ordering = tsp_order_christofides_path(nodes, labs, bg_cluster, fg_cluster)
    # Optionally prune out outliers.
    final_ordering = prune_ordering(
        final_ordering, labs, bg_cluster, fg_cluster, min_length=3, improvement_factor=3
    )

    # Create a mapping that covers all clusters.
    new_values = create_mapping(final_ordering, labs, unique_clusters)
    new_labels = np.vectorize(lambda x: new_values[x])(labels).astype(np.float32)
    pixel_height_logits = np.log((new_labels + eps) / (1 - new_labels + eps))

    ordering_metric = compute_ordering_metric(final_ordering, labs)
    return pixel_height_logits, ordering_metric


def run_init_threads(
    target,
    max_layers,
    h,  # unused here but preserved for API compatibility
    background_tuple,
    eps=1e-6,
    random_seed=None,
    num_threads=64,
):
    """
    Initializes pixel height logits using multiple threads.
    :param target:
    :param max_layers:
    :param h:
    :param background_tuple:
    :param eps:
    :param random_seed:
    :param num_threads:
    :return:
    """
    if random_seed is None:
        random_seed = np.random.randint(1e6)
    exec = ThreadPoolExecutor(max_workers=num_threads)
    futures = []
    for i in range(num_threads):
        futures.append(
            exec.submit(
                init_height_map,
                target,
                max_layers,
                h,
                background_tuple,
                eps,
                random_seed + i,
            )
        )
    results = [
        f.result()
        for f in tqdm(
            concurrent.futures.as_completed(futures),
            desc="Running Height Map Initialization Threads",
            total=num_threads,
        )
    ]
    # print(f"metric values: {[r[1] for r in results]}")
    # get lowest ordering metric
    best_result = min(results, key=lambda x: x[1])

    return best_result[0]
