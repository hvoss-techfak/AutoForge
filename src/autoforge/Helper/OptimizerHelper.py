import random
from itertools import permutations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm
from transformers import pipeline
from PIL import Image
from sklearn.metrics import silhouette_score


@torch.jit.script
def adaptive_round(
    x: torch.Tensor, tau: float, high_tau: float, low_tau: float, temp: float
) -> torch.Tensor:
    """
    Smooth rounding based on temperature 'tau'.
    """
    if tau <= low_tau:
        return torch.round(x)
    elif tau >= high_tau:
        floor_val = torch.floor(x)
        diff = x - floor_val
        soft_round = floor_val + torch.sigmoid((diff - 0.5) / temp)
        return soft_round
    else:
        ratio = (tau - low_tau) / (high_tau - low_tau)
        hard_round = torch.round(x)
        floor_val = torch.floor(x)
        diff = x - floor_val
        soft_round = floor_val + torch.sigmoid((diff - 0.5) / temp)
        return ratio * soft_round + (1 - ratio) * hard_round


# A deterministic random generator that mimics torch.rand_like.
@torch.jit.script
def deterministic_rand_like(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    # Compute the total number of elements.
    n: int = 1
    for d in tensor.shape:
        n = n * d
    # Create a 1D tensor of indices [0, 1, 2, ..., n-1].
    indices = torch.arange(n, dtype=torch.float32, device=tensor.device)
    # Offset the indices by the seed.
    indices = indices + seed
    # Use a simple hash function: sin(x)*constant, then take the fractional part.
    r = torch.sin(indices) * 43758.5453123
    r = r - torch.floor(r)
    # Reshape to the shape of the original tensor.
    return r.view(tensor.shape)


@torch.jit.script
def deterministic_gumbel_softmax(
    logits: torch.Tensor, tau: float, hard: bool, rng_seed: int
) -> torch.Tensor:
    eps: float = 1e-20
    # Instead of torch.rand_like(..., generator=...), use our deterministic_rand_like.
    U = deterministic_rand_like(logits, rng_seed)
    # Compute Gumbel noise.
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    y = (logits + gumbel_noise) / tau
    y_soft = F.softmax(y, dim=-1)
    if hard:
        # Compute one-hot using argmax and scatter.
        index = torch.argmax(y_soft, dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        # Use the straight-through estimator.
        y = (y_hard - y_soft).detach() + y_soft
    return y


@torch.jit.script
def composite_image(
    pixel_height_logits: torch.Tensor,
    global_logits: torch.Tensor,
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,
    material_TDs: torch.Tensor,
    background: torch.Tensor,
    mode: str = "continuous",
    rng_seed: int = -1,  # <--- new optional argument
) -> torch.Tensor:
    """
    Vectorized compositing over all pixels (H x W).
    ...
    If rng_seed >= 0 and mode=="discrete", we will fix the random seed
    for each layer so that we get the same random Gumbel noise for reproducibility.
    """
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logits)
    continuous_layers = pixel_height / h

    # same as before
    adaptive_layers = adaptive_round(
        continuous_layers, tau_height, high_tau=0.1, low_tau=0.01, temp=0.1
    )
    discrete_layers_temp = torch.round(continuous_layers)
    discrete_layers = (
        discrete_layers_temp + (adaptive_layers - discrete_layers_temp).detach()
    )
    discrete_layers = discrete_layers.to(torch.int32)

    H, W = pixel_height.shape
    comp = torch.zeros(H, W, 3, dtype=torch.float32, device=pixel_height.device)
    remaining = torch.ones(H, W, dtype=torch.float32, device=pixel_height.device)

    o = -1.2416557e-02
    A = 9.6407950e-01
    k = 3.4103447e01
    b = -4.1554203e00

    for i in range(max_layers):
        layer_idx = max_layers - 1 - i
        p_print = (discrete_layers > layer_idx).float()
        eff_thick = p_print * h

        # This is the key difference for "discrete":
        # we now optionally fix the random seed if rng_seed >= 0
        if mode == "discrete":
            if rng_seed >= 0:
                # Use a context manager so we don’t pollute the global seed
                p_i = deterministic_gumbel_softmax(
                    global_logits[layer_idx],
                    tau_global,
                    hard=True,
                    rng_seed=rng_seed + layer_idx,
                )
            else:
                # same as before, but will be random every call
                p_i = F.gumbel_softmax(global_logits[layer_idx], tau_global, hard=True)

        elif mode == "continuous":
            if tau_global < 1e-3:
                p_i = F.gumbel_softmax(global_logits[layer_idx], tau_global, hard=True)
            else:
                p_i = F.gumbel_softmax(global_logits[layer_idx], tau_global, hard=False)
        else:
            # e.g. "pruning" or any other fallback
            p_i = F.gumbel_softmax(global_logits[layer_idx], tau_global, hard=True)

        color_i = torch.matmul(p_i, material_colors)
        TD_i = torch.matmul(p_i, material_TDs)
        TD_i = torch.clamp(TD_i, 1e-8, 1e8)

        opac = o + (A * torch.log1p(k * (eff_thick / TD_i)) + b * (eff_thick / TD_i))
        # opac = torch.where(TD_i / 10 <= h, torch.tensor(1.0), opac)
        opac = torch.clamp(opac, 0.0, 1.0)

        comp = comp + ((remaining * opac).unsqueeze(-1) * color_i)
        remaining = remaining * (1 - opac)

    comp = comp + remaining.unsqueeze(-1) * background
    return comp * 255.0


@torch.jit.script
def adaptive_round(
    x: torch.Tensor, tau: float, high_tau: float, low_tau: float, temp: float
) -> torch.Tensor:
    """
    Smooth rounding based on temperature 'tau'.
    """
    if tau <= low_tau:
        return torch.round(x)
    elif tau >= high_tau:
        floor_val = torch.floor(x)
        diff = x - floor_val
        soft_round = floor_val + torch.sigmoid((diff - 0.5) / temp)
        return soft_round
    else:
        ratio = (tau - low_tau) / (high_tau - low_tau)
        hard_round = torch.round(x)
        floor_val = torch.floor(x)
        diff = x - floor_val
        soft_round = floor_val + torch.sigmoid((diff - 0.5) / temp)
        return ratio * soft_round + (1 - ratio) * hard_round


def discretize_solution(
    params: dict, tau_global: float, h: float, max_layers: int, rng_seed: int = -1
):
    """
    Convert continuous logs to discrete layer counts and discrete color IDs.
    """
    pixel_height_logits = params["pixel_height_logits"]
    global_logits = params["global_logits"]
    pixel_heights = (max_layers * h) * torch.sigmoid(pixel_height_logits)
    discrete_height_image = torch.round(pixel_heights / h).to(torch.int32)
    discrete_height_image = torch.clamp(discrete_height_image, 0, max_layers)

    num_layers = global_logits.shape[0]
    discrete_global_vals = []
    for j in range(num_layers):
        p = deterministic_gumbel_softmax(
            global_logits[j], tau_global, hard=True, rng_seed=rng_seed + j
        )
        discrete_global_vals.append(torch.argmax(p))
    discrete_global = torch.stack(discrete_global_vals, dim=0)
    return discrete_global, discrete_height_image


def initialize_pixel_height_logits(target):
    """
    Initialize pixel height logits based on the luminance of the target image.

    Assumes target is a jnp.array of shape (H, W, 3) in the range [0, 255].
    Uses the formula: L = 0.299*R + 0.587*G + 0.114*B.

    Args:
        target (jnp.ndarray): The target image array with shape (H, W, 3).

    Returns:
        jnp.ndarray: The initialized pixel height logits.
    """
    # Compute normalized luminance in [0,1]
    normalized_lum = (
        0.299 * target[..., 0] + 0.587 * target[..., 1] + 0.114 * target[..., 2]
    ) / 255.0
    # To avoid log(0) issues, add a small epsilon.
    eps = 1e-6
    # Convert normalized luminance to logits using the inverse sigmoid (logit) function.
    # This ensures that jax.nn.sigmoid(pixel_height_logits) approximates normalized_lum.
    pixel_height_logits = np.log((normalized_lum + eps) / (1 - normalized_lum + eps))
    return pixel_height_logits


@ignore_warnings(category=ConvergenceWarning)
def init_height_map_depth_color_adjusted(
    target,
    max_layers,
    eps=1e-6,
    random_seed=None,
    depth_strength=0.25,
    depth_threshold=0.2,
    min_cluster_value=0.1,
    w_depth=0.5,
    w_lum=0.5,
    order_blend=0.1,
):
    """
    Initialize pixel height logits by combining depth and color information while allowing a blend
    between the original luminance-based ordering and a depth-informed ordering.

    Steps:
      1. Obtain a normalized depth map using Depth Anything v2.
      2. Determine the optimal number of color clusters (between 2 and max_layers) via silhouette score.
      3. Cluster the image colors and (if needed) split clusters with large depth spreads.
      4. For each final cluster, compute its average depth and average luminance.
      5. Compute two orderings:
            - ordering_orig: Sorted purely by average luminance (approximating the original code).
            - ordering_depth: A TSP-inspired ordering using a weighted distance based on depth and luminance.
      6. For each cluster, blend its rank (normalized position) between the two orderings using order_blend.
      7. Based on the blended ordering, assign an even spacing value from min_cluster_value to 1.
      8. Finally, blend the even spacing with the cluster's average depth using depth_strength and
         convert the result to logits via an inverse sigmoid transform.

    Args:
        target (np.ndarray): Input image of shape (H, W, 3) in [0, 255].
        max_layers (int): Maximum number of clusters to consider.
        eps (float): Small constant to avoid division by zero.
        random_seed (int): Random seed for reproducibility.
        depth_strength (float): Weight (0 to 1) for blending even spacing with the cluster's average depth.
        depth_threshold (float): If a cluster’s depth spread exceeds this value, it is split.
        min_cluster_value (float): Minimum normalized value for the lowest cluster.
        w_depth (float): Weight for depth difference in ordering_depth.
        w_lum (float): Weight for luminance difference in ordering_depth.
        order_blend (float): Slider (0 to 1) blending original luminance ordering (0) and depth-informed ordering (1).

    Returns:
        np.ndarray: Pixel height logits (H, W).
    """

    # ---------------------------
    # Step 1: Obtain normalized depth map using Depth Anything v2
    # ---------------------------
    target_uint8 = target.astype(np.uint8)
    image_pil = Image.fromarray(target_uint8)
    pipe = pipeline(
        task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf"
    )
    depth_result = pipe(image_pil)
    depth_map = depth_result["depth"]
    if hasattr(depth_map, "convert"):
        depth_map = np.array(depth_map)
    depth_map = depth_map.astype(np.float32)
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + eps)

    # ---------------------------
    # Step 2: Find optimal number of clusters (n in [2, max_layers]) for color clustering
    # ---------------------------
    H, W, _ = target.shape
    pixels = target.reshape(-1, 3).astype(np.float32)

    def find_best_n_clusters(pixels, max_layers, random_seed):
        sample_size = 1000
        if pixels.shape[0] > sample_size:
            indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
            sample_pixels = pixels[indices]
        else:
            sample_pixels = pixels
        best_n = None
        best_score = -np.inf
        for n in range(2, max_layers + 1):
            kmeans_temp = KMeans(n_clusters=n, random_state=random_seed)
            labels_temp = kmeans_temp.fit_predict(sample_pixels)
            score = silhouette_score(sample_pixels, labels_temp)
            if score > best_score:
                best_score = score
                best_n = n
        return best_n

    optimal_n = find_best_n_clusters(pixels, max_layers, random_seed)
    print(f"Optimal number of clusters: {optimal_n}")
    # ---------------------------
    # Step 3: Perform color clustering on the full image
    # ---------------------------
    kmeans = KMeans(n_clusters=optimal_n, random_state=random_seed).fit(pixels)
    labels = kmeans.labels_.reshape(H, W)

    # ---------------------------
    # Step 4: Adjust clusters based on depth (split clusters with high depth spread)
    # ---------------------------
    final_labels = np.copy(labels)
    new_cluster_id = 0
    cluster_info = {}  # Mapping: final_cluster_id -> avg_depth
    unique_labels = np.unique(labels)
    for orig_label in unique_labels:
        mask = labels == orig_label
        cluster_depths = depth_norm[mask]
        avg_depth = np.mean(cluster_depths)
        depth_range = cluster_depths.max() - cluster_depths.min()
        if depth_range > depth_threshold:
            # Split this cluster into 2 subclusters based on depth values.
            depth_values = cluster_depths.reshape(-1, 1)
            k_split = 2
            kmeans_split = KMeans(n_clusters=k_split, random_state=random_seed)
            split_labels = kmeans_split.fit_predict(depth_values)
            indices = np.argwhere(mask)
            for split in range(k_split):
                sub_mask = split_labels == split
                inds = indices[sub_mask]
                if inds.size == 0:
                    continue
                for i, j in inds:
                    final_labels[i, j] = new_cluster_id
                sub_avg_depth = np.mean(depth_norm[mask][split_labels == split])
                cluster_info[new_cluster_id] = sub_avg_depth
                new_cluster_id += 1
        else:
            indices = np.argwhere(mask)
            for i, j in indices:
                final_labels[i, j] = new_cluster_id
            cluster_info[new_cluster_id] = avg_depth
            new_cluster_id += 1

    num_final_clusters = new_cluster_id

    # ---------------------------
    # Step 5: Compute average luminance for each final cluster (using standard weights)
    # ---------------------------
    cluster_colors = {}
    for cid in range(num_final_clusters):
        mask = final_labels == cid
        if np.sum(mask) == 0:
            continue
        avg_color = np.mean(
            target.reshape(-1, 3)[final_labels.reshape(-1) == cid], axis=0
        )
        lum = (
            0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2]
        ) / 255.0
        cluster_colors[cid] = lum

    # ---------------------------
    # Step 6: Build cluster feature list: (cid, avg_depth, avg_luminance)
    # ---------------------------
    cluster_features = []
    for cid in range(num_final_clusters):
        avg_depth = cluster_info[cid]
        avg_lum = cluster_colors.get(cid, 0.5)
        cluster_features.append((cid, avg_depth, avg_lum))

    # ---------------------------
    # Step 7: Compute depth-informed ordering (TSP-inspired, using w_depth and w_lum)
    # ---------------------------
    def distance(feat1, feat2):
        return w_depth * abs(feat1[1] - feat2[1]) + w_lum * abs(feat1[2] - feat2[2])

    # Greedy nearest-neighbor ordering starting from cluster with lowest avg_depth
    start_idx = np.argmin([feat[1] for feat in cluster_features])
    unvisited = cluster_features.copy()
    ordering_depth = []
    current = unvisited.pop(start_idx)
    ordering_depth.append(current)
    while unvisited:
        next_idx = np.argmin([distance(current, candidate) for candidate in unvisited])
        current = unvisited.pop(next_idx)
        ordering_depth.append(current)

    # 2-opt refinement for ordering_depth
    def total_distance(ordering):
        return sum(
            distance(ordering[i], ordering[i + 1]) for i in range(len(ordering) - 1)
        )

    improved = True
    best_order_depth = ordering_depth
    best_dist = total_distance(ordering_depth)
    while improved:
        improved = False
        for i in range(1, len(best_order_depth) - 1):
            for j in range(i + 1, len(best_order_depth)):
                new_order = (
                    best_order_depth[:i]
                    + best_order_depth[i : j + 1][::-1]
                    + best_order_depth[j + 1 :]
                )
                new_dist = total_distance(new_order)
                if new_dist < best_dist:
                    best_order_depth = new_order
                    best_dist = new_dist
                    improved = True
        ordering_depth = best_order_depth

    # ---------------------------
    # Step 8: Compute original (luminance-based) ordering: simply sort by avg_lum (darkest first)
    # ---------------------------
    ordering_orig = sorted(cluster_features, key=lambda x: x[2])

    # ---------------------------
    # Step 9: Blend the two orderings via their rank positions using order_blend
    # ---------------------------
    # Map each cluster id to its rank in each ordering.
    rank_orig = {feat[0]: idx for idx, feat in enumerate(ordering_orig)}
    rank_depth = {feat[0]: idx for idx, feat in enumerate(ordering_depth)}
    # Normalize ranks to [0, 1]
    norm_rank_orig = {
        cid: rank_orig[cid] / (len(ordering_orig) - 1) if len(ordering_orig) > 1 else 0
        for cid in rank_orig
    }
    norm_rank_depth = {
        cid: rank_depth[cid] / (len(ordering_depth) - 1)
        if len(ordering_depth) > 1
        else 0
        for cid in rank_depth
    }

    # Compute blended rank for each cluster
    blended_ranks = {}
    for cid in norm_rank_orig:
        blended_ranks[cid] = (1 - order_blend) * norm_rank_orig[
            cid
        ] + order_blend * norm_rank_depth[cid]

    # Final ordering: sort clusters by blended rank (ascending)
    final_order = sorted(cluster_features, key=lambda x: blended_ranks[x[0]])

    # ---------------------------
    # Step 10: Assign new normalized values to clusters
    # Even spacing from min_cluster_value to 1 based on final ordering
    even_spacing = np.linspace(min_cluster_value, 1, num_final_clusters)
    final_mapping = {}
    for rank, (cid, avg_depth, avg_lum) in enumerate(final_order):
        # Blend even spacing with the cluster's average depth using depth_strength.
        # (When depth_strength=0, purely even spacing; when 1, purely avg_depth.)
        blended_value = (1 - depth_strength) * even_spacing[
            rank
        ] + depth_strength * avg_depth
        blended_value = np.clip(blended_value, min_cluster_value, 1)
        final_mapping[cid] = blended_value

    # ---------------------------
    # Step 11: Create new normalized label image and convert to logits.
    # ---------------------------
    new_labels = np.vectorize(lambda x: final_mapping[x])(final_labels).astype(
        np.float32
    )
    if new_labels.max() > 0:
        new_labels = new_labels / new_labels.max()
    pixel_height_logits = np.log((new_labels + eps) / (1 - new_labels + eps))
    return pixel_height_logits


def tsp_simulated_annealing(
    band_reps,
    start_band,
    end_band,
    initial_order=None,
    initial_temp=1000,
    cooling_rate=0.995,
    num_iter=10000,
):
    """
    Solve the band ordering problem using simulated annealing.

    Args:
        band_reps (list or np.ndarray): List of Lab color representations.
        start_band (int): Index for the darkest band.
        end_band (int): Index for the brightest band.
        initial_order (list, optional): Initial ordering of band indices.
        initial_temp (float): Starting temperature.
        cooling_rate (float): Factor to cool the temperature.
        num_iter (int): Maximum number of iterations.

    Returns:
        list: An ordering of band indices from start_band to end_band.
    """
    if initial_order is None:
        # Use a simple ordering: start, middle bands as given, then end.
        middle_indices = [
            i for i in range(len(band_reps)) if i not in (start_band, end_band)
        ]
        order = [start_band] + middle_indices + [end_band]
    else:
        order = initial_order.copy()

    def total_distance(order):
        return sum(
            np.linalg.norm(band_reps[order[i]] - band_reps[order[i + 1]])
            for i in range(len(order) - 1)
        )

    current_distance = total_distance(order)
    best_order = order.copy()
    best_distance = current_distance
    temp = initial_temp

    for _ in range(num_iter):
        # Randomly swap two indices in the middle of the order
        new_order = order.copy()
        idx1, idx2 = random.sample(range(1, len(order) - 1), 2)
        new_order[idx1], new_order[idx2] = new_order[idx2], new_order[idx1]

        new_distance = total_distance(new_order)
        delta = new_distance - current_distance

        # Accept the new order if it improves or with a probability to escape local minima
        if delta < 0 or np.exp(-delta / temp) > random.random():
            order = new_order.copy()
            current_distance = new_distance
            if current_distance < best_distance:
                best_order = order.copy()
                best_distance = current_distance

        temp *= cooling_rate
        if temp < 1e-6:
            break
    return best_order


def choose_optimal_num_bands(centroids, min_bands=2, max_bands=15, random_seed=None):
    """
    Determine the optimal number of clusters (bands) for the centroids
    by maximizing the silhouette score.

    Args:
        centroids (np.ndarray): Array of centroid colors (e.g., shape (n_clusters, 3)).
        min_bands (int): Minimum number of clusters to try.
        max_bands (int): Maximum number of clusters to try.
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        int: Optimal number of bands.
    """
    best_num = min_bands
    best_score = -1

    for num in range(min_bands, max_bands + 1):
        kmeans = KMeans(n_clusters=num, random_state=random_seed).fit(centroids)
        labels = kmeans.labels_
        # If there's only one unique label, skip to avoid errors.
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(centroids, labels)
        if score > best_score:
            best_score = score
            best_num = num

    return best_num


def init_height_map(target, max_layers, h, eps=1e-6, random_seed=None):
    """
    Initialize pixel height logits based on the luminance of the target image.

    Assumes target is a jnp.array of shape (H, W, 3) in the range [0, 255].
    Uses the formula: L = 0.299*R + 0.587*G + 0.114*B.

    Args:
        target (jnp.ndarray): The target image array with shape (H, W, 3).

    Returns:
        jnp.ndarray: The initialized pixel height logits.
    """

    target_np = np.asarray(target).reshape(-1, 3)

    kmeans = KMeans(n_clusters=max_layers, random_state=random_seed).fit(target_np)
    labels = kmeans.labels_
    labels = labels.reshape(target.shape[0], target.shape[1])
    centroids = kmeans.cluster_centers_

    def luminance(col):
        return 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]

    # --- Step 2: Second clustering of centroids into bands ---
    num_bands = choose_optimal_num_bands(
        centroids, min_bands=8, max_bands=10, random_seed=random_seed
    )
    band_kmeans = KMeans(n_clusters=num_bands, random_state=random_seed).fit(centroids)
    band_labels = band_kmeans.labels_

    # Group centroids by band and sort within each band by luminance
    bands = []  # each entry will be (band_avg_luminance, sorted_indices_in_this_band)
    for b in range(num_bands):
        indices = np.where(band_labels == b)[0]
        if len(indices) == 0:
            continue
        lum_vals = np.array([luminance(centroids[i]) for i in indices])
        sorted_indices = indices[np.argsort(lum_vals)]
        band_avg = np.mean(lum_vals)
        bands.append((band_avg, sorted_indices))

    # --- Step 3: Compute a representative color for each band in Lab space ---
    # (Using the average of the centroids in that band)
    band_reps = []  # will hold Lab colors
    for _, indices in bands:
        band_avg_rgb = np.mean(centroids[indices], axis=0)
        # Normalize if needed (assumes image pixel values are 0-255)
        band_avg_rgb_norm = (
            band_avg_rgb / 255.0 if band_avg_rgb.max() > 1 else band_avg_rgb
        )
        # Convert to Lab (expects image in [0,1])
        lab = rgb2lab(np.array([[band_avg_rgb_norm]]))[0, 0, :]
        band_reps.append(lab)

    # --- Step 4: Identify darkest and brightest bands based on L channel ---
    L_values = [lab[0] for lab in band_reps]
    start_band = np.argmin(L_values)  # darkest band index
    end_band = np.argmax(L_values)  # brightest band index

    # --- Step 5: Find the best ordering for the middle bands ---
    # We want to order the bands so that the total perceptual difference (Euclidean distance in Lab)
    # between consecutive bands is minimized, while forcing the darkest band first and brightest band last.
    all_indices = list(range(len(bands)))
    middle_indices = [i for i in all_indices if i not in (start_band, end_band)]

    min_total_distance = np.inf
    best_order = None
    total = len(middle_indices) * len(middle_indices)
    # Try all permutations of the middle bands
    ie = 0
    tbar = tqdm(
        permutations(middle_indices),
        total=total,
        desc="Finding best ordering for color bands:",
    )
    for perm in tbar:
        candidate = [start_band] + list(perm) + [end_band]
        total_distance = 0
        for i in range(len(candidate) - 1):
            total_distance += np.linalg.norm(
                band_reps[candidate[i]] - band_reps[candidate[i + 1]]
            )
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_order = candidate
            tbar.set_description(
                f"Finding best ordering for color bands: Total distance = {min_total_distance:.2f}"
            )
        ie += 1
        if ie > 500000:
            break

    new_order = []
    for band_idx in best_order:
        # Each band tuple is (band_avg, sorted_indices)
        new_order.extend(bands[band_idx][1].tolist())

    # Remap each pixel's label so that it refers to its new palette index
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order)}
    new_labels = np.vectorize(lambda x: mapping[x])(labels)

    new_labels = new_labels.astype(np.float32) / new_labels.max()

    normalized_lum = np.array(new_labels, dtype=np.float32)
    # convert out to inverse sigmoid logit function
    pixel_height_logits = np.log((normalized_lum + eps) / (1 - normalized_lum + eps))

    H, W, _ = target.shape
    return pixel_height_logits
