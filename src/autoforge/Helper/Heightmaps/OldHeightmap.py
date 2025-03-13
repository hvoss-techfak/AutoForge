from itertools import permutations

import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


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
        max_layers (int): Maximum number of layers.
        h (float): Height of each layer.
        eps (float): Small constant to avoid division by zero.
        random_seed (int): Random seed for reproducibility.

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
