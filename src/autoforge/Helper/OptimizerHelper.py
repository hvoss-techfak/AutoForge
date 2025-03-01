from itertools import permutations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from tqdm import tqdm


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

    A = 0.1215
    k = 61.6970
    b = 0.4773

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
        TD_i = torch.matmul(p_i, material_TDs) * 0.1
        TD_i = torch.clamp(TD_i, 1e-8, 1e8)
        opac = A * torch.log1p(k * (eff_thick / TD_i)) + b * (eff_thick / TD_i)
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
    num_bands = 9
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

    normalized_lum = np.array(new_labels, dtype=np.float64)
    # convert out to inverse sigmoid logit function
    pixel_height_logits = np.log((normalized_lum + eps) / (1 - normalized_lum + eps))

    H, W, _ = target.shape
    return pixel_height_logits
