import math

import torch
from tqdm import tqdm
from joblib import Parallel, delayed
import threading

from autoforge.Helper.OptimizerHelper import discretize_solution, composite_image_disc
from autoforge.Loss.LossFunctions import compute_loss
from autoforge.Modules.Optimizer import FilamentOptimizer

# One global lock that serialises every call that needs GPU / VRAM
_gpu_lock = threading.Lock()

def disc_to_logits(
    dg: torch.Tensor, num_materials: int, big_pos: float = 1e5
) -> torch.Tensor:
    """
    Convert a discrete [max_layers] material assignment into [max_layers, num_materials] "logits"
    suitable for compositing with mode='discrete'.

    We place a very large positive logit on the chosen material index, and large negative on others.
    That ensures Gumbel softmax with hard=True picks that color with probability ~1.
    """
    max_layers = dg.size(0)
    # Start with a big negative for all materials:
    logits = dg.new_full(
        (max_layers, num_materials), fill_value=-big_pos, dtype=torch.float32
    )
    # Put a large positive at each chosen ID:
    for layer_idx in range(max_layers):
        c_id = dg[layer_idx].item()
        logits[layer_idx, c_id] = big_pos
    return logits



def prune_num_colors(
    optimizer: FilamentOptimizer,
    max_colors_allowed: int,
    tau_for_comp: float,                # kept for API compatibility, not used here
    perception_loss_module: torch.nn.Module,  # kept for API compatibility, not used here
    n_jobs: int | None = -1,            # extra arg: how many workers joblib should start
) -> torch.Tensor:
    """
    Iteratively merge materials until the number of distinct colors is
    <= max_colors_allowed or until merging would worsen the loss.

    Joblib parallelises the evaluation of all merge candidates in each
    iteration, but every GPU call is protected by _gpu_lock so only one
    of them is active at any moment.
    """
    num_materials = optimizer.material_colors.shape[0]

    # Start from the optimiser’s current best discrete solution
    disc_global, _ = optimizer.get_discretized_solution(best=True)

    def score_color(
            dg_base: torch.Tensor,
            c_from: int,
            c_to: int,
            num_materials: int,
            optimizer: FilamentOptimizer,
    ) -> tuple[float, torch.Tensor]:
        """
        Helper that creates a merge candidate and returns its loss.
        The VRAM-hungry part is wrapped with _gpu_lock so it never runs concurrently.
        """
        dg_test = merge_color(dg_base, c_from, c_to)

        # Build logits for the discrete assignment
        logits_for_disc = disc_to_logits(dg_test, num_materials=num_materials, big_pos=1e5)

        # Only this block touches the GPU, so we protect it
        with _gpu_lock:
            with torch.no_grad():
                out_im = optimizer.get_best_discretized_image(
                    custom_global_logits=logits_for_disc
                )
                loss = compute_loss(comp=out_im, target=optimizer.target)

        return loss, dg_test



    # Convenience wrapper to compute the loss of the current best_dg
    def get_image_loss(dg_test: torch.Tensor) -> float:
        logits_for_disc = disc_to_logits(
            dg_test, num_materials=num_materials, big_pos=1e5
        )
        with _gpu_lock:
            with torch.no_grad():
                out_im = optimizer.get_best_discretized_image(
                    custom_global_logits=logits_for_disc
                )
                loss = compute_loss(comp=out_im, target=optimizer.target)
        return loss

    best_dg = disc_global.clone()
    best_loss = get_image_loss(best_dg)

    tbar = tqdm(total=100, leave=False)
    while True:
        distinct_mats = torch.unique(best_dg)


        # Generate all possible one-to-one merges for the current palette
        merge_pairs = [
            (c_from.item(), c_to.item())
            for c_from in distinct_mats
            for c_to in distinct_mats
            if c_from != c_to
        ]

        tbar.set_description(
            f"Current colors: {len(distinct_mats)}  |  Best loss: {best_loss:.4f} | Possible Merge Pairs: {len(merge_pairs)}"
        )
        tbar.update(1)

        if not merge_pairs:  # nothing left to merge
            break

        # Evaluate every merge candidate in parallel
        cand_results = Parallel(
            n_jobs=n_jobs,
            backend="threading",       # threads let us share the lock easily
            prefer="threads",
        )(
            delayed(score_color)(
                best_dg, c_from, c_to, num_materials, optimizer
            )
            for c_from, c_to in merge_pairs
        )

        # Select the best candidate (lowest loss)
        best_merge_loss, best_merge_dg_candidate = min(cand_results, key=lambda x: x[0])

        # Accept the merge if it improves the loss
        # or if we are still above the allowed color budget
        if (
            best_merge_loss < best_loss
            or len(distinct_mats) > max_colors_allowed
        ):
            best_dg = best_merge_dg_candidate
            best_loss = best_merge_loss
        else:
            break  # no further improvement and we are within the budget

    tbar.close()

    # Store result back in the optimiser so downstream code can pick it up
    optimizer.best_params["global_logits"] = disc_to_logits(
        best_dg, num_materials=num_materials, big_pos=1e5
    )

    return best_dg


def prune_num_swaps(
    optimizer: FilamentOptimizer,
    max_swaps_allowed: int,
    tau_for_comp: float,                       # kept for API compatibility
    perception_loss_module: torch.nn.Module,   # kept for API compatibility
    n_jobs: int | None = -1,                   # how many workers joblib should start
) -> torch.Tensor:
    """
    Reduce the number of color boundaries until it is below or equal to
    max_swaps_allowed, stopping earlier if any further merge would worsen the loss.
    Every candidate merge is scored in parallel but GPU work is serialised.
    """
    num_materials = optimizer.material_colors.shape[0]
    disc_global, _ = optimizer.get_discretized_solution(best=True)

    # Helper for the current best_dg
    def get_image_loss(dg_test: torch.Tensor) -> float:
        logits_for_disc = disc_to_logits(
            dg_test, num_materials=num_materials, big_pos=1e5
        )
        with _gpu_lock:
            with torch.no_grad():
                out_im = optimizer.get_best_discretized_image(
                    custom_global_logits=logits_for_disc
                )
                loss = compute_loss(comp=out_im, target=optimizer.target)
        return loss

    def score_swap(
            dg_base: torch.Tensor,
            band_a,
            band_b,
            direction: str,
            num_materials: int,
            optimizer: FilamentOptimizer,
    ) -> tuple[float, torch.Tensor]:
        """
        Merge two neighbouring bands in the chosen direction and return (loss, dg_candidate).
        The VRAM intensive part is wrapped by _gpu_lock so it runs one at a time.
        """
        dg_test = merge_bands(dg_base, band_a, band_b, direction=direction)
        logits_for_disc = disc_to_logits(dg_test, num_materials=num_materials, big_pos=1e5)

        with _gpu_lock:
            with torch.no_grad():
                out_im = optimizer.get_best_discretized_image(
                    custom_global_logits=logits_for_disc
                )
                loss = compute_loss(comp=out_im, target=optimizer.target)

        return loss, dg_test

    best_dg = disc_global.clone()
    best_loss = get_image_loss(best_dg)

    tbar = tqdm(total=100, leave=False)
    while True:
        bands = find_color_bands(best_dg)
        num_swaps = len(bands) - 1


        if num_swaps == 0:
            break

        # Build all forward and backward merge candidates for the current arrangement
        merge_specs = [
            (bands[i], bands[i + 1], dirn)
            for i in range(num_swaps)
            for dirn in ("forward", "backward")
            if bands[i][2] != bands[i + 1][2]   # skip if both bands already share a color
        ]

        tbar.set_description(
            f"Current swaps: {num_swaps}  |  Best loss: {best_loss:.4f} | Possible Merge Swaps: {len(merge_specs)}"
        )
        tbar.update(1)

        if not merge_specs:
            break

        # Parallel evaluation of all candidates
        cand_results = Parallel(
            n_jobs=n_jobs,
            backend="threading",
            prefer="threads",
        )(
            delayed(score_swap)(
                best_dg, band_a, band_b, direction, num_materials, optimizer
            )
            for band_a, band_b, direction in merge_specs
        )

        best_merge_loss, best_merge_dg_candidate = min(cand_results, key=lambda x: x[0])

        # Accept if it improves loss or if still above the swap budget
        if best_merge_loss < best_loss or num_swaps > max_swaps_allowed:
            best_dg = best_merge_dg_candidate
            best_loss = best_merge_loss
        else:
            break

    tbar.close()

    # Persist the result inside the optimiser
    optimizer.best_params["global_logits"] = disc_to_logits(
        best_dg, num_materials=num_materials, big_pos=1e5
    )

    return best_dg

def merge_color(dg: torch.Tensor, c_from: int, c_to: int) -> torch.Tensor:
    """
    Return a copy of dg where every layer with material c_from is replaced by c_to.
    """
    dg_new = dg.clone()
    dg_new[dg_new == c_from] = c_to
    return dg_new


def find_color_bands(dg: torch.Tensor):
    """
    Return a list of (start_idx, end_idx, color_id) for each contiguous band
    in 'dg'. Example: if dg = [0,0,1,1,1,2,2], we get:
       [(0,1,0), (2,4,1), (5,6,2)]
    """
    bands = []
    dg_cpu = dg.detach().cpu().numpy()
    start_idx = 0
    current_color = dg_cpu[0]
    n = len(dg_cpu)

    for i in range(1, n):
        if dg_cpu[i] != current_color:
            bands.append((start_idx, i - 1, current_color))
            start_idx = i
            current_color = dg_cpu[i]
    # finish last band
    bands.append((start_idx, n - 1, current_color))

    return bands


def merge_bands(
    dg: torch.Tensor, band_a: (int, int, int), band_b: (int, int, int), direction: str
):
    """
    Merge band_a and band_b. If direction=="forward", unify band_b's color to band_a's color;
    otherwise unify band_a's color to band_b's color.
    band_a, band_b = (start_idx, end_idx, color_id)
    """
    dg_new = dg.clone()
    c_a = band_a[2]
    c_b = band_b[2]
    if direction == "forward":
        dg_new[band_b[0] : band_b[1] + 1] = c_a
    else:
        dg_new[band_a[0] : band_a[1] + 1] = c_b
    return dg_new


def remove_layer_from_solution(
    params, layer_to_remove, final_tau, h, current_max_layers, rng_seed
):
    """
    Remove one layer from the solution.

    Args:
        params (dict): Current parameters with keys "global_logits" and "pixel_height_logits".
        layer_to_remove (int): Candidate layer index to remove.
        final_tau (float): Final tau value used in discretization/compositing.
        h (float): Layer height.
        current_max_layers (int): Current total number of layers.
        rng_seed (int): Seed used in discretization/compositing.

    Returns:
        new_params (dict): New parameters with the candidate layer removed.
        new_max_layers (int): Updated number of layers.
    """
    # Get the current discrete height image (used to decide which pixels need adjusting)
    _, disc_height = discretize_solution(
        params, final_tau, h, current_max_layers, rng_seed
    )

    # Remove the candidate layer from the global (color) assignment.
    new_global_logits = torch.cat(
        [
            params["global_logits"][:layer_to_remove],
            params["global_logits"][layer_to_remove + 1 :],
        ],
        dim=0,
    )
    new_max_layers = current_max_layers - 1

    # Compute current effective height: height = (current_max_layers * h) * sigmoid(pixel_height_logits)
    current_height = (
        current_max_layers * h * torch.sigmoid(params["pixel_height_logits"])
    )

    # For pixels where the discrete height is at or above the removed layer, subtract h.
    new_height = current_height.clone()
    mask = disc_height >= layer_to_remove  # <-- Changed from > to >= here
    new_height[mask] = new_height[mask] - h

    # Invert the sigmoid mapping:
    # We need new_pixel_height_logits such that:
    #    sigmoid(new_pixel_height_logits) = new_height / (new_max_layers * h)
    eps = 1e-6
    new_ratio = torch.clamp(new_height / (new_max_layers * h), eps, 1 - eps)
    new_pixel_height_logits = torch.log(new_ratio) - torch.log(1 - new_ratio)

    new_params = {
        "global_logits": new_global_logits,
        "pixel_height_logits": new_pixel_height_logits,
    }
    return new_params, new_max_layers


def prune_redundant_layers(
    optimizer: FilamentOptimizer,
    perception_loss_module,
    pruning_min_layers: int = 0,
    pruning_max_layers: int = int(1e6),
    tolerance: float = 0.10,          # kept for API compatibility, still unused
    n_jobs: int | None = -1,          # new: how many workers joblib should spawn
):
    """
    Iteratively drop layers until no removal improves the loss
    (unless still above pruning_max_layers, in which case we accept the
    least harmful removal).  All candidates in an iteration are scored
    in parallel, but CUDA work is serialised by _gpu_lock.
    """
    current_max_layers = optimizer.max_layers

    def score_layer(
            layer_idx: int,
            base_params: dict,
            final_tau: float,
            h: float,
            current_max_layers: int,
            best_seed: int,
            optimizer: FilamentOptimizer,
    ) -> tuple[float, dict, int]:
        """
        Remove layer `layer_idx`, composite the image, measure loss.
        GPU work is wrapped with _gpu_lock so only one thread touches VRAM.
        """
        cand_params, cand_max_layers = remove_layer_from_solution(
            base_params,
            layer_idx,
            final_tau,
            h,
            current_max_layers,
            best_seed,
        )

        with _gpu_lock:
            with torch.no_grad():
                cand_comp = composite_image_disc(
                    cand_params["pixel_height_logits"],
                    cand_params["global_logits"],
                    final_tau,
                    final_tau,
                    h,
                    cand_max_layers,
                    optimizer.material_colors,
                    optimizer.material_TDs,
                    optimizer.background,
                    rng_seed=best_seed,
                )
                cand_loss = compute_loss(cand_comp, optimizer.target).item()

        return cand_loss, cand_params, cand_max_layers

    # Baseline composite and loss
    with _gpu_lock:
        with torch.no_grad():
            comp = optimizer.get_best_discretized_image()
            best_loss = compute_loss(comp, optimizer.target).item()

    tbar = tqdm(
        desc=f"Pruning redundant layers  |  Loss {best_loss:.4f}",
        total=max(current_max_layers - 1, 0),
        leave=False,
    )
    removed_layers = 0
    improvement = True

    while current_max_layers > pruning_min_layers and (
        improvement or current_max_layers > pruning_max_layers
    ):
        tbar.update(1)
        improvement = False

        # Evaluate removal of every layer in parallel
        cand_results = Parallel(
            n_jobs=n_jobs,
            backend="threading",
            prefer="threads",
        )(
            delayed(score_layer)(
                layer_idx,
                optimizer.best_params,
                optimizer.final_tau,
                optimizer.h,
                current_max_layers,
                optimizer.best_seed,
                optimizer,
            )
            for layer_idx in range(current_max_layers)
        )

        # Select the candidate with the lowest loss
        best_cand_loss, best_cand_params, best_cand_max_layers = min(
            cand_results, key=lambda x: x[0]
        )

        # Accept if it improves loss or if we are forced to prune further
        if best_cand_loss <= best_loss or current_max_layers > pruning_max_layers:
            removed_layers += 1
            best_loss = best_cand_loss
            current_max_layers = best_cand_max_layers

            # Persist the new best solution
            optimizer.best_params = best_cand_params
            optimizer.max_layers = current_max_layers

            tbar.set_description(
                f"Pruning redundant layers  |  Loss {best_loss:.4f}  |  Removed {removed_layers}  |  Layers left {current_max_layers}"
            )
            improvement = True
        else:
            break

    tbar.close()
    return optimizer.best_params, best_loss, current_max_layers

def remove_outlier_pixels(
    height_logits: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    For every pixel in `height_logits`, if at least six out of its 8 neighbors have an
    absolute difference greater than `threshold` from the center pixel, replace the pixel
    with the average of only those neighbors exceeding the threshold.

    Args:
        height_logits (torch.Tensor): 2D tensor representing the depth map.
        threshold (float): The threshold value for differences.

    Returns:
        torch.Tensor: The cleaned depth map.
    """
    # Define the eight neighbor shifts (row_shift, col_shift)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Collect neighbors using torch.roll (note: this wraps around at the edges)
    neighbors = []
    for dx, dy in shifts:
        shifted = torch.roll(
            torch.roll(height_logits, shifts=dx, dims=0), shifts=dy, dims=1
        )
        neighbors.append(shifted)

    # Stack the neighbors to shape (8, H, W)
    neighbors = torch.stack(neighbors, dim=0)

    # Compute the absolute difference between each neighbor and the center pixel
    diff = torch.abs(neighbors - height_logits)

    # Create a boolean mask for neighbors exceeding the threshold
    valid = diff > threshold  # shape (8, H, W)
    count_valid = valid.sum(dim=0)  # number of neighbors exceeding threshold per pixel

    # Determine which pixels should be replaced (if at least six neighbors exceed threshold)
    mask = count_valid >= 6

    # Compute the sum of valid neighbor values for each pixel
    # We use torch.where to zero out values not exceeding the threshold.
    valid_neighbors = torch.where(
        valid,
        neighbors,
        torch.tensor(0.0, dtype=neighbors.dtype, device=neighbors.device),
    )
    sum_valid = valid_neighbors.sum(dim=0)

    # Compute the average for the valid neighbors.
    # We use count_valid as the divisor; note that for pixels not meeting the criteria, the value is unused.
    avg_valid = sum_valid / count_valid.clamp(min=1)

    # Create a copy to update the pixels in outlier regions
    cleaned_height_logits = height_logits.clone()
    cleaned_height_logits[mask] = avg_valid[mask]

    num_changed = mask.sum().item()

    return cleaned_height_logits


def prune_fireflies(optimizer, start_threshold=10, auto_set=True):
    """
    Iteratively reduces the threshold, computes the loss for each,
    and returns the pixel_height_logits with the best loss.

    Args:
        optimizer: Object that provides get_best_discretized_image() and has target attribute.
        compute_loss (function): Function to compute loss; expects parameters comp and target.
        start_threshold (float): Initial threshold value.
        auto_set (bool): Automatically set the best pixel_height_logits in the optimizer.

    Returns:
        best_custom_height_logits (torch.Tensor): The processed depth map with best loss.
        best_threshold (float): The threshold value that achieved the best loss.
        best_loss (float): The best loss achieved.
    """
    # Generate a series of threshold values between start_threshold and end_threshold.
    pixel_height_logits = optimizer.best_params["pixel_height_logits"]
    best_custom_height_logits = pixel_height_logits
    best_threshold = start_threshold
    th = start_threshold

    with torch.no_grad():
        out_im = optimizer.get_best_discretized_image(
            custom_height_logits=pixel_height_logits
        )
        best_loss = compute_loss(
            comp=out_im,
            target=optimizer.target,
        )
    new_loss = best_loss
    while th > 0.1:
        # Apply your outlier removal with the current threshold.
        custom_height_logits = remove_outlier_pixels(pixel_height_logits, threshold=th)

        # Evaluate the resulting image by computing the loss.
        with torch.no_grad():
            out_im = optimizer.get_best_discretized_image(
                custom_height_logits=custom_height_logits
            )
            loss = compute_loss(
                comp=out_im,
                target=optimizer.target,
            )

        # print(f"Threshold: {th:.3f}, Loss: {loss:.4f}")
        # Track the best performing threshold.
        if loss < best_loss * 1.05:
            new_loss = best_loss
            best_custom_height_logits = custom_height_logits
            best_threshold = th

        th *= 0.95

    print(f"New loss: {new_loss:.4f}, Best threshold: {best_threshold:.3f}")
    if auto_set:
        optimizer.best_params["pixel_height_logits"] = best_custom_height_logits

    return best_custom_height_logits


def smooth_coplanar_faces(
    height_logits: torch.Tensor, angle_threshold: float
) -> torch.Tensor:
    """
    Smooths regions in the depth map that are considered coplanar.
    For each pixel, this function computes an approximate surface normal via finite differences
    and compares it to the normals of the eight neighboring pixels. Neighbors with an angle
    difference (in degrees) less than `angle_threshold` are considered coplanar. The pixel is
    replaced with the average of itself and its coplanar neighbors.

    Args:
        height_logits (torch.Tensor): 2D tensor representing the depth map.
        angle_threshold (float): Maximum angle difference (in degrees) for neighbors to be
                                 considered coplanar.

    Returns:
        torch.Tensor: The smoothed depth map.
    """
    # Convert the angle threshold from degrees to radians.
    threshold_rad = math.radians(angle_threshold)

    # Compute gradients using central differences via torch.roll (wraps at the edges)
    grad_x = (
        torch.roll(height_logits, shifts=-1, dims=1)
        - torch.roll(height_logits, shifts=1, dims=1)
    ) / 2.0
    grad_y = (
        torch.roll(height_logits, shifts=-1, dims=0)
        - torch.roll(height_logits, shifts=1, dims=0)
    ) / 2.0

    # Approximate the surface normals.
    # For a height function h(x,y), one common estimate is n = [-dh/dx, -dh/dy, 1] (then normalized)
    ones = torch.ones_like(height_logits)
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = ones
    norm = torch.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    # Stack into a tensor of shape (3, H, W)
    normals = torch.stack([normal_x, normal_y, normal_z], dim=0)

    # Define the eight neighbor shifts (for height map, we work with the spatial dims 0 and 1)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Initialize accumulators for the heights and a count of contributing pixels.
    # We always include the center pixel.
    coplanar_sum = height_logits.clone()
    count = torch.ones_like(height_logits)

    # For each neighbor, compute the angle difference between the center and neighbor normals.
    for dx, dy in shifts:
        # Shift the normals to obtain the neighbor's normal at each pixel.
        # Note: the normals tensor shape is (3, H, W) so we shift dims 1 and 2.
        neighbor_normals = torch.roll(
            torch.roll(normals, shifts=dx, dims=1), shifts=dy, dims=2
        )
        # Dot product between the center normals and the neighbor normals.
        dot = (normals * neighbor_normals).sum(dim=0)
        # Clamp dot products for numerical stability.
        dot = dot.clamp(-1.0, 1.0)
        # Compute the angle difference (in radians)
        angle_diff = torch.acos(dot)
        # Determine which neighbors are nearly coplanar.
        mask = angle_diff < threshold_rad
        # Retrieve the corresponding neighbor heights from height_logits.
        neighbor_heights = torch.roll(
            torch.roll(height_logits, shifts=dx, dims=0), shifts=dy, dims=1
        )
        # Add neighbor height values where the coplanar condition is met.
        coplanar_sum += neighbor_heights * mask.float()
        count += mask.float()

    # Compute the average height for the coplanar region.
    smoothed_height_logits = coplanar_sum / count.clamp(min=1)

    return smoothed_height_logits
