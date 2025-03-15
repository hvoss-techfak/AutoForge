import torch
from tqdm import tqdm

from autoforge.Helper.OptimizerHelper import discretize_solution, composite_image_disc
from autoforge.Loss.LossFunctions import compute_loss
from autoforge.Modules.Optimizer import FilamentOptimizer


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
    tau_for_comp: float,
    perception_loss_module: torch.nn.Module,
) -> torch.Tensor:
    """
    Iteratively merge materials until #distinct <= max_colors_allowed.
    Using mode='discrete' on the composite_image, but with
    artificially constructed logits from disc_global.
    """
    num_materials = optimizer.material_colors.shape[0]

    disc_global, pixel_height_logits = optimizer.get_discretized_solution(best=True)

    def get_image_loss(dg_test):
        # Turn discrete array [max_layers] into [max_layers, num_materials] "logits"
        logits_for_disc = disc_to_logits(
            dg_test, num_materials=num_materials, big_pos=1e5
        )

        with torch.no_grad():
            out_im = optimizer.get_best_discretized_image(
                custom_global_logits=logits_for_disc
            )
            loss = compute_loss(
                material_assignment=dg_test,  # shape [max_layers], discrete
                comp=out_im,
                target=optimizer.target,
                perception_loss_module=perception_loss_module,
                tau_global=tau_for_comp,
                num_materials=num_materials,
                add_penalty_loss=False,
            )

        # For the loss, we pass the actual dg_test so that compute_loss() knows it’s a discrete assignment
        return loss

    best_dg = disc_global.clone()
    best_loss = get_image_loss(best_dg)

    distinct_mats = torch.unique(best_dg)
    tbar = tqdm(
        range(100),
        desc=f"Pruning colors down to {max_colors_allowed} or until loss increases. Current colors: {len(distinct_mats)}, Loss: {best_loss:.4f}",
    )
    while True:
        found_merge = False
        best_merge_loss = None
        best_merge_dg_candidate = None

        distinct_mats = torch.unique(best_dg)
        tbar.set_description(
            f"Pruning colors down to {max_colors_allowed} or until loss increases. Current colors: {len(distinct_mats)}, Loss: {best_loss:.4f}",
        )
        tbar.update(1)

        for c_from in distinct_mats:
            for c_to in distinct_mats:
                if c_to == c_from:
                    continue
                # Merge color c_from -> c_to
                dg_test = merge_color(best_dg, c_from.item(), c_to.item())
                test_loss = get_image_loss(dg_test)
                if best_merge_loss is None or test_loss < best_merge_loss:
                    best_merge_loss = test_loss
                    best_merge_dg_candidate = dg_test

        if best_merge_dg_candidate is not None:
            if best_merge_loss < best_loss or len(distinct_mats) > max_colors_allowed:
                best_dg = best_merge_dg_candidate
                best_loss = best_merge_loss
                found_merge = True

        if not found_merge:
            break
    tbar.close()

    # set the best solution in the optimizer
    optimizer.best_params["global_logits"] = disc_to_logits(
        best_dg, num_materials=num_materials, big_pos=1e5
    )

    return best_dg


def prune_num_swaps(
    optimizer: FilamentOptimizer,
    max_swaps_allowed: int,
    tau_for_comp: float,
    perception_loss_module: torch.nn.Module,
) -> torch.Tensor:
    """
    Iteratively reduce the number of color boundaries until <= max_swaps_allowed,
    using mode='discrete' compositing on the “fake logits.”
    """
    num_materials = optimizer.material_colors.shape[0]
    disc_global, pixel_height_logits = optimizer.get_discretized_solution(best=True)

    def get_image_loss(dg_test):
        logits_for_disc = disc_to_logits(
            dg_test, num_materials=num_materials, big_pos=1e5
        )

        with torch.no_grad():
            out_im = optimizer.get_best_discretized_image(
                custom_global_logits=logits_for_disc
            )
            loss = compute_loss(
                material_assignment=dg_test,  # shape [max_layers], discrete
                comp=out_im,
                target=optimizer.target,
                perception_loss_module=perception_loss_module,
                tau_global=tau_for_comp,
                num_materials=num_materials,
                add_penalty_loss=False,
            )

        # For the loss, we pass the actual dg_test so that compute_loss() knows it’s a discrete assignment
        return loss

    best_dg = disc_global.clone()
    best_loss = get_image_loss(best_dg)
    tbar = tqdm(
        range(100),
        desc=f"Pruning swaps down to {max_swaps_allowed} or until loss increases. Current swaps: {len(find_color_bands(best_dg)) - 1}, Loss = {best_loss:.4f}",
    )
    while True:
        bands = find_color_bands(best_dg)
        num_swaps = len(bands) - 1
        tbar.set_description(
            f"Pruning swaps down to {max_swaps_allowed} or until loss increases. Current swaps: {num_swaps}, Loss = {best_loss:.4f}",
        )
        tbar.update(1)

        best_merge_loss = None
        best_merge_dg_candidate = None

        for i in range(num_swaps):
            band_a = bands[i]
            band_b = bands[i + 1]
            if band_a[2] == band_b[2]:
                continue
            dg_fwd = merge_bands(best_dg, band_a, band_b, direction="forward")
            loss_fwd = get_image_loss(dg_fwd)
            dg_bwd = merge_bands(best_dg, band_a, band_b, direction="backward")
            loss_bwd = get_image_loss(dg_bwd)

            if loss_fwd < loss_bwd:
                candidate_loss = loss_fwd
                candidate_dg = dg_fwd
            else:
                candidate_loss = loss_bwd
                candidate_dg = dg_bwd

            if best_merge_loss is None or candidate_loss < best_merge_loss:
                best_merge_loss = candidate_loss
                best_merge_dg_candidate = candidate_dg

        if best_merge_dg_candidate is not None and (
            best_merge_loss < best_loss or num_swaps > max_swaps_allowed
        ):
            best_dg = best_merge_dg_candidate
            best_loss = best_merge_loss
        else:
            break
    tbar.close()

    # set the best solution in the optimizer
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
    pruning_max_layers: int = 1e6,
    tolerance: float = 0.02,
):
    """
    Iteratively search for the best layer to remove.
    At each iteration, evaluate removal of each layer, and choose the candidate that gives
    the highest loss decrease (or, if no removal improves loss and the current number of layers
    exceeds pruning_max_layers, the removal with the minimal loss increase within tolerance).
    After a removal is accepted, the search restarts from the beginning.

    Args:
        params (dict): Dictionary with keys "global_logits" and "pixel_height_logits".
        final_tau (float): Final tau (decay) value for discretization.
        h (float): Layer height.
        target (torch.Tensor): Target image tensor.
        material_colors (torch.Tensor): Tensor of material colors.
        material_TDs (torch.Tensor): Tensor of material transmission parameters.
        background (torch.Tensor): Background color tensor.
        rng_seed (int): Random seed for discretization.
        perception_loss_module (torch.nn.Module): Perceptual loss module.
        tolerance (float): Acceptable increase in loss.
        pruning_min_layers (int): Minimum number of layers to keep.
        pruning_max_layers (int): Maximum number of layers allowed after pruning.

    Returns:
        current_params (dict): Updated parameters with redundant layers removed.
        best_loss (float): Loss corresponding to the pruned solution.
        current_max_layers (int): New number of layers after pruning.
    """

    current_max_layers = optimizer.max_layers
    # Compute baseline composite and loss.
    comp = optimizer.get_best_discretized_image()

    best_loss = compute_loss(
        None,
        comp,
        optimizer.target,
        None,
        None,
        None,
        add_penalty_loss=False,
    ).item()

    from tqdm import tqdm

    tbar = tqdm(
        desc=f"Pruning redundant layers: Loss {best_loss:.4f}",
        total=current_max_layers - 1,
    )
    removed_layers = 0

    improvement = True
    # Continue iterating while we can still remove a layer.
    # We stop if no candidate qualifies and we are not forced to prune by pruning_max_layers.
    while current_max_layers > pruning_min_layers and (
        improvement or current_max_layers > pruning_max_layers
    ):
        tbar.update(1)
        improvement = False
        best_candidate = None
        best_candidate_loss = 1e10
        # For each layer candidate, compute the loss if that layer were removed.
        for layer in range(current_max_layers):
            candidate_params, candidate_max_layers = remove_layer_from_solution(
                optimizer.best_params,
                layer,
                optimizer.final_tau,
                optimizer.h,
                current_max_layers,
                optimizer.best_seed,
            )
            with torch.no_grad():
                candidate_comp = composite_image_disc(
                    candidate_params["pixel_height_logits"],
                    candidate_params["global_logits"],
                    optimizer.final_tau,
                    optimizer.final_tau,
                    optimizer.h,
                    candidate_max_layers,
                    optimizer.material_colors,
                    optimizer.material_TDs,
                    optimizer.background,
                    rng_seed=optimizer.best_seed,
                )
                candidate_loss = compute_loss(
                    None,
                    candidate_comp,
                    optimizer.target,
                    None,
                    None,
                    None,
                    add_penalty_loss=False,
                ).item()

            if candidate_loss <= best_candidate_loss + best_candidate_loss * tolerance:
                best_candidate = candidate_params
                best_candidate_loss = candidate_loss

        # If we found a candidate, remove it and restart the search.
        if best_candidate is not None:
            if (
                best_candidate_loss < best_loss
                or current_max_layers > pruning_max_layers
            ):
                removed_layers += 1
                # Update the progress bar description.
                tbar.set_description(
                    f"Pruning redundant layers: Loss {best_candidate_loss:.4f}, Removed {removed_layers}, new max layers {current_max_layers - 1}"
                )
                current_params = best_candidate
                current_max_layers = current_params["global_logits"].shape[0]

                # set optimizer
                optimizer.best_params = current_params
                optimizer.max_layers = current_max_layers

                best_loss = best_candidate_loss
                improvement = True
            else:
                break
        else:
            # No candidate met the criteria.
            break

    tbar.close()
    return current_params, best_loss, current_max_layers
