import torch
import torch.nn.functional as F


@torch.jit.script
def adaptive_round(
    x: torch.Tensor, tau: float, high_tau: float, low_tau: float, temp: float
) -> torch.Tensor:
    """
    Smooth rounding based on temperature 'tau'.

    Args:
        x (torch.Tensor): The input tensor to be rounded.
        tau (float): The current temperature parameter.
        high_tau (float): The high threshold for the temperature.
        low_tau (float): The low threshold for the temperature.
        temp (float): The temperature parameter for the sigmoid function.

    Returns:
        torch.Tensor: The rounded tensor.
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
    """
    Generate a deterministic random tensor that mimics torch.rand_like.

    Args:
        tensor (torch.Tensor): The input tensor whose shape and device will be used.
        seed (int): The seed for the deterministic random generator.

    Returns:
        torch.Tensor: A tensor with the same shape as the input tensor, filled with deterministic random values.
    """
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
    """
    Apply the Gumbel-Softmax trick in a deterministic manner using a fixed random seed.

    Args:
        logits (torch.Tensor): The input logits tensor.
        tau (float): The temperature parameter for the Gumbel-Softmax.
        hard (bool): If True, the output will be one-hot encoded.
        rng_seed (int): The seed for the deterministic random generator.

    Returns:
        torch.Tensor: The resulting tensor after applying the Gumbel-Softmax trick.
    """
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
def composite_image_cont(
    pixel_height_logits: torch.Tensor,
    global_logits: torch.Tensor,
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,  # [n_materials, 3]
    material_TDs: torch.Tensor,  # [n_materials]
    background: torch.Tensor,  # [3]
) -> torch.Tensor:
    """
    Continuous compositing over all pixels.
    Uses Gumbel softmax with either hard or soft sampling depending on tau_global.

    Args:
        pixel_height_logits (torch.Tensor): Logits for pixel heights, shape [H, W].
        global_logits (torch.Tensor): Logits for global material assignment, shape [max_layers, n_materials].
        tau_height (float): Temperature parameter for height rounding.
        tau_global (float): Temperature parameter for global material assignment.
        h (float): Height of each layer.
        max_layers (int): Maximum number of layers.
        material_colors (torch.Tensor): Tensor of material colors, shape [n_materials, 3].
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters, shape [n_materials].
        background (torch.Tensor): Background color tensor, shape [3].

    Returns:
        torch.Tensor: Composite image tensor, shape [H, W, 3].
    """
    # Compute continuous and discrete layers for each pixel
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logits)
    continuous_layers = pixel_height / h

    # adaptive rounding
    adaptive_layers = adaptive_round(
        continuous_layers, tau_height, high_tau=0.1, low_tau=0.01, temp=0.1
    )

    # "Straight-through" trick to keep gradient from adaptive rounding
    discrete_layers_temp = torch.round(continuous_layers)
    discrete_layers = (
        discrete_layers_temp + (adaptive_layers - discrete_layers_temp).detach()
    ).to(torch.int32)  # shape [H, W]

    # Precompute global material assignments for each layer
    # Use a single Gumbel Softmax call for all layers.
    hard_flag = tau_global < 1e-3
    # global_logits: [max_layers, n_materials]
    p = F.gumbel_softmax(
        global_logits, tau_global, hard=hard_flag, dim=1
    )  # [L, n_materials]

    # Compute color_i and TD_i for each layer in one go.
    # color_i -> [max_layers, 3], TD_i -> [max_layers]
    layer_colors = p @ material_colors  # [L, 3]
    layer_TDs = p @ material_TDs  # [L]
    layer_TDs.clamp_(1e-8, 1e8)

    # Prepare output and compositing variables
    H, W = pixel_height.shape
    comp = torch.zeros(H, W, 3, dtype=torch.float32, device=pixel_height.device)
    remaining = torch.ones(H, W, dtype=torch.float32, device=pixel_height.device)

    # Opacity function parameters
    o = -1.2416557e-02
    A = 9.6407950e-01
    k = 3.4103447e01
    b = -4.1554203e00

    # Composite from top layer (max_layers-1) down to 0
    for i in range(max_layers):
        layer_idx = max_layers - 1 - i
        # Mask of which pixels actually print this layer:
        p_print = (discrete_layers > layer_idx).to(dtype=comp.dtype)

        # Effective thickness for those pixels
        eff_thick = p_print * h

        # Opacity based on thickness and material TD
        TD_i = layer_TDs[layer_idx]
        opac = o + (A * torch.log1p(k * (eff_thick / TD_i)) + b * (eff_thick / TD_i))
        opac = torch.clamp(opac, 0.0, 1.0)

        # Add contribution to comp
        color_i = layer_colors[layer_idx]  # shape [3]
        # We need to broadcast opac: [H, W] -> [H, W, 1]
        comp = comp + (remaining * opac).unsqueeze(-1) * color_i

        # Update the remaining factor
        remaining = remaining * (1.0 - opac)

    # Add background contribution
    comp = comp + remaining.unsqueeze(-1) * background

    # 6. Scale to [0, 255] range
    return comp * 255.0


@torch.jit.script
def composite_image_disc(
    pixel_height_logits: torch.Tensor,  # [H,W]
    global_logits: torch.Tensor,  # [max_layers, n_materials]
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,  # [n_materials, 3]
    material_TDs: torch.Tensor,  # [n_materials]
    background: torch.Tensor,  # [3]
    rng_seed: int = -1,
) -> torch.Tensor:
    """
    Perform discrete compositing over all pixels.

    Args:
        pixel_height_logits (torch.Tensor): Logits for pixel heights, shape [H, W].
        global_logits (torch.Tensor): Logits for global material assignment, shape [max_layers, n_materials].
        tau_height (float): Temperature parameter for height rounding.
        tau_global (float): Temperature parameter for global material assignment.
        h (float): Height of each layer.
        max_layers (int): Maximum number of layers.
        material_colors (torch.Tensor): Tensor of material colors, shape [n_materials, 3].
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters, shape [n_materials].
        background (torch.Tensor): Background color tensor, shape [3].
        rng_seed (int, optional): Random seed for deterministic sampling. Defaults to -1.

    Returns:
        torch.Tensor: Composite image tensor, shape [H, W, 3].
    """

    # -------------------------------------------------------------------------
    # 1) Compute discrete per-pixel layer counts (discrete_layers).
    # -------------------------------------------------------------------------
    #  pixel_height ~ [0, max_layers*h]
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logits)

    #  continuous_layers ~ [0, max_layers]
    continuous_layers = pixel_height / h

    #  Use your "adaptive rounding" trick if desired:
    adaptive_layers = adaptive_round(
        continuous_layers, tau_height, high_tau=0.1, low_tau=0.01, temp=0.1
    )
    discrete_layers_temp = torch.round(continuous_layers)
    discrete_layers = (
        discrete_layers_temp + (adaptive_layers - discrete_layers_temp).detach()
    ).to(torch.int32)  # [H,W]

    # -------------------------------------------------------------------------
    # 2) Pick a single global material per layer, either deterministically
    #    or via gumbel softmax
    # -------------------------------------------------------------------------
    if rng_seed >= 0:
        # Deterministic sampling for each layer
        new_mats_list = []
        for layer_idx in range(max_layers):
            p_i = deterministic_gumbel_softmax(
                global_logits[layer_idx],
                tau_global,
                hard=True,
                rng_seed=(rng_seed + layer_idx),
            )
            mat_i = torch.argmax(p_i, dim=0).to(torch.int32)
            new_mats_list.append(mat_i)
        new_mats = torch.stack(new_mats_list, dim=0)  # [max_layers]
    else:
        # Standard (random) Gumbel softmax
        p_all = F.gumbel_softmax(global_logits, tau_global, hard=True, dim=1)  # [L, M]
        new_mats = torch.argmax(p_all, dim=1).to(torch.int32)  # [max_layers]

    H, W = pixel_height.shape
    device = pixel_height.device

    comp = torch.zeros(H, W, 3, dtype=torch.float32, device=device)
    remaining = torch.ones(H, W, dtype=torch.float32, device=device)

    # Current material index for each pixel, or -1 for none
    cur_mat = -torch.ones((H, W), dtype=torch.int32, device=device)

    # Accumulated thickness for the current segment
    acc_thick = torch.zeros((H, W), dtype=torch.float32, device=device)

    # Opacity function parameters
    o = 0.10868816
    A = 0.3077416
    k = 76.928215
    b = 2.2291653

    # Main compositing loop: top to bottom
    for layer_idx in range(max_layers - 1, -1, -1):
        # layer_mat is the global material chosen for this layer (int32 scalar).
        layer_mat = new_mats[layer_idx]  # shape []

        # Which pixels actually print on this layer?
        # p_print = (discrete_layers > layer_idx)
        p_print = discrete_layers.gt(layer_idx)  # bool

        # ---------------------------------------------------------------------
        # (A) "Finish" any existing segments that are now 'done'.
        #
        # A segment is done if:
        #   1) cur_mat != -1, i.e. the pixel had an ongoing segment
        #   2) EITHER
        #       - the pixel does not print now (~p_print),
        #       - OR the new layer material differs (cur_mat != layer_mat).
        # ---------------------------------------------------------------------
        mask_done = (cur_mat.ne(-1)) & ((~p_print) | (cur_mat.ne(layer_mat)))

        # Convert to float for multiplications
        mask_done_f = mask_done.to(torch.float32)

        # Gather thickness densities & colors for the old segment
        # We'll clamp cur_mat so -1 becomes 0 (doesn't matter since we multiply by 0).
        old_inds_clamped = torch.clamp(cur_mat, min=0)
        td_vals = material_TDs[old_inds_clamped]  # [H, W]
        col_vals = material_colors[old_inds_clamped]  # [H, W, 3]

        # Compute alpha from accumulated thickness
        thick_ratio = acc_thick / td_vals
        opac_vals = o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio)
        opac_vals = torch.clamp(opac_vals, 0.0, 1.0)  # [H, W]

        # Compositing the old segment:
        #   comp += mask_done_f * remaining * opac_vals * col_vals
        #   remaining *= (1 - mask_done_f * opac_vals) ...
        # but we have to broadcast for color:
        comp_add = (mask_done_f * remaining * opac_vals).unsqueeze(-1) * col_vals
        comp = comp + comp_add
        remaining = remaining - (mask_done_f * remaining * opac_vals)

        # Reset old segment where mask_done is True
        #   cur_mat = -1,  acc_thick = 0
        # We'll do it by `torch.where(mask, val_if_true, val_if_false)`
        cur_mat = torch.where(mask_done, torch.full_like(cur_mat, -1), cur_mat)
        acc_thick = torch.where(mask_done, torch.zeros_like(acc_thick), acc_thick)

        # ---------------------------------------------------------------------
        # (B) For pixels that print this layer:
        #     - Start a new segment if cur_mat == -1
        #     - Accumulate thickness if cur_mat == layer_mat
        # ---------------------------------------------------------------------
        eff_thick = p_print.to(torch.float32) * h

        # (B1) Start new segment where cur_mat == -1
        mask_new = p_print & (cur_mat.eq(-1))
        mask_new_f = mask_new.to(torch.float32)

        # Set cur_mat to layer_mat if mask_new is True
        # (layer_mat is shape [], so it will broadcast)
        cur_mat = torch.where(mask_new, layer_mat, cur_mat)

        # We add thickness:
        acc_thick = acc_thick + mask_new_f * eff_thick

        # (B2) Accumulate thickness where cur_mat == layer_mat
        # We do this in a second mask to avoid confusion, but you can combine.
        mask_same = p_print & (cur_mat.eq(layer_mat))
        acc_thick = acc_thick + (mask_same.to(torch.float32) * eff_thick)

    # -------------------------------------------------------------------------
    # 5) After the loop, composite any remaining segments (cur_mat != -1).
    # -------------------------------------------------------------------------
    mask_remain = cur_mat.ne(-1)
    mask_remain_f = mask_remain.to(torch.float32)

    old_inds_clamped = torch.clamp(cur_mat, min=0)
    td_vals = material_TDs[old_inds_clamped]
    col_vals = material_colors[old_inds_clamped]

    thick_ratio = acc_thick / td_vals
    opac_vals = o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio)
    opac_vals = torch.clamp(opac_vals, 0.0, 1.0)

    comp_add = (mask_remain_f * remaining * opac_vals).unsqueeze(-1) * col_vals
    comp = comp + comp_add
    remaining = remaining - (mask_remain_f * remaining * opac_vals)

    # -------------------------------------------------------------------------
    # 6) Composite background
    # -------------------------------------------------------------------------
    comp = comp + remaining.unsqueeze(-1) * background
    return comp * 255.0


@torch.jit.script
def composite_image_combined(
    pixel_height_logits: torch.Tensor,  # [H,W]
    global_logits: torch.Tensor,  # [max_layers, n_materials]
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,  # [n_materials, 3]
    material_TDs: torch.Tensor,  # [n_materials]
    background: torch.Tensor,  # [3]
    rng_seed: int = -1,
) -> torch.Tensor:
    """
    Combine continuous and discrete compositing over all pixels.

    Args:
        pixel_height_logits (torch.Tensor): Logits for pixel heights, shape [H, W].
        global_logits (torch.Tensor): Logits for global material assignment, shape [max_layers, n_materials].
        tau_height (float): Temperature parameter for height rounding.
        tau_global (float): Temperature parameter for global material assignment.
        h (float): Height of each layer.
        max_layers (int): Maximum number of layers.
        material_colors (torch.Tensor): Tensor of material colors, shape [n_materials, 3].
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters, shape [n_materials].
        background (torch.Tensor): Background color tensor, shape [3].
        rng_seed (int, optional): Random seed for deterministic sampling. Defaults to -1.

    Returns:
        torch.Tensor: Composite image tensor, shape [H, W, 3].
    """
    cont = composite_image_cont(
        pixel_height_logits,
        global_logits,
        tau_height,
        tau_global,
        h,
        max_layers,
        material_colors,
        material_TDs,
        background,
    )
    if tau_global < 1.0:
        disc = composite_image_disc(
            pixel_height_logits,
            global_logits,
            tau_height,
            tau_global,
            h,
            max_layers,
            material_colors,
            material_TDs,
            background,
            rng_seed,
        )
        return cont * tau_global + disc * (1 - tau_global)
    else:
        return cont


def discretize_solution(
    params: dict, tau_global: float, h: float, max_layers: int, rng_seed: int = -1
):
    """
    Convert continuous logs to discrete layer counts and discrete color IDs.

    Args:
        params (dict): Dictionary containing the parameters 'pixel_height_logits' and 'global_logits'.
        tau_global (float): Temperature parameter for global material assignment.
        h (float): Height of each layer.
        max_layers (int): Maximum number of layers.
        rng_seed (int, optional): Random seed for deterministic sampling. Defaults to -1.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Discrete global material assignments, shape [max_layers].
            - torch.Tensor: Discrete height image, shape [H, W].
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
