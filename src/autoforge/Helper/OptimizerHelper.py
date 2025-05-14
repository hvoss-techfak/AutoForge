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
    pixel_height_logits: torch.Tensor,  # [H,W]
    global_logits: torch.Tensor,  # [L,M]
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,  # [M,3]
    material_TDs: torch.Tensor,  # [M]
    background: torch.Tensor,  # [3]
) -> torch.Tensor:
    # 1. per-pixel continuous layer index
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logits)  # [H,W]
    continuous_z = pixel_height / h  # [H,W]

    # 2. global material weights with Gumbel-Softmax
    hard_flag = tau_global < 1e-3
    p_mat = F.gumbel_softmax(global_logits, tau_global, hard=hard_flag, dim=1)  # [L,M]

    layer_colors = p_mat @ material_colors  # [L,3]
    layer_TDs = (p_mat @ material_TDs).clamp(1e-8, 1e8)  # [L]

    # 3. soft print mask for all layers (layer 0 = bottom, layer L-1 = top)
    scale = 10.0 * tau_height
    layer_idx = torch.arange(
        max_layers, dtype=torch.float32, device=pixel_height.device
    ).view(-1, 1, 1)  # [L,1,1]
    p_print = torch.sigmoid(
        (continuous_z.unsqueeze(0) - (layer_idx + 0.5)) * scale
    )  # [L,H,W]

    # 4. thickness and opacity
    eff_thick = p_print * h  # [L,H,W]
    thick_ratio = eff_thick / layer_TDs.view(-1, 1, 1)  # [L,H,W]

    o, A, k, b = -1.2416557e-02, 9.6407950e-01, 3.4103447e01, -4.1554203e00
    opac = o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio)
    opac = torch.clamp(opac, 0.0, 1.0)  # [L,H,W]

    # 5. flip to top→bottom order before compositing
    opac_fb = torch.flip(opac, dims=[0])  # [L,H,W]
    colors_fb = torch.flip(layer_colors, dims=[0])  # [L,3]

    trans_fb = 1.0 - opac_fb  # [L,H,W]
    trans_shift = torch.cat([torch.ones_like(trans_fb[:1]), trans_fb[:-1]], dim=0)
    remain_fb = torch.cumprod(trans_shift, dim=0)  # remaining before each layer [L,H,W]

    comp_layers = (remain_fb * opac_fb).unsqueeze(-1) * colors_fb.view(
        -1, 1, 1, 3
    )  # [L,H,W,3]
    comp = comp_layers.sum(dim=0)  # [H,W,3]

    # 6. background
    rem_after = remain_fb[-1] * trans_fb[-1]  # remaining after bottom layer
    comp = comp + rem_after.unsqueeze(-1) * background  # [H,W,3]
    return comp * 255.0


@torch.jit.script
def _runs_from_materials(mats: torch.Tensor):
    """
    Given a 1D int tensor of per-layer materials (top to bottom),
    return the start indices, end indices (exclusive) and material id
    for each run of equal values.

    Returns:
        run_starts  [R] int64
        run_ends    [R] int64
        run_mats    [R] same dtype as mats
    """
    L = int(mats.shape[0])
    if L == 0:
        empty_i = torch.empty(0, dtype=torch.int64, device=mats.device)
        return empty_i, empty_i, torch.empty(0, dtype=mats.dtype, device=mats.device)

    change = torch.ones(L, dtype=torch.bool, device=mats.device)
    change[1:] = mats[1:] != mats[:-1]

    # TorchScript friendly: no keyword args
    run_starts = torch.nonzero(change).squeeze(1).to(torch.int64)  # [R]
    run_ends = torch.cat([run_starts[1:], torch.tensor([L], device=mats.device)])
    run_mats = mats[run_starts]  # [R]

    return run_starts, run_ends, run_mats


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
    # 1. discrete layer counts per pixel (with adaptive rounding)
    pixel_height = (max_layers * h) * torch.sigmoid(pixel_height_logits)
    continuous_z = pixel_height / h
    adaptive_layers = adaptive_round(
        continuous_z, tau_height, high_tau=0.1, low_tau=0.01, temp=0.1
    )
    discrete_temp = torch.round(continuous_z)
    discrete_layers = (discrete_temp + (adaptive_layers - discrete_temp).detach()).to(
        torch.int32
    )  # [H,W]

    # 2. one material per physical layer
    if rng_seed >= 0:
        mats = []
        for i in range(max_layers):
            p_i = deterministic_gumbel_softmax(
                global_logits[i], tau_global, hard=True, rng_seed=rng_seed + i
            )
            mats.append(torch.argmax(p_i).to(torch.int32))
        new_mats = torch.stack(mats, dim=0)  # [L]
    else:
        probs = F.gumbel_softmax(global_logits, tau_global, hard=True, dim=1)  # [L,M]
        new_mats = torch.argmax(probs, dim=1).to(torch.int32)  # [L]

    # 3. flip to top→bottom order and collect material runs
    new_mats_top = torch.flip(new_mats, dims=[0])  # index 0 = top
    run_starts, run_ends, run_m = _runs_from_materials(new_mats_top)  # R runs

    R = int(run_starts.shape[0])
    H, W = pixel_height.shape
    device = pixel_height.device
    dtype_f = torch.float32

    # 4. printed layers per run (vectorised)  –  same logic as the loop
    #    top 'D_no_print' layers stay empty
    D_no_print = (max_layers - discrete_layers).to(torch.int32)  # [H,W]
    run_s = run_starts.view(R, 1, 1)
    run_e = run_ends.view(R, 1, 1)
    printed_L = torch.clamp(
        run_e - torch.maximum(run_s, D_no_print.unsqueeze(0)), min=0
    )  # [R,H,W]

    # *** FIX ***
    # add one extra layer thickness where at least one layer is printed
    extra_layer = (printed_L > 0).to(dtype_f)  # [R,H,W]
    thickness_r = (printed_L.to(dtype_f) + extra_layer) * h  # [R,H,W]

    # 5. opacity for every run
    TD_r = material_TDs[run_m].view(R, 1, 1)  # [R,1,1]
    thick_ratio = thickness_r / TD_r  # [R,H,W]

    o, A, k, b = 0.10868816, 0.3077416, 76.928215, 2.2291653
    opac_r = o + (A * torch.log1p(k * thick_ratio) + b * thick_ratio)
    opac_r = torch.clamp(opac_r, 0.0, 1.0) * (printed_L > 0).to(
        dtype_f
    )  # zero when nothing printed

    # 6. front-to-back compositing run by run
    trans_r = 1.0 - opac_r
    trans_shift = torch.cat([torch.ones_like(trans_r[:1]), trans_r[:-1]], dim=0)
    remain_r = torch.cumprod(trans_shift, dim=0)  # remaining light before run

    colors_r = material_colors[run_m].view(R, 1, 1, 3)  # [R,1,1,3]
    comp_runs = (remain_r * opac_r).unsqueeze(-1) * colors_r  # [R,H,W,3]
    comp = comp_runs.sum(dim=0)  # [H,W,3]

    # 7. background
    rem_after = remain_r[-1] * trans_r[-1]
    comp = comp + rem_after.unsqueeze(-1) * background
    return comp * 255.0


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
