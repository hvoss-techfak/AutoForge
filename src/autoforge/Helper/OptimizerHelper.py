import torch
import torch.nn.functional as F
import math



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
def _sample_layer_probs(
    logits: torch.Tensor,        # [L,M]
    tau: float,
    rng_seed: int = -1,
) -> torch.Tensor:               # returns P  [L,M]
    hard_flag: bool = bool(tau < 1e-3)        # same rule in both paths
    if rng_seed >= 0:
        # deterministic Gumbel
        L, M = logits.shape
        gumbel = torch.empty_like(logits)
        eps: float = 1e-20
        for l in range(L):
            u = deterministic_rand_like(logits[l], rng_seed + l)
            g = -torch.log(-torch.log(u + eps) + eps)
            gumbel[l] = g
        P = F.softmax((logits + gumbel) / tau, dim=1)
        if hard_flag:                        # straight-through one-hot
            idx = torch.argmax(P, dim=1, keepdim=True)
            P_h = torch.zeros_like(P).scatter_(1, idx, 1.0)
            P = (P_h - P).detach() + P
        return P
    else:
        return F.gumbel_softmax(logits, tau, hard=hard_flag, dim=1)

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
    # Instead of torch.rand_like use our deterministic_rand_like.
    U = deterministic_rand_like(logits, rng_seed)
    #compute Gumbel noise.
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    y = (logits + gumbel_noise) / tau
    y_soft = F.softmax(y, dim=-1)
    if hard:
        #compute one-hot using argmax and scatter.
        index = torch.argmax(y_soft, dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        #use the straight-through estimator.
        y = (y_hard - y_soft).detach() + y_soft
    return y



@torch.jit.script
def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:          # c ∈ [0,1]
    return torch.where(c <= 0.04045, c / 12.92,
                       torch.pow((c + 0.055) / 1.055, 2.4))


@torch.jit.script
def _rgb_to_CIELab_L(rgb: torch.Tensor) -> torch.Tensor:       # rgb[...,3]
    """
    Return the CIE-L* channel (D65/sRGB).
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    rl = _srgb_to_linear(r)
    gl = _srgb_to_linear(g)
    bl = _srgb_to_linear(b)

    # sRGB → XYZ
    X = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    Y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    Z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    # Normalise to D65 white
    X = X / 0.95047
    Y = Y / 1.0
    Z = Z / 1.08883

    delta   = 6.0 / 29.0
    delta3  = delta * delta * delta
    inv_den = 1.0 / (3.0 * delta * delta)

    fy = torch.where(
        Y > delta3,
        torch.pow(Y, 1.0 / 3.0),
        Y * inv_den + 4.0 / 29.0,
    )

    L = 116.0 * fy - 16.0
    return L


@torch.jit.script
def _clamp01_scalar(x: float) -> float:
    if x < 0.0:
        return 0.0
    elif x > 1.0:
        return 1.0
    else:
        return x

@torch.jit.script
def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


@torch.jit.script
def composite_image_cont(                      # noqa: C901 – keep it flat
    pixel_height_logits: torch.Tensor,         # [H,W]
    global_logits: torch.Tensor,               # [max_layers,n_materials]
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,             # [n_materials,3]  sRGB 0-1
    material_TDs: torch.Tensor,                # [n_materials]
    background: torch.Tensor,                  # [3]
    rng_seed: int = -1,
) -> torch.Tensor:
    """
    Continuous compositing over all pixels with learnable layer heights.
    Uses Gumbel softmax for global material assignment and a sigmoid-based soft mask
    to determine per-pixel layer contribution. The sigmoid is nearly binary when tau_height > 0.9
    and becomes increasingly soft as tau_height approaches zero, allowing gradients to flow.

    Args:
        pixel_height_logits (torch.Tensor): Logits for pixel heights, shape [H, W].
        global_logits (torch.Tensor): Logits for global material assignment, shape [max_layers, n_materials].
        tau_height (float): Temperature parameter controlling the softness of the layer height.
                              High values yield nearly discrete (binary) behavior.
        tau_global (float): Temperature parameter for global material assignment.
        h (float): Height of each layer.
        max_layers (int): Maximum number of layers.
        material_colors (torch.Tensor): Tensor of material colors, shape [n_materials, 3].
        material_TDs (torch.Tensor): Tensor of material transmission/opacity parameters, shape [n_materials].
        background (torch.Tensor): Background color tensor, shape [3].

    Returns:
        torch.Tensor: Composite image tensor, shape [H, W, 3].
    """

    #layer-wise material choice via Gumbel soft-max
    hard_flag: bool = bool(tau_global < 1e-3)
    P = _sample_layer_probs(global_logits, tau_global, rng_seed)   # [L,M]

    layer_rgb   = torch.matmul(P, material_colors)            # [L,3]
    layer_thick = torch.matmul(P, material_TDs).clamp_(1e-8)  # [L]

    #Per-pixel continuous layer index
    H, W = pixel_height_logits.shape
    device = pixel_height_logits.device

    pixel_height   = (max_layers * h) * torch.sigmoid(pixel_height_logits)
    continuous_idx = pixel_height / h                         # [H,W]

    # Start with background colour. Set alpha too maybe 0.15
    comp_rgb   = background.view(1, 1, 3).expand(H, W, 3).clone()
    comp_alpha = torch.full((H, W), 0.15, dtype=torch.float32, device=device)

    # color cielab tracker. We currently only use lightness as this makes it a bit easier.
    L_mean = _rgb_to_CIELab_L(comp_rgb).mean().item()

    depth_acc: float = h

    const_surface_alpha: float = 0.15

    #front to back
    for l in range(max_layers):
        # keep gradients alive
        scale = tau_height
        p_print = torch.sigmoid((continuous_idx - (l + 0.5)) * scale)   # [H,W]

        #limit absolute coeff. This should probably help with the runaway problem?
        t = layer_thick[l].item()                     # scalar
        abs_coeff = t * 0.1 if t > 0.0 else 0.001

        L_layer = _rgb_to_CIELab_L(
            layer_rgb[l].view(1, 1, 3)
        ).mean().item() * 0.01

        branch_a = (L_mean - L_layer) > 0.0 and (L_layer < 0.5) and (t > 1.0) #quick workaround to limit influence.

        if branch_a:
            scale_d = max(L_mean / (1.0 - (L_mean - L_layer)), 1.0)
            scale_d = math.sqrt(scale_d * abs_coeff)
            atten = depth_acc / scale_d
        else:
            atten = depth_acc / abs_coeff

        atten = _clamp01_scalar(atten)
        mix_coeff = math.sqrt(atten)

        #effective per-pixel mix (scalar × soft mask)
        eff = mix_coeff * p_print                    # [H,W]

        #over blend
        lay_rgb = layer_rgb[l].view(1, 1, 3)
        comp_rgb   = comp_rgb * (1.0 - eff.unsqueeze(-1)) + lay_rgb * eff.unsqueeze(-1)
        comp_alpha = comp_alpha * (1.0 - eff) + const_surface_alpha * eff

        # Update for next layer
        L_mean = _rgb_to_CIELab_L(comp_rgb).mean().item()
        depth_acc += h

    # Return 0-255
    return comp_rgb * 255.0


@torch.jit.script
def composite_image_disc(
    pixel_height_logits: torch.Tensor,       # [H,W]
    global_logits: torch.Tensor,             # [max_layers,n_materials]
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,           # [n_materials,3]  sRGB 0-1
    material_TDs: torch.Tensor,              # [n_materials]
    background: torch.Tensor,                # [3]
    rng_seed: int = -1,
) -> torch.Tensor:

    device = global_logits.device
    probs = _sample_layer_probs(global_logits, tau_global, rng_seed)   # [L,M]
    layer_mat = torch.argmax(probs, dim=1)                            # [L]

    layer_rgb   = material_colors[layer_mat]        # [L,3]
    layer_thick = material_TDs[layer_mat]           # [L]

    pixel_height   = (max_layers * h) * torch.sigmoid(pixel_height_logits)
    continuous_idx = pixel_height / h                                  # [H,W]

    adaptive_layers = adaptive_round(
        continuous_idx, tau_height, high_tau=0.1, low_tau=0.01, temp=0.1)
    dl_tmp   = torch.round(continuous_idx)
    d_layers = (dl_tmp + (adaptive_layers - dl_tmp).detach()).to(torch.int32)  # [H,W]

    H, W = pixel_height_logits.shape
    comp_rgb   = background.view(1, 1, 3).expand(H, W, 3).clone()
    const_alpha: float = 0.15
    L_mean = _rgb_to_CIELab_L(comp_rgb).mean().item()
    depth_acc: float = h

    #same front to black but with hard mask
    for l in range(max_layers):
        #HARDMASK: print if discrete_layers > current layer
        p_print = d_layers.gt(l).to(torch.float32)             # [H,W]
        if not torch.any(p_print):
            depth_acc += h
            continue
        
        t  = float(layer_thick[l].item())
        abs_coeff = t * 0.1 if t > 0.0 else 0.001

        L_layer = _rgb_to_CIELab_L(
            layer_rgb[l].view(1, 1, 3)
        ).mean().item() * 0.01

        branch_a = (L_mean - L_layer) > 0.0 and (L_layer < 0.5) and (t > 1.0)

        if branch_a:
            scale_d = max(L_mean / (1.0 - (L_mean - L_layer)), 1.0)
            scale_d = math.sqrt(scale_d * abs_coeff)
            atten = depth_acc / scale_d
        else:
            atten = depth_acc / abs_coeff

        atten = _clamp01_scalar(atten)
        mix_coeff = math.sqrt(atten)

        eff = mix_coeff * p_print                                 # [H,W]

        lay_rgb = layer_rgb[l].view(1, 1, 3)
        comp_rgb = comp_rgb * (1.0 - eff.unsqueeze(-1)) + lay_rgb * eff.unsqueeze(-1)

        L_mean = _rgb_to_CIELab_L(comp_rgb).mean().item()
        depth_acc += h

    return comp_rgb * 255.0



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
