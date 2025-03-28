import torch
import torch.nn.functional as F

from autoforge.Helper.ImageHelper import srgb_to_lab
from autoforge.Helper.OptimizerHelper import composite_image_cont


def loss_fn(
    params: dict,
    target: torch.Tensor,
    tau_height: float,
    tau_global: float,
    h: float,
    max_layers: int,
    material_colors: torch.Tensor,
    material_TDs: torch.Tensor,
    background: torch.Tensor,
    add_penalty_loss: float = 0.0,
    focus_map: torch.Tensor = None,
    focus_strength: float = 10.0,
) -> torch.Tensor:
    """
    Full forward pass for continuous assignment:
    composite, then compute unified loss on (global_logits).
    """
    comp = composite_image_cont(
        params["pixel_height_logits"],
        params["global_logits"],
        tau_height,
        tau_global,
        h,
        max_layers,
        material_colors,
        material_TDs,
        background,
    )
    return compute_loss(
        comp=comp,
        target=target,
        pixel_height_logits=params["pixel_height_logits"],
        tau_height=tau_height,
        add_penalty_loss=add_penalty_loss,
        focus_map=focus_map,
        focus_strength=focus_strength,
    )


def compute_loss(
    comp: torch.Tensor,
    target: torch.Tensor,
    pixel_height_logits: torch.Tensor = None,
    tau_height: float = 1.0,
    add_penalty_loss: float = 0.0,
    focus_map: torch.Tensor = None,
    focus_strength: float = 10.0,
) -> torch.Tensor:
    comp_lab = srgb_to_lab(comp)
    target_lab = srgb_to_lab(target)

    mse_loss = F.huber_loss(comp_lab, target_lab, delta=1.0)

    if pixel_height_logits is not None:
        # Existing neighbor-based smoothness loss:
        target_gray = target.mean(dim=2)  # shape becomes [H, W]
        weight_x = torch.exp(-torch.abs(target_gray[:, 1:] - target_gray[:, :-1]))
        weight_y = torch.exp(-torch.abs(target_gray[1:, :] - target_gray[:-1, :]))
        weight_x = torch.clamp(weight_x, 0.5, 1.0)
        weight_y = torch.clamp(weight_y, 0.5, 1.0)
        dx = torch.abs(pixel_height_logits[:, 1:] - pixel_height_logits[:, :-1])
        dy = torch.abs(pixel_height_logits[1:, :] - pixel_height_logits[:-1, :])
        loss_dx = torch.mean(F.huber_loss(dx * weight_x, torch.zeros_like(dx)))
        loss_dy = torch.mean(F.huber_loss(dy * weight_y, torch.zeros_like(dy)))
        smoothness_loss = (loss_dx + loss_dy) * add_penalty_loss

        # Additional patch-based smoothness loss (using a 3x3 Laplacian):
        # laplacian_kernel = (
        #     torch.tensor(
        #         [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        #         dtype=pixel_height_logits.dtype,
        #         device=pixel_height_logits.device,
        #     )
        #     .unsqueeze(0)
        #     .unsqueeze(0)
        # )
        # height_map = pixel_height_logits.unsqueeze(0).unsqueeze(0)
        # laplacian_output = F.conv2d(height_map, laplacian_kernel, padding=1)
        # patch_smooth_loss = F.huber_loss(
        #     laplacian_output, torch.zeros_like(laplacian_output)
        # )
        total_loss = mse_loss + smoothness_loss
    else:
        # + add_penalty_loss * patch_smooth_loss
        total_loss = mse_loss

    return total_loss

    # if focus_map is not None and focus_strength > 0.0:
    #     # Expand focus_map to [H, W, 1] to match the color channels.
    #     focus_map_exp = focus_map.unsqueeze(-1)
    #     base_loss = F.huber_loss(comp_mse, target_mse, reduction="none")
    #     # we need to sum up to a maximum of one, so normalize based on strength
    #     min_strength = 1 / focus_strength
    #     max_strength = 1 - min_strength
    #     # focus map is normalized between 0 and 1
    #     normalized_focus = min_strength + focus_map_exp * max_strength
    #     weighted_loss = base_loss * normalized_focus
    #
    #     mse_loss = weighted_loss.mean()
    # else:
    #

    # loss_color = global_color_loss(comp_lab, target_lab)

    # if pixel_height_logits is not None:
    #     # Existing neighbor-based smoothness loss:
    #     target_gray = target.mean(dim=2)  # shape becomes [H, W]
    #     weight_x = torch.exp(-torch.abs(target_gray[:, 1:] - target_gray[:, :-1]))
    #     weight_y = torch.exp(-torch.abs(target_gray[1:, :] - target_gray[:-1, :]))
    #     weight_x = torch.clamp(weight_x, 0.5, 1.0)
    #     weight_y = torch.clamp(weight_y, 0.5, 1.0)
    #     dx = torch.abs(pixel_height_logits[:, 1:] - pixel_height_logits[:, :-1])
    #     dy = torch.abs(pixel_height_logits[1:, :] - pixel_height_logits[:-1, :])
    #     loss_dx = torch.mean(F.huber_loss(dx * weight_x, torch.zeros_like(dx)))
    #     loss_dy = torch.mean(F.huber_loss(dy * weight_y, torch.zeros_like(dy)))
    #     smoothness_loss = (loss_dx + loss_dy) * add_penalty_loss
    #
    #     # Additional patch-based smoothness loss (using a 3x3 Laplacian):
    #     laplacian_kernel = (
    #         torch.tensor(
    #             [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    #             dtype=pixel_height_logits.dtype,
    #             device=pixel_height_logits.device,
    #         )
    #         .unsqueeze(0)
    #         .unsqueeze(0)
    #     )
    #     height_map = pixel_height_logits.unsqueeze(0).unsqueeze(0)
    #     laplacian_output = F.conv2d(height_map, laplacian_kernel, padding=1)
    #     patch_smooth_loss = F.huber_loss(
    #         laplacian_output, torch.zeros_like(laplacian_output)
    #     )
    #     total_loss = mse_loss + smoothness_loss + add_penalty_loss * patch_smooth_loss
    # else:
