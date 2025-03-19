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
    perception_loss_module: torch.nn.Module,
    add_penalty_loss: float = 0.0,
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
    )


def compute_loss(
    comp: torch.Tensor,
    target: torch.Tensor,
    pixel_height_logits: torch.Tensor = None,
    tau_height: float = 1.0,
    add_penalty_loss: float = 0.0,
) -> torch.Tensor:
    """
    Combined MSE + Perceptual + penalty losses with patch-based smoothness.
    """
    # MSE Loss
    comp_mse = srgb_to_lab(comp)
    target_mse = srgb_to_lab(target)
    mse_loss = F.huber_loss(comp_mse, target_mse)

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
        smoothness_loss = (loss_dx + loss_dy) * (10 * add_penalty_loss)

        # Additional patch-based smoothness loss (using a 3x3 Laplacian):
        laplacian_kernel = (
            torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                dtype=pixel_height_logits.dtype,
                device=pixel_height_logits.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        height_map = pixel_height_logits.unsqueeze(0).unsqueeze(0)
        laplacian_output = F.conv2d(height_map, laplacian_kernel, padding=1)
        patch_smooth_loss = (
            F.huber_loss(laplacian_output, torch.zeros_like(laplacian_output)) * 10
        )
        total_loss = mse_loss + smoothness_loss + add_penalty_loss * patch_smooth_loss
    else:
        total_loss = mse_loss
    return total_loss
