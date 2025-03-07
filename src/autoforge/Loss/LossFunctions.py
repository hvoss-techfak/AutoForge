import torch
import torch.nn.functional as F

from autoforge.Helper.ImageHelper import srgb_to_lab, increase_saturation
from autoforge.Helper.OptimizerHelper import composite_image


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
    add_penalty_loss: bool = True,
) -> torch.Tensor:
    """
    Full forward pass for continuous assignment:
    composite, then compute unified loss on (global_logits).
    """
    comp = composite_image(
        params["pixel_height_logits"],
        params["global_logits"],
        tau_height,
        tau_global,
        h,
        max_layers,
        material_colors,
        material_TDs,
        background,
        mode="continuous",
    )
    return compute_loss(
        material_assignment=params["global_logits"],
        comp=comp,
        target=target,
        perception_loss_module=perception_loss_module,
        tau_global=tau_global,
        num_materials=material_colors.shape[0],
        add_penalty_loss=add_penalty_loss,
    )


def compute_loss(
    material_assignment: torch.Tensor,
    comp: torch.Tensor,
    target: torch.Tensor,
    perception_loss_module: torch.nn.Module,
    tau_global: float,
    num_materials: int,
    add_penalty_loss: bool = True,
) -> torch.Tensor:
    """
    Combined MSE + Perceptual + penalty losses.
    """
    # MSE
    comp_mse = srgb_to_lab(comp)
    # we slightly increase saturation of our target image to make the color matching more robust
    target = increase_saturation(target, 0.1)
    target_mse = srgb_to_lab(target)
    mse_loss = F.huber_loss(comp_mse, target_mse)
    # Perceptual Loss
    # comp_batch = comp.permute(2, 0, 1).unsqueeze(0)
    # target_batch = target.permute(2, 0, 1).unsqueeze(0)
    # perception_loss = perception_loss_module(comp_batch, target_batch)

    # Basic penalty
    # if material_assignment.dim() == 2:
    #     # Continuous assignment => shape (max_layers, num_materials)
    #     p = F.softmax(material_assignment, dim=1)
    #     dot_products = torch.sum(p[:-1] * p[1:], dim=1)  # shape (max_layers-1,)
    #     color_change_penalty = torch.mean(1.0 - dot_products)
    #
    #     color_usage = torch.mean(p, dim=0)
    #     few_colors_penalty = torch.sum(torch.sqrt(1e-8 + color_usage))
    # else:
    #     # Discrete assignment => shape (max_layers,)
    #     disc = material_assignment
    #     same_color = (disc[:-1] == disc[1:]).float()
    #     dot_products = same_color
    #     color_change_penalty = (
    #         torch.mean(1.0 - dot_products) if add_penalty_loss else 0.0
    #     )
    #
    #     max_layers = disc.shape[0]
    #     usage_counts = torch.bincount(disc, minlength=num_materials).float()
    #     color_usage = usage_counts / float(max_layers)
    #     few_colors_penalty = (
    #         torch.sum(torch.sqrt(1e-8 + color_usage)) if add_penalty_loss else 0.0
    #     )
    #
    # # Weighted sum
    # lambda_swap = (1.0 - tau_global) * 0.1
    # total_loss = (
    #     mse_loss + lambda_swap * color_change_penalty + lambda_swap * few_colors_penalty
    # )

    # combine with the perceptual loss
    total_loss = mse_loss  # + perception_loss * 10.0
    return total_loss
