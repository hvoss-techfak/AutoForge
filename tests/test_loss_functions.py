import torch
from autoforge.Loss.LossFunctions import loss_fn, compute_loss


def test_compute_loss_basic():
    target = torch.randint(0, 256, (16, 16, 3), dtype=torch.float32)
    comp = target.clone()
    l = compute_loss(comp=comp, target=target)
    assert torch.isfinite(l)
    assert l.item() < 1e-6  # identical -> near zero


def test_loss_fn_pipeline(material_data_tensors):
    material_colors, material_TDs, background = material_data_tensors
    target = torch.randint(0, 256, (16, 16, 3), dtype=torch.float32)
    pixel_height_logits = torch.zeros(16, 16)
    global_logits = torch.zeros(8, material_colors.shape[0])
    params = {
        "pixel_height_logits": pixel_height_logits,
        "global_logits": global_logits,
    }
    out = loss_fn(
        params,
        target,
        tau_height=0.5,
        tau_global=0.5,
        h=0.2,
        max_layers=8,
        material_colors=material_colors,
        material_TDs=material_TDs,
        background=background,
    )
    assert torch.isfinite(out)
