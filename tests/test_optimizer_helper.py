import torch
from autoforge.Helper.OptimizerHelper import (
    adaptive_round,
    deterministic_rand_like,
    deterministic_gumbel_softmax,
    bleed_layer_effect,
    composite_image_cont,
    composite_image_disc,
)


def test_adaptive_round_limits():
    x = torch.tensor([0.2, 0.7, 1.4, 2.6])
    hard = adaptive_round(x, tau=0.0, high_tau=1.0, low_tau=0.0, temp=0.1)
    soft = adaptive_round(x, tau=1.0, high_tau=1.0, low_tau=0.0, temp=0.1)
    assert torch.all(hard == torch.round(x))
    assert torch.all(soft >= torch.floor(x)) and torch.all(soft <= torch.ceil(x))


def test_deterministic_rand_like():
    t = torch.zeros(5)
    r1 = deterministic_rand_like(t, 123)
    r2 = deterministic_rand_like(t, 123)
    r3 = deterministic_rand_like(t, 124)
    assert torch.allclose(r1, r2)
    assert not torch.allclose(r1, r3)


def test_deterministic_gumbel_softmax_repro():
    logits = torch.tensor([0.1, 2.0, 0.3])
    y1 = deterministic_gumbel_softmax(logits, tau=1.0, hard=False, rng_seed=42)
    y2 = deterministic_gumbel_softmax(logits, tau=1.0, hard=False, rng_seed=42)
    assert torch.allclose(y1, y2)


def _simple_composite_inputs(H=8, W=8, L=4, M=3):
    pixel_logits = torch.zeros(H, W)
    global_logits = torch.zeros(L, M)
    material_colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    material_TDs = torch.tensor([0.5, 0.7, 0.9])
    background = torch.tensor([1.0, 1.0, 1.0])
    return pixel_logits, global_logits, material_colors, material_TDs, background


def test_composite_image_cont_shapes():
    pl, gl, mc, td, bg = _simple_composite_inputs()
    out = composite_image_cont(pl, gl, 0.5, 0.5, 0.2, gl.shape[0], mc, td, bg)
    assert out.shape == (pl.shape[0], pl.shape[1], 3)
    assert torch.isfinite(out).all()


def test_composite_image_disc_shapes():
    pl, gl, mc, td, bg = _simple_composite_inputs()
    out = composite_image_disc(
        pl, gl, 0.5, 0.5, 0.2, gl.shape[0], mc, td, bg, rng_seed=0
    )
    assert out.shape == (pl.shape[0], pl.shape[1], 3)


def test_bleed_layer_effect_monotonic():
    mask = torch.zeros(2, 4, 4)
    mask[0, 2, 2] = 1.0
    out = bleed_layer_effect(mask, strength=0.5)
    assert out.shape == mask.shape
    assert out[0, 2, 2] >= mask[0, 2, 2]
