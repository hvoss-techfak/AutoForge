import os
import torch
import pytest

from autoforge.Helper.AmpUtils import safe_autocast, get_selected_autocast


def _device_from_env():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_safe_autocast_backward_does_not_crash():
    # Ensure automatic mode
    os.environ.pop("AUTOFORGE_AMP", None)
    device = _device_from_env()

    x = torch.randn(16, 16, device=device, dtype=torch.float32)
    w = torch.randn(16, 16, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randn(16, 16, device=device, dtype=torch.float32)

    dtype, _reason = get_selected_autocast(device)
    # Should not raise regardless of dtype selection
    with safe_autocast(device):
        y = x @ w
        loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    # If CUDA fp16 was selected, ensure grads exist and are finite
    if device.type == "cuda" and dtype == torch.float16:
        assert w.grad is not None
        assert torch.isfinite(w.grad).all()


@pytest.mark.parametrize("force", ["off", "bf16", "fp16"])  # exercise overrides
def test_env_override_modes(monkeypatch, force):
    monkeypatch.setenv("AUTOFORGE_AMP", force)
    device = _device_from_env()

    x = torch.randn(8, 8, device=device, dtype=torch.float32)
    w = torch.randn(8, 8, device=device, dtype=torch.float32, requires_grad=True)
    target = torch.randn(8, 8, device=device, dtype=torch.float32)

    with safe_autocast(device):
        y = x @ w
        loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    assert w.grad is not None

