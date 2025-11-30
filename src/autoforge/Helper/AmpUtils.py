from __future__ import annotations

from contextlib import nullcontext, contextmanager
from typing import Optional, Tuple, Dict

import os
import torch

# Cache the selection per device key so CPU/CUDA/MPS don't interfere
_SELECTED_MAP: Dict[str, Tuple[Optional[torch.dtype], str]] = {}


def _device_key(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{idx}"
    return device.type


def _probe_backward(device: torch.device, dtype: torch.dtype) -> bool:
    """Try a tiny forward+backward under autocast for given dtype on the device.

    Returns True if successful, False if a RuntimeError occurs (e.g., oneDNN bf16/f16 backward not supported).
    """
    try:
        if device.type == "mps":
            return False  # no stable autocast on MPS yet
        x = torch.randn(8, 8, device=device, dtype=torch.float32, requires_grad=False)
        w = torch.randn(8, 8, device=device, dtype=torch.float32, requires_grad=True)
        target = torch.randn(8, 8, device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = x @ w
            loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
        # Step with a dummy optimizer to ensure gradients are usable
        opt = torch.optim.SGD([w], lr=1e-3)
        opt.step()
        return True
    except Exception:
        return False


def _select_dtype_for_device(device: torch.device) -> Tuple[Optional[torch.dtype], str]:
    """Decide the best autocast dtype for the current device, probing backward support.

    Returns (dtype or None, reason string).
    None means: do not use autocast; stick to full float32.
    """
    # Allow override via env for debugging
    mode = os.getenv("AUTOFORGE_AMP", default="auto").lower()

    if mode in {"off", "fp32", "float32"}:
        print("AUTOFORGE_AMP=off: disabling automatic mixed precision")
        return None, "AUTOFORGE_AMP=off"
    if mode in {"bf16", "bfloat16"}:
        # Only honor if probe succeeds
        if _probe_backward(device, torch.bfloat16):
            return torch.bfloat16, "forced bf16"
        print("AUTOFORGE_AMP=bf16: disabling bf16 as probe failed")
        return None, "forced bf16 but probe failed"
    if mode in {"fp16", "float16", "half"}:
        if device.type == "cuda" and _probe_backward(device, torch.float16):
            return torch.float16, "forced f16"
        print("AUTOFORGE_AMP=fp16: disabling fp16 as probe failed or non-CUDA")
        return None, "forced f16 but probe failed or non-CUDA"

    # Automatic selection
    if device.type == "cuda":
        # Prefer bf16 if it actually works end-to-end
        if torch.cuda.is_bf16_supported() and _probe_backward(device, torch.bfloat16):
            print("Using bf16 autocast on CUDA device")
            return torch.bfloat16, "cuda bf16"
        # Next try fp16 if the GPU supports it
        major, minor = torch.cuda.get_device_capability(device)
        if (major * 10 + minor) >= 53 and _probe_backward(device, torch.float16):
            print("Using fp16 autocast on CUDA device")
            return torch.float16, "cuda f16"
        # Fall back to fp32; enable TF32 to speed up matmuls if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Falling back to fp32 autocast on CUDA device with TF32 enabled")
        return None, "cuda fp32 (tf32 on)"

    if device.type == "cpu":
        # Some CPUs (AVX512) can do bf16 forward/backward with oneDNN, but many cannot.
        # Probe and use bf16 only if backward succeeds; otherwise stay fp32.
        if _probe_backward(device, torch.bfloat16):
            print("Using bf16 autocast on CPU device")
            return torch.bfloat16, "cpu bf16"
        # No reliable fp16 on CPU for backward; use fp32
        print("Falling back to fp32 autocast on CPU device")
        return None, "cpu fp32"

    if device.type == "mps":
        print("MPS device detected: autocast not reliably supported, using fp32")
        return None, "mps fp32"

    # Default: no autocast
    print(f"Unknown device type '{device.type}': disabling autocast, using fp32")
    return None, f"{device.type} fp32"


def get_selected_autocast(device: torch.device) -> Tuple[Optional[torch.dtype], str]:
    key = _device_key(device)
    if key not in _SELECTED_MAP:
        _SELECTED_MAP[key] = _select_dtype_for_device(device)
    return _SELECTED_MAP[key]


@contextmanager
def safe_autocast(device: torch.device):
    """Context manager that uses autocast only if the runtime probe succeeded.

    Falls back to a no-op context on unsupported platforms to avoid runtime errors like:
    RuntimeError: DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2.
    """
    dtype, _reason = get_selected_autocast(device)
    if dtype is None:
        with nullcontext():
            yield
    else:
        with torch.autocast(device_type=device.type, dtype=dtype):
            yield
