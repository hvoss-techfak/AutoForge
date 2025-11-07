import types
import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from autoforge.Modules.Optimizer import FilamentOptimizer


def make_args():
    # Minimal args needed by optimizer
    return types.SimpleNamespace(
        iterations=2,
        warmup_fraction=0.0,
        learning_rate_warmup_fraction=0.0,
        init_tau=1.0,
        final_tau=0.5,
        learning_rate=0.01,
        layer_height=0.04,
        max_layers=4,
        visualize=False,
        tensorboard=False,
        disable_visualization_for_gradio=1,
        output_folder="/tmp",
    )


def test_optimizer_one_step_cpu():
    device = torch.device("cpu")
    H, W = 16, 16
    target = torch.randint(0, 255, (H, W, 3), dtype=torch.uint8).to(torch.float32)
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = np.zeros((H, W), dtype=np.int32)
    global_logits_init = np.zeros((4, 3), dtype=np.float32)
    material_colors = torch.rand(3, 3)
    material_TDs = torch.ones(3)
    background = torch.tensor([0.0, 0.0, 0.0])
    optimizer = FilamentOptimizer(
        args=make_args(),
        target=target,
        pixel_height_logits_init=pixel_height_logits_init,
        pixel_height_labels=pixel_height_labels,
        global_logits_init=global_logits_init,
        material_colors=material_colors,
        material_TDs=material_TDs,
        background=background,
        device=device,
        perception_loss_module=None,
    )
    loss = optimizer.step(record_best=True)
    assert isinstance(loss, float)
    dg, dh = optimizer.get_discretized_solution()
    assert dg.shape[0] == optimizer.params["global_logits"].shape[0]
    assert dh.shape == pixel_height_logits_init.shape
