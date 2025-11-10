import numpy as np
import torch
import argparse

from autoforge.Modules.Optimizer import FilamentOptimizer


class DummyLoss(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)


def _args():
    ns = argparse.Namespace()
    ns.max_layers = 3
    ns.layer_height = 0.2
    ns.learning_rate = 1e-2
    ns.final_tau = 0.5
    ns.init_tau = 0.5
    ns.iterations = 5
    ns.warmup_fraction = 0.2
    ns.learning_rate_warmup_fraction = 0.2
    ns.visualize = False
    ns.tensorboard = False
    ns.run_name = ""
    ns.disable_visualization_for_gradio = 1
    ns.output_folder = "."
    return ns


def _make_optimizer():
    H = W = 4
    target = torch.zeros(H, W, 3, dtype=torch.float32)
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = np.zeros((H, W), dtype=np.int32)
    global_logits_init = np.random.randn(3, 2).astype(np.float32)
    material_colors = torch.rand(2, 3)
    material_TDs = torch.ones(2)
    background = torch.zeros(3)
    device = torch.device("cpu")
    return FilamentOptimizer(
        _args(),
        target,
        pixel_height_logits_init,
        pixel_height_labels,
        global_logits_init,
        material_colors,
        material_TDs,
        background,
        device,
        DummyLoss(),
    )


def test_optimizer_step_and_discretize():
    opt = _make_optimizer()
    loss = opt.step(record_best=False)
    assert isinstance(loss, float)

    disc_global, disc_height = opt.get_discretized_solution(best=False)
    assert disc_global.shape[0] == opt.params["global_logits"].shape[0]
    assert disc_height.shape == (opt.H, opt.W)

    # test get_current_parameters returns clones with expected keys
    params = opt.get_current_parameters()
    assert set(params.keys()) == {
        "pixel_height_logits",
        "global_logits",
        "height_offsets",
    }


def test_optimizer_rng_seed_search():
    opt = _make_optimizer()
    # set up a minimal best_params to avoid None paths
    opt.best_params = opt.get_current_parameters()
    opt.best_seed = 0
    start_loss = 1.0
    best_seed, best_loss = opt.rng_seed_search(
        start_loss, num_seeds=3, autoset_seed=True
    )
    assert isinstance(best_loss, float)
