import numpy as np
import torch
import pytest

from autoforge.Modules.Optimizer import FilamentOptimizer
from autoforge.Loss.PerceptionLoss import MultiLayerVGGPerceptualLoss


class DummyArgs:
    def __init__(self):
        self.max_layers = 6
        self.layer_height = 0.2
        self.learning_rate = 0.01
        self.final_tau = 0.05
        self.init_tau = 0.8
        self.iterations = 15
        self.warmup_fraction = 0.2
        self.learning_rate_warmup_fraction = 0.2
        self.tensorboard = False
        self.run_name = ""
        self.visualize = False
        self.disable_visualization_for_gradio = 1
        self.output_folder = "."
        self.background_height = 0.6


@pytest.fixture(autouse=True)
def mock_vgg16_e2e(monkeypatch):
    import torchvision.models as models

    class DummyFeatures(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.Conv2d(3, 3, 3, padding=1),
                    torch.nn.ReLU(),
                ]
            )

        def __getitem__(self, idx):
            return self.layers[idx]

        def __iter__(self):
            return iter(self.layers)

        def __len__(self):
            return len(self.layers)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = DummyFeatures()

    def dummy_vgg16(weights=None):
        return DummyModel()

    monkeypatch.setattr(models, "vgg16", dummy_vgg16)
    yield


@pytest.fixture
def dummy_optimizer_setup():
    torch.manual_seed(0)
    np.random.seed(0)
    H, W = 16, 16
    target = torch.randint(0, 256, (H, W, 3), dtype=torch.float32)
    pixel_height_logits_init = np.zeros((H, W), dtype=np.float32)
    pixel_height_labels = np.zeros((H, W), dtype=np.int32)
    global_logits_init = None
    material_colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    material_TDs = torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32)
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    args = DummyArgs()
    device = torch.device("cpu")
    percept = MultiLayerVGGPerceptualLoss(
        layers=[0]
    )  # mocked by earlier fixture if any
    opt = FilamentOptimizer(
        args,
        target,
        pixel_height_logits_init,
        pixel_height_labels,
        global_logits_init,
        material_colors,
        material_TDs,
        background,
        device,
        percept,
    )
    # fabricate a best_params snapshot so pruning helpers relying on it do not fail
    opt.best_params = opt.get_current_parameters()
    opt.best_seed = 0
    return opt


def test_optimizer_runs_steps(dummy_optimizer_setup):
    opt = dummy_optimizer_setup
    losses = []
    for i in range(5):
        l = opt.step(record_best=True)
        losses.append(l)
    # ensure finite
    assert all(np.isfinite(losses))
    # ensure best params stored
    assert opt.best_params is not None
    disc_global, disc_height = opt.get_discretized_solution(best=True)
    assert disc_global is not None
    assert disc_height.shape == (opt.H, opt.W)


def test_rng_seed_search(dummy_optimizer_setup):
    opt = dummy_optimizer_setup
    opt.step(record_best=True)
    start_loss = opt.best_discrete_loss
    seed, new_loss = opt.rng_seed_search(start_loss, num_seeds=3, autoset_seed=True)
    assert new_loss <= start_loss * 1.05  # allow small drift
