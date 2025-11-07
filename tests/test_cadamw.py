import torch
import pytest

from autoforge.Helper.CAdamW import CAdamW


def test_cadamw_invalid_params():
    p = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError):
        CAdamW([p], lr=-1e-3)
    with pytest.raises(ValueError):
        CAdamW([p], betas=(-0.1, 0.999))
    with pytest.raises(ValueError):
        CAdamW([p], betas=(0.9, 1.5))
    with pytest.raises(ValueError):
        CAdamW([p], eps=-1e-6)


def test_cadamw_step_updates_and_weight_decay():
    # Simple linear model
    model = torch.nn.Linear(4, 2)
    opt = CAdamW(model.parameters(), lr=1e-2, weight_decay=0.1, correct_bias=True)
    x = torch.randn(3, 4)
    y = torch.randn(3, 2)
    criterion = torch.nn.MSELoss()
    initial_weights = [p.clone() for p in model.parameters()]
    loss = criterion(model(x), y)
    loss.backward()
    opt.step()
    # Ensure parameters changed
    for init, p in zip(initial_weights, model.parameters()):
        assert not torch.equal(init, p)
    # Run a second step without closure to cover state reuse
    loss2 = criterion(model(x), y)
    loss2.backward()
    opt.step()
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_cadamw_no_bias_correction():
    model = torch.nn.Linear(2, 1)
    opt = CAdamW(model.parameters(), lr=1e-2, correct_bias=False)
    x = torch.randn(5, 2)
    y = torch.randn(5, 1)
    loss = (model(x) - y).pow(2).mean()
    loss.backward()
    opt.step()
    for p in model.parameters():
        assert torch.isfinite(p).all()
