import torch
from matplotlib import pyplot as plt
from torch import nn
from util import PINN, derivative, plot_progression_1d


def test_pinn_uses_tanh_by_default():
    model = PINN([1, 4, 1])

    assert any(isinstance(module, nn.Tanh) for module in model.net)


def test_pinn_accepts_relu_activation():
    model = PINN([1, 4, 1], activation=nn.ReLU)

    assert any(isinstance(module, nn.ReLU) for module in model.net)
    assert not any(isinstance(module, nn.Tanh) for module in model.net)


def test_pinn_scales_physical_input_without_breaking_derivatives():
    model = PINN([1, 1], input_bounds=[(2.0, 6.0)])
    with torch.no_grad():
        model.net[0].weight.fill_(1.0)
        model.net[0].bias.zero_()

    x = torch.tensor([[2.0], [4.0], [6.0]], requires_grad=True)
    output = model(x)

    assert torch.allclose(output, torch.tensor([[-1.0], [0.0], [1.0]]))
    assert torch.allclose(derivative(output, x), torch.full_like(x, 0.5))


def test_progression_curves_shift_hue_and_emphasise_the_final_epoch(monkeypatch):
    monkeypatch.setattr(plt.Figure, "savefig", lambda *args, **kwargs: None)
    x = torch.linspace(0.0, 1.0, 5).reshape(-1, 1)
    snapshots = {0: x, 10: 2.0 * x, 20: 3.0 * x}

    plot_progression_1d(x, snapshots, x, "progression", "unused.png")

    axes = plt.gcf().axes
    prediction_lines = [ax.lines[0] for ax in axes]
    exact_lines = [ax.lines[1] for ax in axes]
    assert [ax.get_title() for ax in axes] == ["epoch 0", "epoch 10", "epoch 20"]
    assert tuple(prediction_lines[0].get_color()) != tuple(
        prediction_lines[1].get_color()
    )
    assert prediction_lines[-1].get_color() == "#D55E00"
    assert [line.get_linewidth() for line in prediction_lines] == [1.2, 1.2, 1.6]
    assert all(line.get_linewidth() == 1.0 for line in exact_lines)
    assert tuple(plt.gcf().get_size_inches()) == (6.7, 2.1)
    plt.close()
