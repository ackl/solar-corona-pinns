import torch
from torch import nn
from util import PINN, derivative


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
