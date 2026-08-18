import numpy as np
import torch
from torch import nn
from torch.nn.functional import linear


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, w0=1.0, c=6.0, is_first=False, use_bias=True):
        super().__init__()
        self.dim_in = in_dim
        self.is_first = is_first

        weight = torch.zeros(out_dim, in_dim)
        bias = torch.zeros(out_dim) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0)

    def init_(self, weight, bias, c, w0):
        w_std = (1 / self.dim_in) if self.is_first else (np.sqrt(c / self.dim_in) / w0)
        weight.uniform_(-w_std, w_std)
        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        return self.activation(linear(x, self.weight, self.bias))


class SirenModel(nn.Module):
    """SIREN used for NF2's trainable field models."""

    def __init__(
        self,
        in_dim=3,
        out_dim=3,
        dim=256,
        n_layers=8,
        w0=1.0,
        w0_init=5.0,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = n_layers
        self.dim_hidden = dim
        self.in_layer = SirenLayer(
            in_dim=in_dim,
            out_dim=dim,
            w0=w0_init,
            is_first=True,
        )
        self.layers = nn.ModuleList(
            [
                SirenLayer(in_dim=dim, out_dim=dim, w0=w0)
                for _ in range(n_layers - 1)
            ]
        )
        self.out_layer = nn.Linear(dim, out_dim)

    def forward(self, coords):
        x = self.in_layer(coords)
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(x)
