import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import scienceplots
from pathlib import Path

plt.style.use(["science", "muted", "grid"])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PINN(nn.Module):
    """Multi-layer perceptron with selectable activation and input scaling.

    The activation defaults to ``nn.Tanh`` to preserve the existing studies.
    `layers` is a list describing number of neurons in each layer, e.g. [1, 32, 32, 1]
    `input_bounds` optionally maps each physical coordinate range to [-1, 1].
    """

    def __init__(self, layers, activation=nn.Tanh, input_bounds=None):
        super().__init__()

        if input_bounds is None:
            self.register_buffer("input_lower", None)
            self.register_buffer("input_scale", None)
        else:
            bounds = torch.as_tensor(input_bounds, dtype=torch.get_default_dtype())
            self.register_buffer("input_lower", bounds[:, 0])
            self.register_buffer("input_scale", 2.0 / (bounds[:, 1] - bounds[:, 0]))

        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(activation())

        self.net = nn.Sequential(*modules)

    def forward(self, *coords):
        inputs = torch.cat(coords, dim=1)
        if self.input_lower is not None:
            inputs = (inputs - self.input_lower) * self.input_scale - 1.0
        return self.net(inputs)


def derivative(y, x):
    """helper function for autodiff to make higher order derivatives cleaner"""
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]


def train(model, loss_fn, epochs, snapshot_epochs, snapshot_fn, lr=1e-3, log_every=100):
    """generic training loop using adam optimiser

    callback function args:
    `loss_fn(model)` should return the scalar loss for autodiff
    `snapshot_fn(model)` should evaluates the model, it will get called
    for each snapshot value in `snapshot_epochs`"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    snapshots = {}
    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = loss_fn(model)

        loss.backward()
        optimizer.step()

        history.append((epoch, loss.item()))

        if epoch % log_every == 0:
            print(f"Epoch {epoch:4d} | Loss = {loss.item():.6e}")

        if epoch in snapshot_epochs:
            with torch.no_grad():
                snapshots[epoch] = snapshot_fn(model)

    return snapshots, history


def save_loss_history(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        writer.writerows(history)


# Plotting helpers
# -----------------
def _show_plot(plt):
    if os.environ.get("SHOWPLOT"):
        plt.show()


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def plot_progression_1d(
    x_plot,
    snapshots,
    exact,
    title,
    path,
    xlabel="x",
    ylabel="u(x)",
    exact_label="exact",
    prediction_cmap="viridis",
    final_color="#D55E00",
    ncols=3,
    figsize=None,
    dpi=300,
):
    x_np = _to_numpy(x_plot)
    exact_np = _to_numpy(exact)
    epochs = sorted(snapshots)
    ncols = min(ncols, len(epochs))
    nrows = (len(epochs) + ncols - 1) // ncols
    if figsize is None:
        figsize = (6.7, 1.65 * nrows + 0.45)
    intermediate_colors = plt.get_cmap(prediction_cmap)(
        np.linspace(0.12, 0.82, max(len(epochs) - 1, 1))
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    for index, (ax, epoch) in enumerate(zip(axes.flat, epochs)):
        is_final = index == len(epochs) - 1
        prediction_line = ax.plot(
            x_np,
            snapshots[epoch],
            color=final_color if is_final else intermediate_colors[index],
            linewidth=1.6 if is_final else 1.2,
            label="PINN",
        )[0]
        exact_line = ax.plot(
            x_np,
            exact_np,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=exact_label,
        )[0]
        ax.set_title(f"epoch {epoch}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.35)
        if index == 0:
            ax.legend(
                handles=[prediction_line, exact_line],
                fontsize=5.5,
                handlelength=1.6,
                labelspacing=0.25,
                borderpad=0.3,
            )

    for ax in axes.flat[len(epochs) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.supxlabel(xlabel, fontsize=8)
    fig.supylabel(ylabel, fontsize=8)

    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(path, dpi=dpi)
    _show_plot(plt)


def plot_field_panels(
    gx,
    gy,
    snapshots,
    exact,
    title,
    path,
    xlabel="x",
    ylabel="y",
    vmin=0.0,
    vmax=1.0,
    cmap="viridis",
    figsize=(15, 9),
    equal_aspect=False,
):
    panels = [(f"epoch {e}", snapshots[e]) for e in sorted(snapshots)]
    panels.append(("exact", _to_numpy(exact)))

    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    gx_np, gy_np = _to_numpy(gx), _to_numpy(gy)

    for ax, (subtitle, field) in zip(axes.flat, panels):
        im = ax.pcolormesh(
            gx_np, gy_np, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax.set_title(subtitle)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if equal_aspect:
            ax.set_aspect("equal")
        fig.colorbar(im, ax=ax)

    # hide any leftover axes when panels don't fill the grid
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    _show_plot(plt)


def plot_fieldlines(
    gx,
    gy,
    pinn_field,
    exact_field,
    title,
    path,
    xlabel="x",
    ylabel="y",
    figsize=(13, 6),
    equal_aspect=False,
):
    gx_np, gy_np = _to_numpy(gx), _to_numpy(gy)
    exact_np = _to_numpy(exact_field)
    pinn_np = _to_numpy(pinn_field)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, (subtitle, field) in zip(axes, [("PINN", pinn_np), ("exact", exact_np)]):
        ax.contour(
            gx_np,
            gy_np,
            field,
            colors="black",
            linewidths=1.0,
            negative_linestyles="dashed",
        )
        ax.set_title(subtitle)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if equal_aspect:
            ax.set_aspect("equal")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    _show_plot(plt)


def plot_surface_3d(
    gx,
    gy,
    pinn_field,
    exact_field,
    title,
    path,
    error_path,
    xlabel="x",
    ylabel="y",
    zlabel="u",
    cmap="viridis",
    error_cmap="viridis",
    figsize=(14, 6),
    error_figsize=(8, 6),
    equal_aspect=False,
):
    gx_np, gy_np = _to_numpy(gx), _to_numpy(gy)
    pinn_np, exact_np = _to_numpy(pinn_field), _to_numpy(exact_field)
    field_min = min(pinn_np.min(), exact_np.min())
    field_max = max(pinn_np.max(), exact_np.max())

    def surface(ax, field, subtitle, zl):
        ax.plot_surface(
            gx_np,
            gy_np,
            field,
            cmap=cmap,
            vmin=field_min,
            vmax=field_max,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zl)
        ax.set_title(subtitle)

    fig, axes = plt.subplots(1, 2, figsize=figsize, subplot_kw={"projection": "3d"})
    surface(axes[0], pinn_np, "PINN", zlabel)
    surface(axes[1], exact_np, "exact", zlabel)
    for ax in axes:
        ax.set_zlim((field_min, field_max))
    fig.suptitle(title)

    fig_err, ax_err = plt.subplots(figsize=error_figsize)
    im = ax_err.pcolormesh(
        gx_np,
        gy_np,
        np.abs(pinn_np - exact_np),
        shading="auto",
        cmap=error_cmap,
    )
    ax_err.set_xlabel(xlabel)
    ax_err.set_ylabel(ylabel)
    ax_err.set_title("absolute error")
    if equal_aspect:
        ax_err.set_aspect("equal")
    fig_err.colorbar(im, ax=ax_err)
    fig_err.suptitle(title)
    fig_err.savefig(error_path, dpi=150, bbox_inches="tight")

    # tight bbox clips z-labels on 3D axes; the science style sets
    # savefig.bbox="tight" globally, so force it off for this save
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(path, dpi=150)
    _show_plot(plt)


def plot_loss(history, title, path):
    epochs, losses = zip(*history)

    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, losses, color="black", linewidth=1.5)

    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.4)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    _show_plot(plt)
