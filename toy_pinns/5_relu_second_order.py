import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from util import PINN, derivative

# Keep the repository's plotting style while remaining runnable on CPU-only
# environments without a complete LaTeX installation.
plt.rcParams["text.usetex"] = False

LAYERS = [1, 32, 32, 1]
N_PERIODS = 3
X_MIN = 0.0
X_MAX = 2.0 * N_PERIODS
N_POINTS = N_PERIODS * 200
SEED = 7
LEARNING_RATE = 1.0e-3

def exact_solution(x):
    return torch.sin(torch.pi * x)


def field_quantities(model, x):
    u = model(x)
    curvature = derivative(derivative(u, x), x)
    residual = curvature + torch.pi**2 * exact_solution(x)
    return u, curvature, residual


def losses(model, x, x_boundary):
    _, _, residual = field_quantities(model, x)
    pde = torch.mean(residual**2)
    boundary = torch.mean(model(x_boundary) ** 2)
    return pde, boundary


def make_model(activation):
    torch.manual_seed(SEED)
    return PINN(
        LAYERS,
        activation=activation,
        input_bounds=[(X_MIN, X_MAX)],
    )


def train_model(name, activation, epochs, x, x_boundary):
    model = make_model(activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        pde, boundary = losses(model, x, x_boundary)
        total = pde + boundary
        total.backward()
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "model": name,
                "total_loss": total.item(),
                "pde_loss": pde.item(),
                "boundary_loss": boundary.item(),
            }
        )

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(
                f"{name:16s} epoch {epoch:4d} | total={total.item():.3e} "
                f"pde={pde.item():.3e} bc={boundary.item():.3e}"
            )

    return model, history


def evaluate(name, model, x_dense, x_boundary):
    u, curvature, residual = field_quantities(model, x_dense)
    pde, boundary = losses(model, x_dense, x_boundary)
    exact = exact_solution(x_dense)
    total = pde + boundary

    summary = {
        "model": name,
        "total_loss": total.item(),
        "pde_loss": pde.item(),
        "boundary_loss": boundary.item(),
        "relative_l2_error": (
            torch.linalg.vector_norm(u - exact) / torch.linalg.vector_norm(exact)
        ).item(),
        "pde_residual_rms": torch.sqrt(torch.mean(residual**2)).item(),
        "boundary_error": torch.max(torch.abs(model(x_boundary))).item(),
        "curvature_rms": torch.sqrt(torch.mean(curvature**2)).item(),
    }
    return summary, u.detach().numpy().ravel()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path, histories, predictions, x_dense):
    labels = {
        "tanh_pinn": "tanh PINN",
        "relu_pinn": "ReLU PINN",
    }
    colors = {
        "tanh_pinn": "#0072B2",
        "relu_pinn": "#D55E00",
    }
    x_np = x_dense.detach().numpy().ravel()
    exact_np = np.sin(np.pi * x_np)

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.4), constrained_layout=True)
    for name, label in labels.items():
        axes[0].plot(x_np, predictions[name], color=colors[name], label=label)
    axes[0].plot(x_np, exact_np, "k--", linewidth=1, label="exact")
    axes[0].set(xlabel="$x$", ylabel="$u(x)$", title="Solution")
    axes[0].legend(fontsize=8)

    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[1].semilogy(
            epochs,
            [row["total_loss"] for row in history],
            color=colors[name],
            label=labels[name],
        )
    axes[1].set(xlabel="epoch", ylabel="total loss", title="PINN training loss")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Activation smoothness in a second-order PINN", fontsize=11)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_default_device("cpu")
    torch.use_deterministic_algorithms(True)
    x = (
        torch.linspace(X_MIN, X_MAX, N_POINTS + 2)[1:-1]
        .reshape(-1, 1)
        .requires_grad_(True)
    )
    x_boundary = torch.tensor([[X_MIN], [X_MAX]])
    configurations = {
        "tanh_pinn": nn.Tanh,
        "relu_pinn": nn.ReLU,
    }
    models = {}
    histories = {}
    for name, activation in configurations.items():
        models[name], histories[name] = train_model(
            name, activation, args.epochs, x, x_boundary
        )

    x_dense = (
        torch.linspace(X_MIN, X_MAX, N_PERIODS * 1000 + 1)
        .reshape(-1, 1)
        .requires_grad_(True)
    )
    summaries = []
    predictions = {}
    for name in configurations:
        summary, predictions[name] = evaluate(name, models[name], x_dense, x_boundary)
        summaries.append(summary)
        print(
            f"SUMMARY {name}: "
            + ", ".join(
                f"{key}={value:.6e}" for key, value in summary.items() if key != "model"
            )
        )

    output_dir = args.output_dir
    write_csv(
        output_dir / "losses" / "5_relu_second_order.csv",
        [row for history in histories.values() for row in history],
    )
    write_csv(output_dir / "5_relu_second_order_summary.csv", summaries)
    plot_results(
        output_dir / "5_relu_second_order.png",
        histories,
        predictions,
        x_dense,
    )


if __name__ == "__main__":
    main()
