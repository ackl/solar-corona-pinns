from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from util import (
    PINN,
    derivative,
    get_device,
    plot_field_panels,
    plot_loss,
    plot_surface_3d,
    save_loss_history,
    train,
)

torch.set_default_device(get_device())

MODEL_SEED = 0
SAMPLING_SEED = 0
torch.manual_seed(MODEL_SEED)

# --------------------------------------------------
# 2D Poisson on (0, X_MAX) x (0, Y_MAX), with u = 0 on the boundary
# Take f = 2 pi^2 sin(pi x) sin(pi y)  ->  u = sin(pi x) sin(pi y)
# Residual: u_xx + u_yy + f = 0
# --------------------------------------------------
N_PERIODS = 3
X_MIN = 0.0
X_MAX = 2.0 * N_PERIODS
Y_MIN = 0.0
Y_MAX = 2.0 * N_PERIODS

lhs = qmc.LatinHypercube(d=2, seed=SAMPLING_SEED)
boundary_rng = np.random.default_rng(SAMPLING_SEED + 1)


def physics_loss(model):
    pts = qmc.scale(lhs.random(700), [X_MIN, Y_MIN], [X_MAX, Y_MAX])
    x = torch.tensor(pts[:, :1], dtype=torch.get_default_dtype(), requires_grad=True)
    y = torch.tensor(pts[:, 1:], dtype=torch.get_default_dtype(), requires_grad=True)

    u = model(x, y)
    u_xx = derivative(derivative(u, x), x)
    u_yy = derivative(derivative(u, y), y)

    f = 2 * (torch.pi**2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    residual = u_xx + u_yy + f
    return torch.mean(residual**2)


def boundary_loss(model):
    # sample along the four edges of the domain
    n = 100
    x = torch.tensor(
        X_MIN + (X_MAX - X_MIN) * boundary_rng.random((n, 1)),
        dtype=torch.get_default_dtype(),
    )
    y = torch.tensor(
        Y_MIN + (Y_MAX - Y_MIN) * boundary_rng.random((n, 1)),
        dtype=torch.get_default_dtype(),
    )
    x_min = torch.full((n, 1), X_MIN)
    x_max = torch.full((n, 1), X_MAX)
    y_min = torch.full((n, 1), Y_MIN)
    y_max = torch.full((n, 1), Y_MAX)

    x_bc = torch.cat([x, x, x_min, x_max], dim=0)
    y_bc = torch.cat([y_min, y_max, y, y], dim=0)

    return torch.mean(model(x_bc, y_bc) ** 2)


model = PINN(
    [2, 64, 64, 64, 1],
    input_bounds=[(X_MIN, X_MAX), (Y_MIN, Y_MAX)],
)

# evaluation grid for snapshots / plotting
n = N_PERIODS * 60
xs = torch.linspace(X_MIN, X_MAX, n)
ys = torch.linspace(Y_MIN, Y_MAX, n)
gx, gy = torch.meshgrid(xs, ys, indexing="xy")
x_flat = gx.reshape(-1, 1)
y_flat = gy.reshape(-1, 1)

snapshot_epochs = [0, 100, 500, 1500, 4999]

snapshots, history = train(
    model,
    loss_fn=lambda m: physics_loss(m) + boundary_loss(m),
    epochs=5000,
    snapshot_epochs=snapshot_epochs,
    snapshot_fn=lambda m: m(x_flat, y_flat).reshape(n, n).cpu().numpy(),
)

Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/3_poisson_2d.pt")

save_loss_history(history, "losses/3_poisson_2d.csv")
plot_loss(history, "Training loss: 2D Poisson", "3_poisson_2d_loss.png")

exact = torch.sin(torch.pi * gx) * torch.sin(torch.pi * gy)
final_error = snapshots[max(snapshot_epochs)] - exact.detach().cpu().numpy()
print(f"Evaluation RMSE = {np.sqrt(np.mean(final_error**2)):.6e}")
print(f"Maximum absolute error = {np.max(np.abs(final_error)):.6e}")

plot_field_panels(
    gx,
    gy,
    snapshots,
    exact,
    title="PINN Learning Progression: 2D Poisson u(x, y)",
    path="3_poisson_2d.png",
    vmin=-1.0,
    vmax=1.0,
    cmap="RdBu_r",
    equal_aspect=True,
)

plot_surface_3d(
    gx,
    gy,
    snapshots[max(snapshot_epochs)],
    exact,
    title="2D Poisson: PINN vs exact u(x, y)",
    path="3_poisson_2d_surface.png",
    error_path="3_poisson_2d_surface_error.png",
    zlabel="u(x, y)",
    cmap="RdBu_r",
    equal_aspect=True,
)
