from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc
from util import (
    PINN,
    derivative,
    get_device,
    plot_field_panels,
    plot_fieldlines,
    plot_loss,
    plot_surface_3d,
    save_loss_history,
    train,
)

torch.set_default_device(get_device())

MODEL_SEED = 0
SAMPLING_SEED = 0

# --------------------------------------------------
# Solar coronal arcade (Baty & Vigon, arXiv:2310.17919, Section 3.1)
#
# Linear force-free field via the flux function psi(x, z):
# Residual: psi_xx + psi_zz + c^2 psi = 0
#
# --------------------------------------------------

# All three cases share L = 3, c = 0.8, a2 = 0
L = 3.0
C = 0.8
X_MIN = -L / 2.0
X_MAX = L / 2.0
Z_MIN = 0.0
Z_MAX = L

# tuples with amplitude values for each case
CASES = {
    "dipole": (1.0, 0.0, 0.0),
    "quadrupole": (0.0, 0.0, 1.0),
    "mixed": (1.0, 0.0, -0.5),
}


def exact_solution(x, z, amps):
    psi = torch.zeros_like(x)
    # using the fourier series analytical solution from the paper
    for k, a_k in enumerate(amps, start=1):
        nu_k = torch.sqrt(torch.tensor((k * torch.pi / L) ** 2 - C**2))
        psi += torch.exp(-nu_k * z) * a_k * torch.cos(k * torch.pi * x / L)
    return psi


def predict(model, x, z):
    return model(x, z)


def sample_training_points(seed):
    lhs = qmc.LatinHypercube(d=2, seed=seed)
    interior = qmc.scale(lhs.random(700), [X_MIN, Z_MIN], [X_MAX, Z_MAX])

    boundary_rng = np.random.default_rng(seed + 1)
    n = 20
    x_bot = X_MIN + (X_MAX - X_MIN) * boundary_rng.random((n, 1))
    x_top = X_MIN + (X_MAX - X_MIN) * boundary_rng.random((n, 1))
    z_left = Z_MIN + (Z_MAX - Z_MIN) * boundary_rng.random((n, 1))
    z_right = Z_MIN + (Z_MAX - Z_MIN) * boundary_rng.random((n, 1))
    x_bc = np.concatenate(
        [x_bot, x_top, np.full((n, 1), X_MIN), np.full((n, 1), X_MAX)]
    )
    z_bc = np.concatenate(
        [np.full((n, 1), Z_MIN), np.full((n, 1), Z_MAX), z_left, z_right]
    )
    return interior, x_bc, z_bc


def physics_loss(model, pts):
    x = torch.tensor(pts[:, :1], dtype=torch.get_default_dtype(), requires_grad=True)
    z = torch.tensor(pts[:, 1:], dtype=torch.get_default_dtype(), requires_grad=True)

    psi = predict(model, x, z)
    psi_xx = derivative(derivative(psi, x), x)
    psi_zz = derivative(derivative(psi, z), z)

    residual = psi_xx + psi_zz + C**2 * psi
    return torch.mean(residual**2)


def boundary_loss(model, amps, x_points, z_points):
    x_bc = torch.tensor(x_points, dtype=torch.get_default_dtype())
    z_bc = torch.tensor(z_points, dtype=torch.get_default_dtype())
    target = exact_solution(x_bc, z_bc, amps)
    return torch.mean((predict(model, x_bc, z_bc) - target) ** 2)


# evaluation grid for snapshots / plotting
n_x = 100
n_z = 100
xs = torch.linspace(X_MIN, X_MAX, n_x)
zs = torch.linspace(Z_MIN, Z_MAX, n_z)
gx, gz = torch.meshgrid(xs, zs, indexing="xy")
x_flat = gx.reshape(-1, 1)
z_flat = gz.reshape(-1, 1)

snapshot_epochs = [0, 1000, 5000, 15000, 49999]

Path("models").mkdir(exist_ok=True)

for case_index, (case, amps) in enumerate(CASES.items()):
    print(f"\nCASE: {case}  (a1, a2, a3) = {amps}")

    torch.manual_seed(MODEL_SEED + case_index)
    model = PINN([2] + [20] * 7 + [1])
    interior, x_boundary, z_boundary = sample_training_points(
        SAMPLING_SEED + case_index
    )

    def loss_fn(
        model,
        points=interior,
        case_amps=amps,
        x_points=x_boundary,
        z_points=z_boundary,
    ):
        return physics_loss(model, points) + boundary_loss(
            model, case_amps, x_points, z_points
        )

    snapshots, history = train(
        model,
        loss_fn=loss_fn,
        epochs=50000,
        snapshot_epochs=snapshot_epochs,
        snapshot_fn=lambda m: (
            predict(m, x_flat, z_flat).reshape(n_z, n_x).cpu().numpy()
        ),
        lr=2e-4,
        log_every=500,
    )

    torch.save(model.state_dict(), f"models/4_corona_arcade_{case}.pt")

    save_loss_history(history, f"losses/4_corona_arcade_{case}.csv")
    plot_loss(
        history,
        f"Training loss: Coronal Arcade ({case})",
        f"4_corona_arcade_{case}_loss.png",
    )

    exact = exact_solution(gx, gz, amps)
    final = snapshots[max(snapshot_epochs)]
    error = final - exact.detach().cpu().numpy()
    print(f"Evaluation RMSE = {np.sqrt(np.mean(error**2)):.6e}")
    print(f"Maximum absolute error = {np.max(np.abs(error)):.6e}")
    field_limit = exact.abs().max().item()

    plot_field_panels(
        gx,
        gz,
        snapshots,
        exact,
        title=f"PINN Learning Progression: Coronal Arcade psi(x, z), {case}",
        path=f"4_corona_arcade_{case}.png",
        ylabel="z",
        vmin=-field_limit,
        vmax=field_limit,
        cmap="RdBu_r",
        figsize=(18, 5),
        equal_aspect=True,
    )

    plot_fieldlines(
        gx,
        gz,
        snapshots[max(snapshot_epochs)],
        exact,
        title=f"Coronal Arcade ({case}): field lines (iso-contours of psi)",
        path=f"4_corona_arcade_{case}_fieldlines.png",
        ylabel="z",
        figsize=(18, 4),
        equal_aspect=True,
    )

    plot_surface_3d(
        gx,
        gz,
        snapshots[max(snapshot_epochs)],
        exact,
        title=f"Coronal Arcade ({case}): PINN vs exact psi(x, z)",
        path=f"4_corona_arcade_{case}_surface.png",
        error_path=f"4_corona_arcade_{case}_surface_error.png",
        ylabel="z",
        zlabel="psi(x, z)",
        cmap="RdBu_r",
        figsize=(18, 6),
        error_figsize=(14, 3.5),
        equal_aspect=True,
    )
