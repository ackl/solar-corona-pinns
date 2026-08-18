import torch
from util import (
    PINN,
    derivative,
    get_device,
    train,
    save_loss_history,
    plot_loss,
    plot_progression_1d,
)

torch.set_default_device(get_device())


# --------------------------------------------------
# 1D Poisson: -u''(x) = f(x) on (0, X_MAX),  u = 0 on the boundary
# Take f(x) = pi^2 sin(pi x)  ->  exact solution u(x) = sin(pi x)
# Residual: u'' + pi^2 sin(pi x) = 0
# --------------------------------------------------
N_PERIODS = 3
X_MIN = 0.0
X_MAX = 2.0 * N_PERIODS


def physics_loss(model):
    x = X_MIN + (X_MAX - X_MIN) * torch.rand(200, 1, requires_grad=True)

    u = model(x)
    d2u_dx2 = derivative(derivative(u, x), x)

    f = (torch.pi**2) * torch.sin(torch.pi * x)
    residual = d2u_dx2 + f
    return torch.mean(residual**2)


def boundary_loss(model):
    x_bc = torch.tensor([[X_MIN], [X_MAX]])
    u_bc = torch.tensor([[0.0], [0.0]])

    return torch.mean((model(x_bc) - u_bc) ** 2)


model = PINN([1, 32, 32, 1], input_bounds=[(X_MIN, X_MAX)])

x_plot = torch.linspace(X_MIN, X_MAX, N_PERIODS * 300).reshape(-1, 1)
snapshot_epochs = [0, 10, 50, 100, 250, 500, 1000, 2500, 4999]

snapshots, history = train(
    model,
    loss_fn=lambda m: physics_loss(m) + boundary_loss(m),
    epochs=5000,
    snapshot_epochs=snapshot_epochs,
    snapshot_fn=lambda m: m(x_plot).cpu().numpy().flatten(),
)

save_loss_history(history, "losses/2_poisson_1d.csv")
plot_loss(history, "Training loss: 1D Poisson", "2_poisson_1d_loss.png")

plot_progression_1d(
    x_plot,
    snapshots,
    exact=torch.sin(torch.pi * x_plot),
    title="PINN Learning Progression: 1D Poisson",
    path="2_poisson_1d.png",
    exact_label="exact sin(pi x)",
)
