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
# Simple harmonic oscillator: u'' + pi^2 u = 0 on (0, X_MAX)
# u(0) = 0, u'(0) = pi -> exact solution u(x) = sin(pi x)
# --------------------------------------------------

N_PERIODS = 3
X_MIN = 0.0
X_MAX = N_PERIODS * 2.0


def physics_loss(model):
    x = X_MIN + (X_MAX - X_MIN) * torch.rand(200, 1, requires_grad=True)

    u = model(x)
    d2u_dx2 = derivative(derivative(u, x), x)

    residual = d2u_dx2 + torch.pi**2 * u
    return torch.mean(residual**2)


def boundary_loss(model):
    x0 = torch.tensor([[X_MIN]], requires_grad=True)

    u0 = model(x0)
    du_dx0 = derivative(u0, x0)

    return torch.mean(u0**2 + (du_dx0 - torch.pi) ** 2)


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

save_loss_history(history, "losses/1_sin_ODE.csv")
plot_loss(history, "Training loss: SHO", "1_sin_ODE_loss.png")

plot_progression_1d(
    x_plot,
    snapshots,
    exact=torch.sin(torch.pi * x_plot),
    title="PINN Learning Progression: SHO",
    path="1_sin_ODE.png",
    exact_label="exact sin(pi x)",
)
