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
# du/dx + u = 0  on (0, X_MAX),  u(0) = 1
# Exact solution: u(x) = exp(-x)
# --------------------------------------------------

X_MIN = 0.0
X_MAX = 5.0


def physics_loss(model):
    x = X_MIN + (X_MAX - X_MIN) * torch.rand(100, 1, requires_grad=True)

    u = model(x)
    du_dx = derivative(u, x)

    residual = du_dx + u
    return torch.mean(residual**2)


def boundary_loss(model):
    x_bc = torch.tensor([[X_MIN]])
    u_bc = torch.tensor([[1.0]])

    return torch.mean((model(x_bc) - u_bc) ** 2)


model = PINN([1, 32, 32, 1], input_bounds=[(X_MIN, X_MAX)])

x_plot = torch.linspace(X_MIN, X_MAX, 500).reshape(-1, 1)
snapshot_epochs = [0, 10, 50, 100, 250, 500, 1000, 2500, 4999]

snapshots, history = train(
    model,
    loss_fn=lambda m: physics_loss(m) + boundary_loss(m),
    epochs=5000,
    snapshot_epochs=snapshot_epochs,
    snapshot_fn=lambda m: m(x_plot).cpu().numpy().flatten(),
)

save_loss_history(history, "losses/0_exp_ODE.csv")
plot_loss(history, "Training loss: exp ODE", "0_exp_ODE_loss.png")

plot_progression_1d(
    x_plot,
    snapshots,
    exact=torch.exp(-x_plot),
    title="PINN Learning Progression",
    path="0_exp_ODE.png",
)
