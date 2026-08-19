import numpy as np
from nf2.data.analytical_field import get_analytic_b_field, solve_P

from .grid import CartesianGrid


def low_lou_field(
    resolution=25,
    n=1,
    m=1,
    l=0.3,
    psi=np.pi / 4,
    bounds=(-1, 1, -1, 1, 0, 2),
):
    """Return the Wheatland (2006) Low-Lou field and exact boundary alpha."""
    if n != 1:
        raise ValueError("only n=1 supported")
    shape = (resolution,) * 3 if isinstance(resolution, int) else tuple(resolution)
    if len(shape) != 3:
        raise ValueError("resolution must be an integer or three-element sequence")
    if len(bounds) != 6:
        raise ValueError("bounds must be (xmin, xmax, ymin, ymax, zmin, zmax)")

    bounds = tuple(float(value) for value in bounds)
    spacing = tuple(
        (stop - start) / (count - 1)
        for start, stop, count in zip(bounds[::2], bounds[1::2], shape)
    )
    grid = CartesianGrid(
        shape=shape,
        spacing=(spacing[0], spacing[1], spacing[2]),
        origin=(bounds[0], bounds[2], bounds[4]),
    )
    field = np.asarray(
        get_analytic_b_field(
            n=n,
            m=m,
            l=l,
            psi=psi,
            resolution=list(shape),
            bounds=list(bounds),
        ),
        dtype=np.float32,
    )

    solution, a_squared = solve_P(n, m)
    x, y, z = grid.coordinates().transpose(3, 0, 1, 2)
    X = x * np.cos(psi) - (z + l) * np.sin(psi)
    Y = y
    Z = x * np.sin(psi) + (z + l) * np.cos(psi)
    radius = np.sqrt(X**2 + Y**2 + Z**2)
    flux_function = solution(Z / radius)[0] / radius**n
    alpha = np.asarray(
        np.sqrt(a_squared) * (1 + 1 / n) * np.abs(flux_function) ** (1 / n),
        dtype=np.float32,
    )
    return grid, field, alpha
