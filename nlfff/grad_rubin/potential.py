import numpy as np
from nf2.potential.potential_field import get_fft_potential_field

from .grid import CartesianGrid


def potential_field_from_bz(bz, grid: CartesianGrid):
    """Evaluate Wheatland (2006), Equation (14), with NF2's FFT solver."""
    if not np.isclose(grid.dx, grid.dy):
        raise ValueError("NF2's FFT potential solver requires equal x and y spacing")

    field = get_fft_potential_field(bz, grid.nz, scale=grid.dz / grid.dx)
    field[:, :, 0, 2] = bz
    return field
