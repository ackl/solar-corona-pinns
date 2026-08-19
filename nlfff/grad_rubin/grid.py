from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartesianGrid:
    """Uniform Cartesian grid with arrays stored as ``(x, y, z, component)``."""

    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def nx(self):
        return self.shape[0]

    @property
    def ny(self):
        return self.shape[1]

    @property
    def nz(self):
        return self.shape[2]

    @property
    def dx(self):
        return self.spacing[0]

    @property
    def dy(self):
        return self.spacing[1]

    @property
    def dz(self):
        return self.spacing[2]

    @property
    def cell_length(self):
        return float(np.cbrt(np.prod(self.spacing)))

    @property
    def extent(self):
        return tuple(
            (start, start + (n - 1) * step)
            for start, n, step in zip(self.origin, self.shape, self.spacing)
        )

    def axes(self):
        return tuple(
            np.float32(start) + np.arange(n, dtype=np.float32) * np.float32(step)
            for start, n, step in zip(self.origin, self.shape, self.spacing)
        )

    def coordinates(self):
        return np.stack(np.meshgrid(*self.axes(), indexing="ij"), axis=-1)
