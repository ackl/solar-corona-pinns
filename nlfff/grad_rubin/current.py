import numpy as np
from scipy import fft

from .grid import CartesianGrid


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def embed_cfit_current(current):
    """Construct Wheatland's mirrored J volume. Padded with zeros to mitigate
    periodicity due to FFT

    The lower layer is duplicated. Consequently the reflection plane is half a
    grid cell below z=0, exactly as described after Equation (16).
    """
    nx, ny, nz, _ = current.shape
    px = _next_power_of_two(nx)
    py = _next_power_of_two(ny)
    pz = _next_power_of_two(2 * nz)
    embedded = np.zeros((px, py, pz, 3), dtype=current.dtype)
    x0 = (px - nx) // 2
    y0 = (py - ny) // 2
    xs = slice(x0, x0 + nx)
    ys = slice(y0, y0 + ny)
    embedded[xs, ys, :nz] = current

    mirror = np.flip(current, axis=2).copy()
    mirror[..., :2] *= -1
    embedded[xs, ys, -nz:] = mirror
    return embedded, (xs, ys, slice(0, nz))


def current_carrying_field(current, grid: CartesianGrid, workers=None):
    """Solve Wheatland (2006), Equations (12)-(16), with SciPy FFTs.

    This intentionally does not post-correct ``Bc_z(z=0)``: CFIT duplicates
    the lower current layer and accepts the resulting small boundary error.
    """
    embedded, crop = embed_cfit_current(current)
    px, py, pz, _ = embedded.shape
    kx = (2 * np.pi * fft.fftfreq(px, d=grid.dx)).astype(np.float32)
    ky = (2 * np.pi * fft.fftfreq(py, d=grid.dy)).astype(np.float32)
    kz = (2 * np.pi * fft.fftfreq(pz, d=grid.dz)).astype(np.float32)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2

    j_hat = fft.fftn(embedded, axes=(0, 1, 2), workers=workers)
    cross = np.empty_like(j_hat)
    cross[..., 0] = ky * j_hat[..., 2] - kz * j_hat[..., 1]
    cross[..., 1] = kz * j_hat[..., 0] - kx * j_hat[..., 2]
    cross[..., 2] = kx * j_hat[..., 1] - ky * j_hat[..., 0]
    denominator = np.where(k2 > 0, k2, 1)
    b_hat = 1j * cross / denominator[..., None]
    b_hat[0, 0, 0] = 0
    return fft.ifftn(b_hat, axes=(0, 1, 2), workers=workers).real[crop]
