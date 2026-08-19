"""Shared visualisation helpers for reconstructed magnetic-field volumes."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm


def height_integrated_vector_magnitude(vectors, dz: float) -> np.ndarray:
    """Project a 3-D vector field to ``sum_z |vector| dz``.

    Arrays use the repository convention ``(x, y, z, component)``. ``dz`` must
    use the physical length unit appropriate to the supplied vector field.
    """
    values = np.asarray(vectors)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(
            "vectors must have shape (nx, ny, nz, 3), "
            f"got {values.shape}"
        )
    if not np.isfinite(dz) or dz <= 0:
        raise ValueError(f"dz must be finite and positive, got {dz}")

    # Sum in z slabs so large SHARP volumes do not require another full 3-D
    # magnitude array alongside the already resident vector cube.
    projected = np.zeros(values.shape[:2], dtype=np.float64)
    for start in range(0, values.shape[2], 16):
        slab = values[:, :, start : start + 16]
        projected += np.nansum(np.linalg.norm(slab, axis=-1), axis=2)
    return projected * dz


def height_integrated_curl_magnitude(field, spacing) -> np.ndarray:
    """Compute ``sum_z |curl(field)| dz`` without a full curl volume.

    The slab-wise implementation keeps large NF2 exports within workstation
    memory while retaining second-order finite differences at every boundary.
    """
    values = np.asarray(field)
    spacing = np.asarray(spacing, dtype=float)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(
            "field must have shape (nx, ny, nz, 3), "
            f"got {values.shape}"
        )
    if min(values.shape[:3]) < 3:
        raise ValueError("each field dimension must contain at least three points")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"spacing must contain three finite positive values, got {spacing}")

    dx, dy, dz = spacing
    projected = np.zeros(values.shape[:2], dtype=np.float64)
    for start in range(0, values.shape[2], 16):
        end = min(start + 16, values.shape[2])
        padded_start = max(0, start - 1)
        padded_end = min(values.shape[2], end + 1)
        slab = values[:, :, padded_start:padded_end]
        target = slice(start - padded_start, end - padded_start)

        curl_x = np.gradient(slab[..., 2], dy, axis=1, edge_order=2)
        curl_x -= np.gradient(slab[..., 1], dz, axis=2, edge_order=2)
        curl_y = np.gradient(slab[..., 0], dz, axis=2, edge_order=2)
        curl_y -= np.gradient(slab[..., 2], dx, axis=0, edge_order=2)
        curl_z = np.gradient(slab[..., 1], dx, axis=0, edge_order=2)
        curl_z -= np.gradient(slab[..., 0], dy, axis=1, edge_order=2)

        magnitude = np.sqrt(
            curl_x[:, :, target] ** 2
            + curl_y[:, :, target] ** 2
            + curl_z[:, :, target] ** 2
        )
        projected += np.nansum(magnitude, axis=2)
    return projected * dz


def positive_lognorm(values, percentiles=(1.0, 99.0)) -> LogNorm | None:
    """Return a robust logarithmic normalisation for positive finite values."""
    values = np.asarray(values)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return None

    vmin, vmax = np.nanpercentile(positive, percentiles)
    vmin = max(float(vmin), np.finfo(float).tiny)
    vmax = max(float(vmax), vmin * 1.01)
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_height_integrated_map(
    axis,
    current_map,
    *,
    extent,
    title=r"height-integrated $|\nabla\times B|$",
    cmap="inferno",
    percentiles=(1.0, 99.0),
):
    """Plot an already projected height-integrated current/curl map."""
    current_map = np.asarray(current_map)
    if current_map.ndim != 2:
        raise ValueError(f"current_map must be two-dimensional, got {current_map.shape}")
    image = axis.imshow(
        current_map.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=positive_lognorm(current_map, percentiles),
    )
    axis.set_title(title)
    return image


def plot_height_integrated_current(
    axis,
    current,
    dz: float,
    *,
    extent,
    title=r"height-integrated $|\nabla\times B|$",
    cmap="inferno",
    percentiles=(1.0, 99.0),
):
    """Plot a height-integrated current/curl magnitude on an existing axis.

    The caller owns the figure and colourbar because the displayed units depend
    on whether ``current`` is physical current density or ``curl(B)``.
    Returns ``(image, projected_map)`` for colourbars and downstream analysis.
    """
    current_map = height_integrated_vector_magnitude(current, dz)
    image = plot_height_integrated_map(
        axis,
        current_map,
        extent=extent,
        title=title,
        cmap=cmap,
        percentiles=percentiles,
    )
    return image, current_map
