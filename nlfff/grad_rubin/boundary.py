from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlphaBoundary:
    alpha: np.ndarray
    valid: np.ndarray
    polarity: int
    uncertainty: np.ndarray | None = None


def _gradient_uncertainty(error, spacing, axis):
    moved = np.moveaxis(error, axis, 0)
    result = np.empty_like(moved)
    result[1:-1] = np.hypot(moved[2:], moved[:-2]) / (2 * spacing)
    result[0] = (
        np.sqrt(2.25 * moved[0] ** 2 + 4 * moved[1] ** 2 + 0.25 * moved[2] ** 2)
        / spacing
    )
    result[-1] = (
        np.sqrt(
            2.25 * moved[-1] ** 2
            + 4 * moved[-2] ** 2
            + 0.25 * moved[-3] ** 2
        )
        / spacing
    )
    return np.moveaxis(result, 0, axis)


def uncertainty_weighted_average(alpha0, sigma0, alpha1, sigma1):
    alpha0, sigma0, alpha1, sigma1 = np.broadcast_arrays(
        alpha0, sigma0, alpha1, sigma1
    )

    floor = np.finfo(np.result_type(sigma0, sigma1, np.float32)).eps
    weight0 = np.zeros_like(sigma0, dtype=float)
    weight1 = np.zeros_like(sigma1, dtype=float)
    finite0 = np.isfinite(sigma0)
    finite1 = np.isfinite(sigma1)
    weight0[finite0] = np.maximum(sigma0[finite0], floor) ** -2
    weight1[finite1] = np.maximum(sigma1[finite1], floor) ** -2
    total = weight0 + weight1
    alpha = np.zeros_like(total)
    uncertainty = np.full_like(total, np.inf)
    known = total > 0
    alpha[known] = (
        weight0[known] * alpha0[known] + weight1[known] * alpha1[known]
    ) / total[known]
    uncertainty[known] = total[known] ** -0.5
    return alpha, uncertainty


def update_alpha_boundary(boundary, mapped_alpha, mapped_uncertainty):
    alpha, uncertainty = uncertainty_weighted_average(
        boundary.alpha,
        boundary.uncertainty,
        mapped_alpha,
        mapped_uncertainty,
    )
    alpha = np.where(boundary.valid, alpha, 0.0)
    uncertainty = np.where(boundary.valid, uncertainty, np.inf)
    return AlphaBoundary(alpha, boundary.valid, boundary.polarity, uncertainty)


def compute_alpha_boundary(
    field, spacing_xy, polarity, errors=None, weak_bz_fraction=0.0
):
    """Compute the (single polarity) Grad-Rubin boundary value ``alpha = curl(B)_z/Bz``.

    Field components are in Gauss and horizontal spacing is in Mm, so alpha is
    returned in Mm^-1. The explicit ``valid`` mask distinguishes the unprescribed
    polarity and undefined values from actual zero current points that we should take seriously.
    """
    dx, dy = spacing_xy
    bx, by, bz = np.moveaxis(field, -1, 0)
    curl_z = np.gradient(by, dx, axis=0, edge_order=2) - np.gradient(
        bx, dy, axis=1, edge_order=2
    )

    if weak_bz_fraction < 0:
        raise ValueError("weak_bz_fraction must be non-negative")
    bz_min = weak_bz_fraction * np.nanmax(np.abs(bz))
    valid = np.isfinite(bz) & np.isfinite(curl_z) & (polarity * bz > 0)
    strong = valid & (np.abs(bz) > bz_min)
    alpha = np.zeros_like(bz)
    alpha[strong] = curl_z[strong] / bz[strong]
    uncertainty = None
    if errors is not None:
        ebx, eby, ebz = np.moveaxis(np.abs(errors), -1, 0)
        sigma_curl = np.hypot(
            _gradient_uncertainty(eby, dx, axis=0),
            _gradient_uncertainty(ebx, dy, axis=1),
        )
        uncertainty = np.full_like(bz, np.inf)
        uncertainty[strong] = np.hypot(
            sigma_curl[strong] / np.abs(bz[strong]),
            curl_z[strong] * ebz[strong] / bz[strong] ** 2,
        )
        finite_uncertainty = uncertainty[strong & np.isfinite(uncertainty)]
        nominal_large = np.max(finite_uncertainty) if finite_uncertainty.size else 1.0
        uncertainty[valid & ~strong] = nominal_large
    return AlphaBoundary(
        alpha=alpha, valid=valid, polarity=polarity, uncertainty=uncertainty
    )
