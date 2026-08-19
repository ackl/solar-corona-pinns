import numpy as np
from nf2.evaluation.metric import curl, divergence, sigma_J, theta_J, vector_norm
from nf2.evaluation.metric import energy as nf2_energy

from .grid import CartesianGrid


def relative_l2(field, reference, eps=1e-30):
    denominator = max(float(np.linalg.norm(reference)), eps)
    return float(np.linalg.norm(field - reference) / denominator)


def fractional_flux(field, grid: CartesianGrid, eps=1e-12, mask=None):
    numerator = grid.cell_length * np.abs(physical_divergence(field, grid))
    values = numerator / (6 * vector_norm(field) + eps)
    if mask is not None:
        values = values[mask]
    value = np.mean(values) if values.size else 0.0
    return float(value)


def significant_fractional_flux(
    field,
    grid: CartesianGrid,
    relative_threshold=1e-6,
    absolute_threshold=1e-12,
):
    norm = vector_norm(field)
    threshold = max(float(np.max(norm)) * relative_threshold, absolute_threshold)
    return fractional_flux(field, grid, mask=norm > threshold)


def physical_curl(field, grid: CartesianGrid):
    spacing = np.array(grid.spacing, dtype=np.float32)
    scaled_field = field * spacing
    output_scale = spacing / np.prod(spacing)
    return curl(scaled_field) * output_scale


def physical_divergence(field, grid: CartesianGrid):
    return divergence(field / np.array(grid.spacing, dtype=np.float32))


def _force_free_metrics(field, current):
    if np.sum(vector_norm(current)) <= 1e-12:
        return 0.0, 0.0
    theta = float(theta_J(field, current))
    sigma = float(sigma_J(field, current))
    if not np.isfinite(sigma):
        sigma = float(np.sin(np.radians(theta)))
    return theta, float(np.clip(sigma, 0.0, 1.0))


def force_free_angles(field, current):
    """Return current-weighted angles over the interior and full grid.

    Curl estimates on the six faces use one-sided differences and include the
    mirrored-current lower-boundary error. The interior value is therefore the
    appropriate Wheatland (2006) convergence diagnostic
    """
    interior = (slice(1, -1),) * 3
    global_theta, global_sigma = _force_free_metrics(field, current)
    interior_theta, interior_sigma = _force_free_metrics(
        field[interior], current[interior]
    )
    return {
        "theta_J_interior_deg": interior_theta,
        "theta_J_global_deg": global_theta,
        "sigma_J_interior": interior_sigma,
        "sigma_J_global": global_sigma,
    }


def field_metrics(field, grid: CartesianGrid, potential):
    current = physical_curl(field, grid)
    angles = force_free_angles(field, current)
    norm = vector_norm(field)
    energy = nf2_energy(field)
    voxel_volume = np.prod(grid.spacing)
    metrics = {
        **angles,
        "mean_abs_fractional_flux": fractional_flux(field, grid),
        "mean_B_G": float(np.mean(norm)),
        "mean_curlB_G_per_Mm": float(np.mean(vector_norm(current))),
        "magnetic_energy_G2_Mm3": float(np.sum(energy) * voxel_volume),
    }
    potential_energy = nf2_energy(potential)
    metrics["energy_ratio_to_potential"] = float(
        np.sum(energy) / np.sum(potential_energy)
    )
    metrics["free_energy_G2_Mm3"] = float(
        np.sum(energy - potential_energy) * voxel_volume
    )
    return metrics
