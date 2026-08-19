from dataclasses import dataclass

from scipy.ndimage import map_coordinates

from .boundary import AlphaBoundary
from .grid import CartesianGrid


@dataclass(frozen=True)
class TransportDiagnostics:
    selected_fraction: float
    unresolved_fraction: float
    open_fraction: float
    weak_field_fraction: float
    mean_steps: float
    setup_seconds: float = 0.0
    field_line_trace_seconds: float = 0.0
    boundary_mapping_seconds: float = 0.0
    finalize_seconds: float = 0.0
    total_seconds: float = 0.0


def sample_boundary(values, feet, periodic_horizontal):
    mode = "grid-wrap" if periodic_horizontal else "nearest"
    return map_coordinates(
        values,
        feet.T,
        order=1,
        mode=mode,
        prefilter=False,
    )


def transport_alpha(
    field,
    grid: CartesianGrid,
    boundary: AlphaBoundary,
    step=0.5,
    max_steps=None,
    batch_size=200_000,
    periodic_horizontal=False,
    backend="ufit",
    ufit_threads=None,
    fastqsl_integrator="rk4",
    fastqsl_adaptive_tolerance=1.0,
    return_uncertainty=False,
):
    if backend == "ufit":
        from .transport_ufit import transport_alpha_ufit

        return transport_alpha_ufit(
            field,
            grid,
            boundary,
            step=step,
            max_steps=max_steps,
            batch_size=batch_size,
            periodic_horizontal=periodic_horizontal,
            threads=ufit_threads,
            return_uncertainty=return_uncertainty,
        )
    if backend == "fastqsl":
        from .transport_fastqsl import transport_alpha_fastqsl

        return transport_alpha_fastqsl(
            field,
            grid,
            boundary,
            step=step,
            batch_size=batch_size,
            periodic_horizontal=periodic_horizontal,
            integrator=fastqsl_integrator,
            adaptive_tolerance=fastqsl_adaptive_tolerance,
            return_uncertainty=return_uncertainty,
        )
    raise ValueError(f"unknown transport backend: {backend!r}")
