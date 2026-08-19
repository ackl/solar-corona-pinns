import time
from dataclasses import asdict, dataclass

import numpy as np

from .boundary import AlphaBoundary
from .current import current_carrying_field
from .diagnostics import (
    field_metrics,
    force_free_angles,
    fractional_flux,
    physical_curl,
    relative_l2,
    significant_fractional_flux,
)
from .grid import CartesianGrid
from .potential import potential_field_from_bz
from .transport import transport_alpha


@dataclass(frozen=True)
class GradRubinConfig:
    max_iterations: int = 30
    stop_on_convergence: bool = True
    trace_step: float = 0.5
    trace_max_steps: int | None = None
    trace_batch_size: int = 200_000
    trace_backend: str = "ufit"
    trace_ufit_threads: int | None = None
    trace_fastqsl_integrator: str = "rk4"
    trace_fastqsl_adaptive_tolerance: float = 1.0
    periodic_horizontal: bool = False
    fft_workers: int | None = None


@dataclass
class GradRubinResult:
    field: np.ndarray
    alpha: np.ndarray
    current: np.ndarray
    potential: np.ndarray
    history: list[dict]
    converged: bool
    iterations: int
    polarity: int
    config: GradRubinConfig
    termination_reason: str
    runtime_seconds: float
    timings: dict
    alpha_uncertainty: np.ndarray | None = None


def _history_total(history, key):
    return sum(item[key] for item in history)


def solve_grad_rubin(
    grid: CartesianGrid,
    bz,
    boundary: AlphaBoundary,
    config: GradRubinConfig,
    callback=None,
):
    solver_start = time.perf_counter()
    setup_start = solver_start
    setup_seconds = time.perf_counter() - setup_start

    potential_start = time.perf_counter()
    potential = potential_field_from_bz(bz, grid)
    potential_seconds = time.perf_counter() - potential_start
    setup_start = time.perf_counter()
    field = potential.copy()

    history = []
    alpha = np.zeros(grid.shape, dtype=np.float32)
    alpha_uncertainty = None
    current = np.zeros((*grid.shape, 3), dtype=np.float32)
    converged = False
    previous_theta = None
    previous_current = None
    termination_reason = "max_iterations"
    setup_seconds += time.perf_counter() - setup_start

    for iteration in range(config.max_iterations):
        iteration_start = time.perf_counter()
        # Equations (6) and (8): alpha^k is constant on field lines of B^k.
        transported = transport_alpha(
            field,
            grid,
            boundary,
            step=config.trace_step,
            max_steps=config.trace_max_steps,
            batch_size=config.trace_batch_size,
            periodic_horizontal=config.periodic_horizontal,
            backend=config.trace_backend,
            ufit_threads=config.trace_ufit_threads,
            fastqsl_integrator=config.trace_fastqsl_integrator,
            fastqsl_adaptive_tolerance=config.trace_fastqsl_adaptive_tolerance,
            return_uncertainty=boundary.uncertainty is not None,
        )
        if boundary.uncertainty is None:
            alpha, transport = transported
        else:
            alpha, alpha_uncertainty, transport = transported

        section_start = time.perf_counter()
        current = alpha[..., None] * field
        current_change = (
            float("inf")
            if previous_current is None
            else relative_l2(current, previous_current)
        )
        current_source_seconds = time.perf_counter() - section_start

        # Equations (5) and (9)-(16): B^(k+1) = B0 + curl(A[J^k]).
        section_start = time.perf_counter()
        updated = potential + current_carrying_field(
            current, grid, workers=config.fft_workers
        )
        current_field_solve_seconds = time.perf_counter() - section_start

        section_start = time.perf_counter()
        change = relative_l2(updated, field)
        field_change_seconds = time.perf_counter() - section_start

        section_start = time.perf_counter()
        diagnostic_current = physical_curl(updated, grid)
        angles = force_free_angles(updated, diagnostic_current)
        theta_deg = angles["theta_J_interior_deg"]
        if previous_theta is None:
            theta_fractional_decrease = float("inf")
        elif abs(previous_theta) <= 1e-30:
            theta_fractional_decrease = (
                0.0 if abs(theta_deg) <= 1e-30 else float("-inf")
            )
        else:
            theta_fractional_decrease = (previous_theta - theta_deg) / previous_theta

        current_fractional_flux = significant_fractional_flux(current, grid)
        flux = fractional_flux(updated, grid)
        diagnostics_seconds = time.perf_counter() - section_start

        section_start = time.perf_counter()
        transport_values = asdict(transport)
        for key in (
            "setup_seconds",
            "field_line_trace_seconds",
            "boundary_mapping_seconds",
            "finalize_seconds",
            "total_seconds",
        ):
            transport_values.pop(key)
        record = {
            "iteration": iteration + 1,
            "relative_field_change": change,
            "relative_current_change": current_change,
            **angles,
            "theta_J_fractional_decrease": theta_fractional_decrease,
            "mean_abs_fractional_flux": flux,
            "current_mean_abs_fractional_flux": current_fractional_flux,
            **transport_values,
            "transport_seconds": transport.total_seconds,
            "transport_setup_seconds": transport.setup_seconds,
            "field_line_trace_seconds": transport.field_line_trace_seconds,
            "boundary_mapping_seconds": transport.boundary_mapping_seconds,
            "transport_finalize_seconds": transport.finalize_seconds,
            "current_source_seconds": current_source_seconds,
            "current_field_solve_seconds": current_field_solve_seconds,
            "field_change_seconds": field_change_seconds,
            "diagnostics_seconds": diagnostics_seconds,
        }
        history.append(record)
        field = updated

        converged_this_iteration = 0 <= theta_fractional_decrease < 0.01
        convergence_seconds = time.perf_counter() - section_start
        iteration_seconds = time.perf_counter() - iteration_start
        record["convergence_seconds"] = convergence_seconds
        record["iteration_seconds"] = iteration_seconds
        if callback is not None:
            callback(field, alpha, current, record)
        if converged_this_iteration:
            converged = True
            if config.stop_on_convergence:
                termination_reason = "theta_J_fractional_decrease"
                break
        previous_theta = theta_deg
        previous_current = current

    # Equation (6) is solved after the field update in the paper's indexing.
    transported = transport_alpha(
        field,
        grid,
        boundary,
        step=config.trace_step,
        max_steps=config.trace_max_steps,
        batch_size=config.trace_batch_size,
        periodic_horizontal=config.periodic_horizontal,
        backend=config.trace_backend,
        ufit_threads=config.trace_ufit_threads,
        fastqsl_integrator=config.trace_fastqsl_integrator,
        fastqsl_adaptive_tolerance=config.trace_fastqsl_adaptive_tolerance,
        return_uncertainty=boundary.uncertainty is not None,
    )
    if boundary.uncertainty is None:
        alpha, final_transport = transported
    else:
        alpha, alpha_uncertainty, final_transport = transported
    section_start = time.perf_counter()
    current = alpha[..., None] * field
    final_current_source_seconds = time.perf_counter() - section_start
    runtime_seconds = time.perf_counter() - solver_start
    timings = {
        "solver_total": runtime_seconds,
        "solver_setup": setup_seconds,
        "potential_field": potential_seconds,
        "iterations_total": _history_total(history, "iteration_seconds"),
        "final_transport": final_transport.total_seconds,
        "final_transport_setup": final_transport.setup_seconds,
        "final_field_line_trace": final_transport.field_line_trace_seconds,
        "final_boundary_mapping": final_transport.boundary_mapping_seconds,
        "final_transport_finalize": final_transport.finalize_seconds,
        "final_current_source": final_current_source_seconds,
    }
    return GradRubinResult(
        field=field,
        alpha=alpha,
        current=current,
        potential=potential,
        history=history,
        converged=converged,
        iterations=len(history),
        polarity=boundary.polarity,
        config=config,
        termination_reason=termination_reason,
        runtime_seconds=runtime_seconds,
        timings=timings,
        alpha_uncertainty=alpha_uncertainty,
    )


def result_metrics(result: GradRubinResult, grid: CartesianGrid):
    metrics = field_metrics(result.field, grid, result.potential)
    iteration_count = max(len(result.history), 1)
    wall_times = {
        **result.timings,
        "iterations_mean": result.timings["iterations_total"] / iteration_count,
        "transport_total": _history_total(result.history, "transport_seconds"),
        "transport_mean": _history_total(result.history, "transport_seconds")
        / iteration_count,
        "transport_setup_total": _history_total(
            result.history, "transport_setup_seconds"
        ),
        "field_line_trace_total": _history_total(
            result.history, "field_line_trace_seconds"
        ),
        "boundary_mapping_total": _history_total(
            result.history, "boundary_mapping_seconds"
        ),
        "transport_finalize_total": _history_total(
            result.history, "transport_finalize_seconds"
        ),
        "current_source_total": _history_total(
            result.history, "current_source_seconds"
        ),
        "current_field_solve_total": _history_total(
            result.history, "current_field_solve_seconds"
        ),
        "field_change_total": _history_total(result.history, "field_change_seconds"),
        "diagnostics_total": _history_total(result.history, "diagnostics_seconds"),
        "convergence_total": _history_total(result.history, "convergence_seconds"),
    }
    metrics.update(
        {
            "converged": result.converged,
            "iterations": result.iterations,
            "termination_reason": result.termination_reason,
            "runtime_seconds": result.runtime_seconds,
            "mean_iteration_seconds": (
                sum(item["iteration_seconds"] for item in result.history)
                / iteration_count
            ),
            "wall_times_seconds": wall_times,
            "polarity": result.polarity,
            "compute_backend": "cpu",
            "lower_bz_relative_error": relative_l2(
                result.field[:, :, 0, 2], result.potential[:, :, 0, 2]
            ),
        }
    )
    return metrics
