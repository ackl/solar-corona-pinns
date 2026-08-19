import math
import os
import time
from pathlib import Path

import numpy as np

from .transport import TransportDiagnostics, sample_boundary


def _load_ufit():
    import ufit
    from ufit import UFiT_Functions_Python

    library = Path(ufit.__file__).with_name("UFiT_Python_Callable.so")
    if not library.is_file():
        raise RuntimeError(
            "the ufit package is installed but its shared library is missing; "
            "re-run `uv sync --group grad-rubin` to rebuild it"
        )
    return UFiT_Functions_Python, library


def _thread_count(threads):
    if threads is not None:
        return threads
    for name in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        value = os.environ.get(name)
        if value:
            return max(1, int(value))
    return os.cpu_count() or 1


def _periodic_field(index_field, periodic_horizontal):
    if not periodic_horizontal:
        return index_field
    with_x_endpoint = np.concatenate((index_field, index_field[:1]), axis=0)
    return np.concatenate((with_x_endpoint, with_x_endpoint[:, :1]), axis=1)


def _trace_chunk(
    module,
    library,
    index_field,
    positions,
    step,
    max_steps,
    weak_eps,
    periodic_horizontal,
    threads,
):
    options = module.UFiT_call_input()
    options.geometry = 0
    options.input_type = 0
    options.grid_regular = True
    options.grid_separate = False
    options.periodic_X = periodic_horizontal
    options.periodic_Y = periodic_horizontal
    options.periodic_Z = False
    options.save_endpoints = True
    options.save_Q = False
    options.save_fieldlines = False
    options.save_connection = False
    options.normalized_B = False
    options.num_proc = threads
    options.MAX_STEPS = max_steps
    options.integration_scheme = 4
    options.step_size = step
    options.weak_field_epsilon = weak_eps
    options.load_B = False
    options.return_output = True
    options.write_output = False
    options.coord1 = positions[:, 0]
    options.coord2 = positions[:, 1]
    options.coord3 = positions[:, 2]
    options.grid1 = np.arange(index_field.shape[0], dtype=float)
    options.grid2 = np.arange(index_field.shape[1], dtype=float)
    options.grid3 = np.arange(index_field.shape[2], dtype=float)
    options.B_grid = np.moveaxis(index_field, -1, 0)
    result = module.call_UFiT(str(library), options)
    # The upstream wrapper allocates this Fortran ``(6, n)`` output as a
    # C-contiguous NumPy array. Reinterpret its storage before indexing it.
    endpoints = result.endpoints.reshape(-1).reshape(len(positions), 6)
    backward = endpoints[:, :3]
    forward = endpoints[:, 3:]
    status = result.fieldline_status.reshape(-1).reshape(len(positions), 2)
    steps = result.fieldline_step_count.reshape(-1).reshape(len(positions), 2)
    return np.stack((backward, forward), axis=1), status, steps


def transport_alpha_ufit(
    field,
    grid,
    boundary,
    step=0.5,
    max_steps=None,
    batch_size=200_000,
    weak_eps=1e-8,
    periodic_horizontal=False,
    threads=None,
    return_uncertainty=False,
):
    if step <= 0 or batch_size <= 0 or (max_steps is not None and max_steps <= 0):
        raise ValueError("step, batch_size, and max_steps must be positive")
    field = np.ascontiguousarray(field, dtype=np.float32)
    total_start = time.perf_counter()
    setup_start = total_start
    if max_steps is None:
        max_steps = math.ceil(4 * sum(grid.shape) / step)
    module, library = _load_ufit()
    threads = _thread_count(threads)

    spacing = np.array(grid.spacing)
    index_field = field.astype(np.float64) / spacing
    index_field = _periodic_field(index_field, periodic_horizontal)

    total = math.prod(grid.shape)
    flat_alpha = np.zeros(total, dtype=np.float32)
    if return_uncertainty:
        finite_uncertainty = boundary.uncertainty[np.isfinite(boundary.uncertainty)]
        max_uncertainty = (
            float(np.max(finite_uncertainty)) if finite_uncertainty.size else 1.0
        )
    else:
        max_uncertainty = np.inf
    flat_uncertainty = np.full(total, max_uncertainty, dtype=np.float32)
    selected = np.zeros(total, dtype=bool)
    open_count = 0
    weak_count = 0
    step_total = 0
    setup_seconds = time.perf_counter() - setup_start
    trace_seconds = 0.0
    mapping_seconds = 0.0

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        flat_indices = np.arange(start, stop)
        positions = np.column_stack(np.unravel_index(flat_indices, grid.shape)).astype(
            float
        )
        section_start = time.perf_counter()
        endpoints, status, steps = _trace_chunk(
            module,
            library,
            index_field,
            positions,
            step,
            max_steps,
            weak_eps,
            periodic_horizontal,
            threads,
        )
        trace_seconds += time.perf_counter() - section_start

        section_start = time.perf_counter()
        closed = (status[:, 0] == 1) & (status[:, 1] == 1)
        feet = endpoints[:, :, :2]
        if periodic_horizontal:
            feet[:, :, 0] %= grid.nx
            feet[:, :, 1] %= grid.ny
        sample_feet = np.nan_to_num(feet, nan=0.0, posinf=0.0, neginf=0.0)
        backward_value = sample_boundary(
            boundary.alpha, sample_feet[:, 0], periodic_horizontal
        )
        forward_value = sample_boundary(
            boundary.alpha, sample_feet[:, 1], periodic_horizontal
        )
        backward_valid = (
            sample_boundary(boundary.valid, sample_feet[:, 0], periodic_horizontal)
            >= 0.5
        )
        forward_valid = (
            sample_boundary(boundary.valid, sample_feet[:, 1], periodic_horizontal)
            >= 0.5
        )
        accepted = closed & (backward_valid | forward_valid)
        values = np.where(forward_valid, forward_value, backward_value)
        flat_alpha[start:stop][accepted] = values[accepted]
        if return_uncertainty:
            backward_uncertainty = sample_boundary(
                boundary.uncertainty, sample_feet[:, 0], periodic_horizontal
            )
            forward_uncertainty = sample_boundary(
                boundary.uncertainty, sample_feet[:, 1], periodic_horizontal
            )
            uncertainties = np.where(
                forward_valid, forward_uncertainty, backward_uncertainty
            )
            flat_uncertainty[start:stop][accepted] = uncertainties[accepted]
        selected[start:stop] = accepted
        open_count += int(np.count_nonzero((status == 2).any(axis=1)))
        weak_count += int(np.count_nonzero((status == 3).any(axis=1)))
        step_total += int(np.sum(steps))
        mapping_seconds += time.perf_counter() - section_start

    finalize_start = time.perf_counter()
    alpha = flat_alpha.reshape(grid.shape)
    finalize_seconds = time.perf_counter() - finalize_start
    diagnostics = TransportDiagnostics(
        selected_fraction=float(np.count_nonzero(selected) / total),
        unresolved_fraction=float(np.count_nonzero(~selected) / total),
        open_fraction=float(open_count / total),
        weak_field_fraction=float(weak_count / total),
        mean_steps=float(step_total / (2 * total)),
        setup_seconds=setup_seconds,
        field_line_trace_seconds=trace_seconds,
        boundary_mapping_seconds=mapping_seconds,
        finalize_seconds=finalize_seconds,
        total_seconds=time.perf_counter() - total_start,
    )
    if return_uncertainty:
        return alpha, flat_uncertainty.reshape(grid.shape), diagnostics
    return alpha, diagnostics
