import math
import time
from functools import lru_cache

import numpy as np

from .transport import TransportDiagnostics, sample_boundary


@lru_cache(maxsize=2)
def _load_cuda_backend(integrator="rk4"):
    import cupy
    from fastqslpy import kernels
    variants = {
        "rk4": kernels.compileTraceBlineRK4,
        "adaptive": kernels.compileTraceBlineAdaptive,
    }
    try:
        compiler = variants[integrator]
    except KeyError as error:
        raise ValueError(f"unknown FastQSL integrator: {integrator!r}") from error
    return cupy, compiler


def _device_components(cupy, index_field):
    # FastQSL's CUDA index is z * Ny * Nx + y * Nx + x. Transpose each
    # component so x, not z, is contiguous in the device allocation.
    return tuple(
        cupy.asarray(
            np.ascontiguousarray(index_field[..., component].transpose(2, 1, 0))
        )
        for component in range(3)
    )


def transport_alpha_fastqsl(
    field,
    grid,
    boundary,
    step=0.5,
    batch_size=200_000,
    periodic_horizontal=False,
    integrator="rk4",
    adaptive_tolerance=1.0,
    return_uncertainty=False,
):
    field = np.ascontiguousarray(field, dtype=np.float32)
    total_start = time.perf_counter()
    setup_start = total_start

    cupy, compile_kernel = _load_cuda_backend(integrator)
    kernel = compile_kernel()
    spacing = np.array(grid.spacing, dtype=np.float32)
    index_field = np.ascontiguousarray(field / spacing)
    device_field = _device_components(cupy, index_field)
    device_shape = cupy.asarray(grid.shape, dtype=cupy.int32)
    device_cross_direction = cupy.asarray([0.0, 0.0, 1.0], dtype=cupy.float32)
    device_current_flag = cupy.asarray([False], dtype=cupy.bool_)
    device_step = cupy.asarray([step], dtype=cupy.float32)
    device_periodic = cupy.asarray([periodic_horizontal], dtype=cupy.bool_)
    device_adaptive_tolerance = cupy.asarray([adaptive_tolerance], dtype=cupy.float32)
    cupy.cuda.get_current_stream().synchronize()
    setup_seconds = time.perf_counter() - setup_start

    total = math.prod(grid.shape)
    flat_alpha = np.zeros(total, dtype=np.float32)
    if return_uncertainty:
        finite_uncertainty = boundary.uncertainty[
            np.isfinite(boundary.uncertainty)
        ]
        nominal_large = (
            float(np.max(finite_uncertainty)) if finite_uncertainty.size else 1.0
        )
    else:
        nominal_large = np.inf
    flat_uncertainty = np.full(total, nominal_large, dtype=np.float32)
    selected = np.zeros(total, dtype=bool)
    trace_seconds = 0.0
    mapping_seconds = 0.0
    open_count = 0
    line_length_total = 0.0
    threads_per_block = 256

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        flat_indices = np.arange(start, stop)
        seeds = np.column_stack(np.unravel_index(flat_indices, grid.shape)).astype(
            np.float32, copy=False
        )

        section_start = time.perf_counter()
        device_seeds = tuple(
            cupy.asarray(np.ascontiguousarray(seeds[:, axis])) for axis in range(3)
        )
        device_start = tuple(
            cupy.zeros(len(seeds), dtype=cupy.float32) for _ in range(3)
        )
        device_end = tuple(cupy.zeros(len(seeds), dtype=cupy.float32) for _ in range(3))
        device_start_flags = cupy.zeros(len(seeds), dtype=cupy.int32)
        device_end_flags = cupy.zeros(len(seeds), dtype=cupy.int32)
        device_line_lengths = cupy.zeros(len(seeds), dtype=cupy.float64)
        device_twist = cupy.zeros(len(seeds), dtype=cupy.float64)
        float_scratch = cupy.zeros(len(seeds), dtype=cupy.float32)
        int_scratch = cupy.zeros(len(seeds), dtype=cupy.int32)
        device_count = cupy.asarray([len(seeds)], dtype=cupy.uint64)
        blocks = (len(seeds) + threads_per_block - 1) // threads_per_block
        kernel_arguments = (
            *device_field,
            device_shape,
            *device_field,
            device_twist,
            device_current_flag,
            *device_seeds,
            device_cross_direction,
            *device_start,
            device_start_flags,
            *device_end,
            device_end_flags,
            float_scratch,
            float_scratch,
            float_scratch,
            int_scratch,
            float_scratch,
            float_scratch,
            float_scratch,
            float_scratch,
            float_scratch,
            float_scratch,
            device_step,
            device_count,
            device_line_lengths,
        )
        if integrator == "adaptive":
            kernel_arguments += (device_adaptive_tolerance,)
        kernel(
            (blocks,),
            (threads_per_block,),
            kernel_arguments + (device_periodic,),
        )
        cupy.cuda.get_current_stream().synchronize()
        trace_seconds += time.perf_counter() - section_start

        section_start = time.perf_counter()
        endpoints = np.stack(
            [
                np.column_stack(
                    (cupy.asnumpy(device_start[0]), cupy.asnumpy(device_start[1]))
                ),
                np.column_stack(
                    (cupy.asnumpy(device_end[0]), cupy.asnumpy(device_end[1]))
                ),
            ],
            axis=1,
        )
        statuses = np.column_stack(
            (cupy.asnumpy(device_start_flags), cupy.asnumpy(device_end_flags))
        )
        closed = (statuses[:, 0] == 5) & (statuses[:, 1] == 5)
        backward_value = sample_boundary(
            boundary.alpha, endpoints[:, 0], periodic_horizontal
        )
        forward_value = sample_boundary(
            boundary.alpha, endpoints[:, 1], periodic_horizontal
        )
        backward_valid = (
            sample_boundary(boundary.valid, endpoints[:, 0], periodic_horizontal) >= 0.5
        )
        forward_valid = (
            sample_boundary(boundary.valid, endpoints[:, 1], periodic_horizontal) >= 0.5
        )
        accepted = closed & (backward_valid | forward_valid)
        values = np.where(forward_valid, forward_value, backward_value)
        flat_alpha[start:stop][accepted] = values[accepted]
        if return_uncertainty:
            backward_uncertainty = sample_boundary(
                boundary.uncertainty, endpoints[:, 0], periodic_horizontal
            )
            forward_uncertainty = sample_boundary(
                boundary.uncertainty, endpoints[:, 1], periodic_horizontal
            )
            uncertainties = np.where(
                forward_valid, forward_uncertainty, backward_uncertainty
            )
            flat_uncertainty[start:stop][accepted] = uncertainties[accepted]
        selected[start:stop] = accepted
        open_count += int(
            np.count_nonzero(np.isin(statuses, (1, 2, 3, 4, 6)).any(axis=1))
        )
        line_length_total += float(np.sum(cupy.asnumpy(device_line_lengths)))
        mapping_seconds += time.perf_counter() - section_start

    finalize_start = time.perf_counter()
    alpha = flat_alpha.reshape(grid.shape)
    finalize_seconds = time.perf_counter() - finalize_start
    diagnostics = TransportDiagnostics(
        selected_fraction=float(np.count_nonzero(selected) / total),
        unresolved_fraction=float(np.count_nonzero(~selected) / total),
        open_fraction=float(open_count / total),
        weak_field_fraction=0.0,
        mean_steps=float(line_length_total / (2 * total * step)),
        setup_seconds=setup_seconds,
        field_line_trace_seconds=trace_seconds,
        boundary_mapping_seconds=mapping_seconds,
        finalize_seconds=finalize_seconds,
        total_seconds=time.perf_counter() - total_start,
    )
    if return_uncertainty:
        return alpha, flat_uncertainty.reshape(grid.shape), diagnostics
    return alpha, diagnostics
