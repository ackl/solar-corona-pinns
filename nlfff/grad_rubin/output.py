import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from nlfff.visualization import plot_height_integrated_current

from .solver import result_metrics


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def save_result(
    result,
    grid,
    output_dir,
    metadata,
    boundary_alpha,
    boundary_valid,
    boundary_uncertainty=None,
    extra_metrics=None,
    wall_times=None,
    end_to_end_start=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_start = time.perf_counter()
    section_start = output_start
    arrays = {
        "b": result.field,
        "alpha": result.alpha,
        "current": result.current,
        "potential": result.potential,
        "spacing_Mm": np.array(grid.spacing),
        "origin_Mm": np.array(grid.origin),
        "alpha_boundary": boundary_alpha,
        "alpha_boundary_valid": boundary_valid,
    }
    if boundary_uncertainty is not None:
        arrays["alpha_boundary_uncertainty"] = boundary_uncertainty
    np.savez_compressed(output_dir / "field.npz", **arrays)
    output_arrays_seconds = time.perf_counter() - section_start

    section_start = time.perf_counter()
    metrics = result_metrics(result, grid)
    if extra_metrics is not None:
        metrics.update(extra_metrics)
    provenance = {
        "method": metadata["method"],
        "grid": asdict(grid),
        "solver": asdict(result.config),
        "polarity": result.polarity,
        **metadata,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=_json_default) + "\n"
    )
    output_diagnostics_seconds = time.perf_counter() - section_start

    section_start = time.perf_counter()
    if result.history:
        with (output_dir / "convergence.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result.history[0]))
            writer.writeheader()
            writer.writerows(result.history)
    output_csv_seconds = time.perf_counter() - section_start

    section_start = time.perf_counter()
    bz = result.field[:, :, 0, 2]
    alpha0 = boundary_alpha
    extent = [
        grid.origin[0],
        grid.origin[0] + (grid.nx - 1) * grid.dx,
        grid.origin[1],
        grid.origin[1] + (grid.ny - 1) * grid.dy,
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    limit = max(float(np.nanpercentile(np.abs(bz), 99)), 1e-12)
    im = axes[0].imshow(
        bz.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    axes[0].set_title(r"lower boundary $B_z$")
    fig.colorbar(im, ax=axes[0], label="G")
    alpha_limit = max(float(np.nanpercentile(np.abs(alpha0), 99)), 1e-12)
    im = axes[1].imshow(
        alpha0.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-alpha_limit,
        vmax=alpha_limit,
    )
    axes[1].set_title(r"transported boundary $\alpha$")
    fig.colorbar(im, ax=axes[1], label=r"Mm$^{-1}$")
    im, _ = plot_height_integrated_current(
        axes[2],
        result.current,
        grid.dz,
        extent=extent,
    )
    fig.colorbar(im, ax=axes[2], label="G")
    for axis in axes:
        axis.set_xlabel("x [Mm]")
        axis.set_ylabel("y [Mm]")
    fig.tight_layout()
    fig.savefig(output_dir / "quicklook.png", dpi=180)
    plt.close(fig)
    output_quicklook_seconds = time.perf_counter() - section_start
    output_total_seconds = time.perf_counter() - output_start

    timings = metrics.setdefault("wall_times_seconds", {})
    if wall_times is not None:
        timings.update(wall_times)
    timings.update(
        {
            "output_arrays": output_arrays_seconds,
            "output_diagnostics": output_diagnostics_seconds,
            "output_csv": output_csv_seconds,
            "output_quicklook": output_quicklook_seconds,
            "output_total": output_total_seconds,
        }
    )
    if end_to_end_start is not None:
        timings["end_to_end"] = time.perf_counter() - end_to_end_start
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_default) + "\n"
    )
    return metrics
