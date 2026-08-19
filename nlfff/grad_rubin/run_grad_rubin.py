"""Run the Wheatland (2006) CFIT NLFFF extrap

The ``low-lou`` command reproduces the nonlinear analytical benchmark used by
Wheatland. The ``sharp`` command consumes the same SHARP CEA FITS components as
``nlfff/pinn/run_nf2_sharp.py``.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from nlfff.config import ConfigError, load_config, public_config

from .analytic import low_lou_field
from .boundary import AlphaBoundary, compute_alpha_boundary, update_alpha_boundary
from .grid import CartesianGrid
from .output import save_result
from .sharp import load_sharp_boundary
from .solver import GradRubinConfig, solve_grad_rubin


@dataclass(frozen=True)
class GradRubinRunConfig(GradRubinConfig):
    output: Path = Path("runs/cfit_sharp")
    data_dir: Path | None = None
    polarity: str = "both"
    resolution: int = 25
    bin: int = 4
    crop: list[int] | None = None
    nz: int = 64
    dz: float = 1.44
    self_consistency_cycles: int = 0
    weak_bz_fraction: float = 0.0


def _config(args):
    return GradRubinConfig(
        **{
            definition.name: vars(args)[definition.name]
            for definition in fields(GradRubinConfig)
        }
    )


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _callback(_field, _alpha, _current, record):
    print(
        "[cfit] iteration={iteration:03d} dB={relative_field_change:.3e} "
        "thetaJ={theta_J_interior_deg:.3f}deg decrease={theta_J_fractional_decrease:.3e} "
        "div={mean_abs_fractional_flux:.3e} mapped={selected_fraction:.3f}".format(**record),
        flush=True,
    )


def run_self_consistency(grid, bz, boundaries, config, cycles, callback):
    if cycles < 1:
        raise ValueError("self-consistency cycles must be positive")
    results = None
    solved_boundaries = boundaries
    config = replace(config, stop_on_convergence=False)
    for cycle in range(cycles):
        print(f"[cfit] self-consistency cycle={cycle + 1}/{cycles}", flush=True)
        solved_boundaries = boundaries
        results = {
            polarity: solve_grad_rubin(
                grid, bz, boundary, config=config, callback=callback
            )
            for polarity, boundary in boundaries.items()
        }
        boundaries = {
            1: update_alpha_boundary(
                boundaries[1],
                results[-1].alpha[:, :, 0],
                results[-1].alpha_uncertainty[:, :, 0],
            ),
            -1: update_alpha_boundary(
                boundaries[-1],
                results[1].alpha[:, :, 0],
                results[1].alpha_uncertainty[:, :, 0],
            ),
        }
    assert results is not None
    return results, solved_boundaries, boundaries


def run_low_lou(args):
    end_to_end_start = time.perf_counter()
    section_start = end_to_end_start
    grid, exact, exact_alpha = low_lou_field(args.resolution)
    bz = exact[:, :, 0, 2]
    polarity = 1 if args.polarity == "positive" else -1
    valid = np.isfinite(bz) & (polarity * bz > 0)
    alpha0 = exact_alpha[:, :, 0].copy()
    alpha0[~valid] = 0
    boundary = AlphaBoundary(
        alpha=alpha0,
        valid=valid,
        polarity=polarity,
    )
    case_setup_seconds = time.perf_counter() - section_start

    section_start = time.perf_counter()
    result = solve_grad_rubin(
        grid, bz, boundary, config=_config(args), callback=_callback
    )
    solver_call_seconds = time.perf_counter() - section_start

    section_start = time.perf_counter()
    field = result.field
    alpha = result.alpha
    relative_error = np.linalg.norm(field - exact) / np.linalg.norm(exact)
    mapped = np.abs(alpha) > 0
    mapped_error = (
        np.linalg.norm((field - exact)[mapped]) / np.linalg.norm(exact[mapped])
        if np.any(mapped)
        else float("nan")
    )
    alpha_error = np.linalg.norm(alpha - exact_alpha) / np.linalg.norm(exact_alpha)
    exact_comparison_seconds = time.perf_counter() - section_start
    metrics = save_result(
        result,
        grid,
        args.output,
        metadata={
            "method": (
                f"cfit_wheatland_2006_{result.config.trace_backend}_"
                f"{result.config.trace_fastqsl_integrator}"
                if result.config.trace_backend == "fastqsl"
                else f"cfit_wheatland_2006_{result.config.trace_backend}"
            ),
            "case": "Low_Lou_1990_n1_m1",
            "analytic_generator": "nf2.data.analytical_field.get_analytic_b_field",
            "n": 1,
            "m": 1,
            "source_depth": 0.3,
            "inclination_rad": float(np.pi / 4),
            "polarity_label": "P" if polarity > 0 else "N",
            "alpha_boundary_source": "exact Low-Lou dQ/dA on every finite selected-polarity pixel",
            "uncertainty_preprocessing": False,
            "normalization": "Low-Lou dimensionless coordinates and field amplitude",
        },
        boundary_alpha=alpha0,
        boundary_valid=valid,
        extra_metrics={
            "relative_exact_error": float(relative_error),
            "mapped_volume_relative_exact_error": float(mapped_error),
            "mapped_volume_fraction": float(np.mean(mapped)),
            "relative_exact_alpha_error": float(alpha_error),
        },
        wall_times={
            "case_setup": case_setup_seconds,
            "solver_call": solver_call_seconds,
            "exact_comparison": exact_comparison_seconds,
        },
        end_to_end_start=end_to_end_start,
    )
    print(
        f"[cfit] converged={result.converged} relative_exact_error={relative_error:.6e}"
    )
    print(f"[cfit] metrics={metrics}")


def run_sharp(args):
    if args.self_consistency_cycles and args.polarity != "both":
        raise ValueError("self-consistency requires --polarity both")
    if args.self_consistency_cycles and args.weak_bz_fraction != 0.01:
        raise ValueError("self-consistency requires --weak-bz-fraction 0.01")
    sharp = load_sharp_boundary(
        args.data_dir,
        bin_factor=args.bin,
        crop=args.crop,
    )
    nx, ny, _ = sharp.field.shape
    grid = CartesianGrid(
        shape=(nx, ny, args.nz),
        spacing=(*sharp.spacing_Mm, args.dz),
    )
    polarities = {
        "both": (1, -1),
        "positive": (1,),
        "negative": (-1,),
    }[args.polarity]
    metadata = {
        "case": "SHARP_CEA",
        "input_loader": "nf2.loader.cartesian_datasets.SHARPDataset",
        "potential_solver": "nf2.potential.potential_field.get_fft_potential_field",
        "bin_factor": args.bin,
        "crop": args.crop,
        "alpha_boundary_policy": (
            f"alpha=0 for |Bz| <= {args.weak_bz_fraction:g} max(|Bz|); "
            "propagated SHARP uncertainty"
            if args.weak_bz_fraction
            else "all finite pixels on the prescribed polarity"
        ),
        "input_sha256": {
            name: _hash(path) for name, path in sharp.files.items() if path.exists()
        },
    }
    flux_imbalance = abs(np.sum(sharp.field[..., 2])) / max(
        np.sum(np.abs(sharp.field[..., 2])), 1e-30
    )
    metadata["unsigned_flux_imbalance"] = float(flux_imbalance)
    print(f"[cfit] unsigned flux imbalance={flux_imbalance:.6f}", flush=True)
    boundaries = {
        polarity: compute_alpha_boundary(
            sharp.field,
            sharp.spacing_Mm,
            polarity,
            errors=sharp.errors if args.self_consistency_cycles else None,
            weak_bz_fraction=args.weak_bz_fraction,
        )
        for polarity in polarities
    }
    if args.self_consistency_cycles:
        if sharp.errors is None:
            raise ValueError("self-consistency requires SHARP uncertainty maps")
        results, solved_boundaries, _ = run_self_consistency(
            grid,
            sharp.field[..., 2],
            boundaries,
            _config(args),
            args.self_consistency_cycles,
            _callback,
        )
        metadata["self_consistency_cycles"] = args.self_consistency_cycles
        metadata["alpha_boundary_update"] = "inverse-variance weighted average"
        metadata["missing_boundary_information"] = (
            "alpha=0 with nominal large uncertainty on open field lines"
        )
    else:
        solved_boundaries = boundaries
        results = {
            polarity: solve_grad_rubin(
                grid,
                sharp.field[..., 2],
                boundary,
                config=_config(args),
                callback=_callback,
            )
            for polarity, boundary in boundaries.items()
        }

    for polarity, result in results.items():
        boundary = solved_boundaries[polarity]
        label = "P" if polarity > 0 else "N"
        print(
            f"[cfit] polarity={label} valid_alpha_pixels={boundary.valid.sum()} "
            f"of {boundary.valid.size}",
            flush=True,
        )
        if args.self_consistency_cycles:
            method = "cfit_wheatland_leka_2011_uncertainty_weighted"
        elif result.config.trace_backend == "fastqsl":
            method = (
                f"cfit_wheatland_2006_fastqsl_"
                f"{result.config.trace_fastqsl_integrator}"
            )
        else:
            method = f"cfit_wheatland_2006_{result.config.trace_backend}"
        metrics = save_result(
            result,
            grid,
            Path(args.output) / label,
            metadata={**metadata, "method": method, "polarity_label": label},
            boundary_alpha=boundary.alpha,
            boundary_valid=boundary.valid,
            boundary_uncertainty=boundary.uncertainty,
        )
        print(f"[cfit] {label} converged={result.converged} metrics={metrics}")


def add_solver_args(parser, periodic_horizontal=False):
    parser.add_argument("--iterations", dest="max_iterations", type=int, default=None)
    parser.add_argument("--trace-step", type=float, default=None)
    parser.add_argument("--trace-max-steps", type=int, default=None)
    parser.add_argument("--trace-batch-size", type=int, default=None)
    parser.add_argument("--trace-backend", choices=["ufit", "fastqsl"], default=None)
    parser.add_argument(
        "--ufit-threads", dest="trace_ufit_threads", type=int, default=None
    )
    parser.add_argument(
        "--fastqsl-integrator",
        dest="trace_fastqsl_integrator",
        choices=["rk4", "adaptive"],
        default=None,
    )
    parser.add_argument(
        "--fastqsl-adaptive-tolerance",
        dest="trace_fastqsl_adaptive_tolerance",
        type=float,
        default=None,
    )
    parser.add_argument("--fft-workers", type=int, default=None)
    parser.add_argument(
        "--periodic-horizontal", action=argparse.BooleanOptionalAction, default=None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    low_lou = sub.add_parser("low-lou", help="run Wheatland's Low--Lou NLFFF benchmark")
    low_lou.add_argument("--config", type=Path, default=None)
    low_lou.add_argument("--output", type=Path, default=None)
    low_lou.add_argument("--resolution", type=int, default=None)
    low_lou.add_argument("--polarity", choices=["positive", "negative"], default=None)
    low_lou.add_argument("--print-resolved-config", action="store_true")
    add_solver_args(low_lou, periodic_horizontal=False)
    low_lou.set_defaults(func=run_low_lou)

    sharp = sub.add_parser("sharp", help="run from SHARP CEA FITS components")
    sharp.add_argument("--config", type=Path, default=None)
    sharp.add_argument("--data-dir", type=Path, default=None)
    sharp.add_argument("--output", type=Path, default=None)
    sharp.add_argument(
        "--polarity", choices=["positive", "negative", "both"], default=None
    )
    sharp.add_argument("--bin", type=int, default=None)
    sharp.add_argument(
        "--crop", type=int, nargs=4, default=None, metavar=("X0", "X1", "Y0", "Y1")
    )
    sharp.add_argument("--nz", type=int, default=None)
    sharp.add_argument("--dz", type=float, default=None)
    sharp.add_argument("--self-consistency-cycles", type=int, default=None)
    sharp.add_argument("--weak-bz-fraction", type=float, default=None)
    sharp.add_argument("--print-resolved-config", action="store_true")
    add_solver_args(sharp)
    sharp.set_defaults(func=run_sharp)

    parsed = parser.parse_args()
    config_fields = {definition.name for definition in fields(GradRubinRunConfig)}
    overrides = {
        name: value for name, value in vars(parsed).items() if name in config_fields
    }
    if parsed.command == "low-lou":
        defaults = GradRubinRunConfig(
            output=Path("runs/cfit_low_lou"), polarity="positive"
        )
    else:
        defaults = GradRubinRunConfig()
    try:
        resolved = load_config(
            type(defaults),
            defaults=defaults,
            config_path=parsed.config,
            env_prefix="SOLAR_GRAD_RUBIN_",
            env_fields=frozenset(
                definition.name for definition in fields(GradRubinRunConfig)
            ),
            cli_overrides=overrides,
        )
    except ConfigError as error:
        parser.error(str(error))

    if parsed.command == "sharp" and resolved.data_dir is None:
        parser.error("sharp data_dir is required after configuration resolution")
    if parsed.print_resolved_config:
        print(json.dumps(public_config(resolved), indent=2, sort_keys=True))
        return
    namespace = argparse.Namespace(**asdict(resolved))
    parsed.func(namespace)


if __name__ == "__main__":
    main()
