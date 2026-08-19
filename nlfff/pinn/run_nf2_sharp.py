"""NF2 SHARP CEA extrapolation, refactored from the Colab notebook for HPC use.

Runs in three stages so the internet-bound work (JSOC download, W&B sync) stays
on a login node and the GPU training runs inside an offline SLURM job:

    nf2-sharp --stage download   # login node (needs internet)
    nf2-sharp --stage train      # compute node (GPU, offline)
    nf2-sharp --stage export     # login or compute node
    nf2-sharp --stage all        # everything in one process

Authored YAML is the primary configuration. Slurm may override allowlisted
fields through the `SOLAR_NF2_` environment namespace.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from nlfff.config import ConfigError, load_config, public_config


@dataclass(frozen=True)
class SharpConfig:
    """Public, authored NF2 SHARP settings; JSOC_EMAIL remains runtime-only."""

    stage: str = "all"
    run_dir: Path = field(default_factory=lambda: Path.cwd() / "runs" / "sharp_cea")
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "sharp")
    sharp_num: int = 377
    noaa_num: int | None = None
    t_start: str = "2011-02-15T00:00:00"
    t_end: str = "2011-02-15T00:12:00"
    cadence: str = "720s"
    series: str = "sharp_cea_720s"
    segments: list[str] = field(
        default_factory=lambda: ["Br", "Bp", "Bt", "Br_err", "Bp_err", "Bt_err"]
    )
    model_field: str = "b"
    model_dim: int = 128
    force_free_weight: float = 1.0e-3
    divergence_weight: float = 1.0e-4
    epochs: int = 15
    iterations: int = 10_000
    sampler_batch_size: int = 8192
    boundary_batch_size: int = 4096
    validation_batch_size: int = 4096
    validation_pixel_per_ds: int = 64
    z_range: list[int] = field(default_factory=lambda: [0, 80])
    export_mm_per_pixel: float = 0.72
    height_range: list[int] = field(default_factory=lambda: [0, 80])
    wandb_mode: str = "offline"

    def __post_init__(self) -> None:
        if self.stage not in {"download", "train", "export", "all"}:
            raise ValueError(f"invalid stage: {self.stage!r}")
        if self.model_field not in {"b", "vector_potential"}:
            raise ValueError(f"invalid model_field: {self.model_field!r}")
        if self.model_dim <= 0 or self.epochs <= 0 or self.iterations <= 0:
            raise ValueError("model_dim, epochs, and iterations must be positive")
        if len(self.z_range) != 2 or len(self.height_range) != 2:
            raise ValueError("z_range and height_range must each contain two values")


SHARP_ENV_FIELDS = frozenset(
    name for name in SharpConfig.__dataclass_fields__ if name != "stage"
)


def _apply_config(config: SharpConfig) -> None:
    global RUN_DIR, DATA_DIR, WORK_DIR, EXPORT_DIR, NF2_PATH
    global SHARP_NUM, NOAA_NUM, T_START, T_END, CADENCE, SERIES, SEGMENTS
    global \
        MODEL_FIELD, \
        MODEL_DIM, \
        FORCE_FREE_WEIGHT, \
        DIVERGENCE_WEIGHT, \
        EPOCHS, \
        ITERATIONS
    global SAMPLER_BATCH_SIZE, BOUNDARY_BATCH_SIZE, VALIDATION_BATCH_SIZE
    global \
        VALIDATION_PIXEL_PER_DS, \
        Z_RANGE, \
        EXPORT_MM_PER_PIXEL, \
        HEIGHT_RANGE, \
        WANDB_MODE
    RUN_DIR, DATA_DIR = config.run_dir.resolve(), config.data_dir.resolve()
    WORK_DIR, EXPORT_DIR = RUN_DIR / "work", RUN_DIR / "exports"
    NF2_PATH = RUN_DIR / "extrapolation_result.nf2"
    SHARP_NUM, NOAA_NUM = config.sharp_num, config.noaa_num
    T_START, T_END, CADENCE, SERIES = (
        config.t_start,
        config.t_end,
        config.cadence,
        config.series,
    )
    SEGMENTS = ",".join(config.segments)
    MODEL_FIELD, MODEL_DIM = config.model_field, config.model_dim
    FORCE_FREE_WEIGHT, DIVERGENCE_WEIGHT = (
        config.force_free_weight,
        config.divergence_weight,
    )
    EPOCHS, ITERATIONS = config.epochs, config.iterations
    SAMPLER_BATCH_SIZE, BOUNDARY_BATCH_SIZE = (
        config.sampler_batch_size,
        config.boundary_batch_size,
    )
    VALIDATION_BATCH_SIZE, VALIDATION_PIXEL_PER_DS = (
        config.validation_batch_size,
        config.validation_pixel_per_ds,
    )
    Z_RANGE, EXPORT_MM_PER_PIXEL, HEIGHT_RANGE = (
        config.z_range,
        config.export_mm_per_pixel,
        config.height_range,
    )
    WANDB_MODE = config.wandb_mode
    os.environ["WANDB_MODE"] = WANDB_MODE
    os.environ.setdefault("WANDB_CONSOLE", "off")


_apply_config(SharpConfig())

# JSOC credentials are deliberately neither authored nor serialized.
JSOC_EMAIL = os.environ.get("JSOC_EMAIL", "you@example.org")

# Exact-segment FITS filenames (filled by resolve_segment_files()).
FIELD_KEYS = ["Br", "Bt", "Bp", "Br_err", "Bt_err", "Bp_err"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def resolve_segment_files(require_errors=True):
    """Match JSOC-style filenames exactly (so Br_err.fits != Br.fits)."""
    files = {}
    for key in FIELD_KEYS:
        matches = sorted(DATA_DIR.glob(f"*.{key}.fits"))
        if len(matches) > 1:
            raise ValueError(f"expected at most one {key} FITS file in {DATA_DIR}")
        if matches:
            files[key] = matches[0]
    required = ["Br", "Bt", "Bp"] + (
        ["Br_err", "Bt_err", "Bp_err"] if require_errors else []
    )
    missing = [key for key in required if key not in files]
    if missing:
        raise FileNotFoundError("Missing required FITS segments: " + ", ".join(missing))
    return files


# ----------------------------------------------------------------------------
# Stage: download  (login node -- needs internet)
# ----------------------------------------------------------------------------


def stage_download():
    import nf2
    from dateutil.parser import parse

    if JSOC_EMAIL == "you@example.org":
        raise SystemExit(
            "Set JSOC_EMAIL to your JSOC-registered email (export JSOC_EMAIL=...)."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] SHARP {SHARP_NUM}  {T_START} .. {T_END}  segments={SEGMENTS}")
    nf2.download_sharp_series(
        str(DATA_DIR),
        JSOC_EMAIL,
        parse(T_START),
        parse(T_END),
        noaa_num=NOAA_NUM,
        sharp_num=SHARP_NUM,
        cadence=CADENCE,
        segments=SEGMENTS,
        series=SERIES,
    )
    # Require errors only if they were requested.
    want_errors = all(f"{c}_err" in SEGMENTS for c in ("Br", "Bt", "Bp"))
    files = resolve_segment_files(require_errors=want_errors)
    print("[download] resolved files:")
    for key, path in files.items():
        print(f"    {key:8s} {path}")


# ----------------------------------------------------------------------------
# Stage: train  (compute node -- GPU, offline)
# ----------------------------------------------------------------------------


def build_config(files, use_errors):
    losses = [
        {"type": "boundary", "name": "boundary", "weight": 1.0, "datasets": "boundary"},
        {
            "type": "boundary",
            "name": "potential_boundary",
            "weight": 10.0,
            "datasets": "potential",
        },
        {
            "type": "force_free",
            "name": "force_free",
            "weight": FORCE_FREE_WEIGHT,
            "datasets": ["random"],
        },
    ]
    if MODEL_FIELD == "b":
        losses.append(
            {
                "type": "divergence",
                "name": "divergence",
                "weight": DIVERGENCE_WEIGHT,
                "datasets": ["random"],
            }
        )

    loss_scaling = [
        {
            "type": "b_height",
            "name": "b_height",
            "loss_ids": [l["name"] for l in losses if l["type"] != "boundary"],
        }
    ]

    field_files = {
        "Br": str(files["Br"]),
        "Bt": str(files["Bt"]),
        "Bp": str(files["Bp"]),
    }
    boundary = {
        "id": "boundary",
        "type": "sharp",
        "files": field_files,
        "batch_size": BOUNDARY_BATCH_SIZE,
    }
    if use_errors:
        boundary["errors"] = {
            "Br_err": str(files["Br_err"]),
            "Bt_err": str(files["Bt_err"]),
            "Bp_err": str(files["Bp_err"]),
        }
    val_boundary = {**boundary, "batch_size": VALIDATION_BATCH_SIZE}

    return {
        "path": str(RUN_DIR),
        "work_path": str(WORK_DIR),
        "logging": {
            "project": "nf2",
            "name": f"SHARP CEA {MODEL_FIELD} dim{MODEL_DIM}",
        },
        "model": {"field": MODEL_FIELD, "network": {"hidden_dim": MODEL_DIM}},
        "data": {
            "geometry": "cartesian",
            "boundaries": [boundary],
            "sampler": {"type": "height", "batch_size": SAMPLER_BATCH_SIZE},
            "potential_boundary": {"type": "potential", "strides": 4},
            "validation": [
                val_boundary,
                {"id": "cube", "type": "cube", "batch_size": VALIDATION_BATCH_SIZE},
                {
                    "id": "slices",
                    "type": "slices",
                    "n_slices": 8,
                    "batch_size": VALIDATION_BATCH_SIZE,
                },
            ],
            "iterations": ITERATIONS,
            "z_range": Z_RANGE,
            "validation_pixel_per_ds": VALIDATION_PIXEL_PER_DS,
        },
        "training": {"epochs": EPOCHS},
        "losses": losses,
        "loss_scaling": loss_scaling,
        "callbacks": [
            {"type": "boundary", "dataset": "boundary"},
            {"type": "metrics", "dataset": "cube"},
            {"type": "slices", "dataset": "slices"},
        ],
    }


def stage_train():
    from importlib.metadata import version

    import nf2
    import torch
    import wandb

    # Fail fast: do not burn a queue slot if the GPU isn't visible.
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available -- check the GPU allocation / --gpus and nvidia-smi."
        )
    print(f"[train] NF2 {version('nf2')}  Torch {torch.__version__}")
    for i in range(torch.cuda.device_count()):
        print(f"[train] CUDA device {i}: {torch.cuda.get_device_name(i)}")

    files = resolve_segment_files(require_errors=False)
    use_errors = all(f"{component}_err" in files for component in ("Br", "Bt", "Bp"))
    print(f"[train] using uncertainty maps: {use_errors}")

    config = build_config(files, use_errors)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    nf2.run(**config)
    wandb.finish()

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[train] peak GPU memory: {peak:.2f} GiB")
    print(f"[train] wrote {NF2_PATH}")


# ----------------------------------------------------------------------------
# Stage: export  (login or compute node)
# ----------------------------------------------------------------------------


def stage_export():
    import matplotlib

    matplotlib.use("Agg")  # headless: no display on compute/login nodes
    import matplotlib.pyplot as plt
    import nf2
    import numpy as np
    from astropy import units as u
    from matplotlib.colors import LogNorm
    from nf2.evaluation.metric import vector_norm

    from nlfff.grad_rubin.grid import CartesianGrid
    from nlfff.visualization import plot_height_integrated_current
    from studies.lib.metrics import physical_consistency_metrics

    if not NF2_PATH.exists():
        raise SystemExit(f"No model at {NF2_PATH} -- run --stage train first.")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    common = {
        "Mm_per_pixel": EXPORT_MM_PER_PIXEL,
        "height_range": HEIGHT_RANGE,
        "metrics": ["j", "alpha", "free_energy_fft"],
    }
    nf2.export_file(str(NF2_PATH), str(EXPORT_DIR / "field.vtk"), fmt="vtk", **common)
    nf2.export_file(str(NF2_PATH), str(EXPORT_DIR / "field.npz"), fmt="npz", **common)

    out = nf2.load(NF2_PATH)
    cube = out.load_cube(
        Mm_per_pixel=EXPORT_MM_PER_PIXEL,
        height_range=HEIGHT_RANGE,
        metrics=["j", "free_energy_fft"],
        progress=True,
    )

    b = cube["b"].to_value(u.G)
    j_current = cube["metrics"]["j"].to_value(u.G / u.s)
    j = vector_norm(j_current)
    free_energy = cube["metrics"]["free_energy_fft"].to_value(u.erg / u.cm**3)
    voxel_volume_cm3 = (EXPORT_MM_PER_PIXEL * u.Mm).to_value(u.cm) ** 3
    grid = CartesianGrid(
        shape=b.shape[:3],
        spacing=(EXPORT_MM_PER_PIXEL,) * 3,
    )
    metrics = {
        **physical_consistency_metrics(b, grid),
        "mean_B_norm": float(np.nanmean(vector_norm(b))),
        "mean_J_norm": float(np.nanmean(j)),
        "total_free_energy_erg": float(np.nansum(free_energy) * voxel_volume_cm3),
    }
    print("[export] metrics:")
    for k, v in metrics.items():
        print(f"    {k:28s} {v:.6g}")
    if metrics["theta_J_interior_deg"] > 20:
        print(
            "[export] NOTE theta_J_interior_deg > 20: "
            "consider a larger FORCE_FREE_WEIGHT."
        )

    # Diagnostic figure (saved, not shown).
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    free_energy_map = np.nansum(free_energy, axis=2) * (
        EXPORT_MM_PER_PIXEL * u.Mm
    ).to_value(u.cm)
    boundary_bz = b[:, :, 0, 2]
    extent = [0, b.shape[0] * EXPORT_MM_PER_PIXEL, 0, b.shape[1] * EXPORT_MM_PER_PIXEL]

    def log_norm(image, lower=1, upper=99):
        positive = image[np.isfinite(image) & (image > 0)]
        if positive.size == 0:
            return LogNorm(vmin=np.nextafter(0, 1), vmax=1.0)
        vmin, vmax = np.nanpercentile(positive, [lower, upper])
        return LogNorm(vmin=max(vmin, np.nextafter(0, 1)), vmax=max(vmax, vmin * 1.01))

    im, _ = plot_height_integrated_current(
        axs[0],
        j_current,
        EXPORT_MM_PER_PIXEL,
        extent=extent,
        title="Height-integrated |J| (log)",
    )
    plt.colorbar(im, ax=axs[0], fraction=0.046, label="|J| [G Mm s$^{-1}$]")
    im = axs[1].imshow(
        free_energy_map.T,
        origin="lower",
        cmap="jet",
        norm=log_norm(free_energy_map),
        extent=extent,
    )
    axs[1].set_title("Height-integrated free energy (log)")
    plt.colorbar(im, ax=axs[1], fraction=0.046, label="free energy [erg cm$^{-2}$]")
    lim = np.nanpercentile(np.abs(boundary_bz), 99)
    im = axs[2].imshow(
        boundary_bz.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=extent
    )
    axs[2].set_title("Model boundary Bz")
    plt.colorbar(im, ax=axs[2], fraction=0.046, label="$B_z$ [G]")
    for ax in axs:
        ax.set_xlabel("x [Mm]")
        ax.set_ylabel("y [Mm]")
    fig.tight_layout()
    fig_path = EXPORT_DIR / "quicklook.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[export] wrote {fig_path}")
    print(f"[export] VTK/NPZ in {EXPORT_DIR}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", type=Path, default=None, help="authored YAML configuration"
    )
    ap.add_argument(
        "--stage", choices=["download", "train", "export", "all"], default=None
    )
    ap.add_argument("--model-dim", type=int, default=None)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--print-resolved-config", action="store_true")
    args = ap.parse_args()
    try:
        config = load_config(
            SharpConfig,
            config_path=args.config,
            env_prefix="SOLAR_NF2_",
            env_fields=SHARP_ENV_FIELDS,
            cli_overrides={
                "stage": args.stage,
                "model_dim": args.model_dim,
                "run_dir": args.run_dir,
                "data_dir": args.data_dir,
            },
        )
    except ConfigError as error:
        ap.error(str(error))
    _apply_config(config)
    if args.print_resolved_config:
        print(json.dumps(public_config(config), indent=2, sort_keys=True))
        return

    print(f"[run] stage={config.stage}  RUN_DIR={RUN_DIR}  WANDB_MODE={WANDB_MODE}")
    if config.stage in ("download", "all"):
        stage_download()
    if config.stage in ("train", "all"):
        stage_train()
    if config.stage in ("export", "all"):
        stage_export()


if __name__ == "__main__":
    main()
