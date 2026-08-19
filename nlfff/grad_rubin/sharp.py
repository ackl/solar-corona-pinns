import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import numpy as np

# set for forgejo runner env
_sunpy_config = Path(gettempdir()) / "sunpy"
_sunpy_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SUNPY_CONFIGDIR", str(_sunpy_config))
_matplotlib_config = Path(gettempdir()) / "matplotlib"
_matplotlib_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_config))

SEGMENTS = ("Br", "Bt", "Bp", "Br_err", "Bt_err", "Bp_err")


@dataclass(frozen=True)
class SharpBoundary:
    field: np.ndarray
    errors: np.ndarray | None
    spacing_Mm: tuple[float, float]
    files: dict[str, Path]


def resolve_sharp_files(data_dir):
    data_dir = Path(data_dir)
    files = {}
    for segment in SEGMENTS:
        matches = sorted(data_dir.glob(f"*.{segment}.fits"))
        if len(matches) > 1:
            raise ValueError(f"expected at most one {segment} FITS file in {data_dir}")
        if matches:
            files[segment] = matches[0]
    missing = [name for name in ("Br", "Bt", "Bp") if name not in files]
    if missing:
        raise FileNotFoundError("Missing SHARP FITS segments: " + ", ".join(missing))
    return files


def load_sharp_boundary(data_dir, bin_factor=1, crop=None):
    """Ingest SHARP CEA data

    NF2 defines ``(Bx, By, Bz) = (Bp, -Bt, Br)``
    """
    from nf2.loader.cartesian_datasets import SHARPDataset

    files = resolve_sharp_files(data_dir)
    have_errors = all(name in files for name in ("Br_err", "Bt_err", "Bp_err"))
    fits_path = {name: str(files[name]) for name in ("Br", "Bt", "Bp")}
    error_path = (
        {name: str(files[name]) for name in ("Br_err", "Bt_err", "Bp_err")}
        if have_errors
        else None
    )

    with TemporaryDirectory(prefix="cfit_nf2_sharp_") as work_directory:
        arguments = {
            "fits_path": fits_path,
            "error_path": error_path,
            "bin": bin_factor,
            "slice": crop,
            "Mm_per_ds": 1,
            "batch_size": 2**30,
            "shuffle": False,
            "filter_nans": False,
            "plot": False,
            "work_path": work_directory,
            "Gauss_per_dB": 1,
        }
        dataset = SHARPDataset(**arguments)
        paths = dataset.file_paths
        shape = (*dataset.cube_shape, 3)
        field = np.load(paths["b_true"]).reshape(shape).astype(np.float32)
        errors = None
        if "b_err" in paths:
            errors = np.load(paths["b_err"]).reshape(shape).astype(np.float32)
        # ``FITSDataset`` preprocesses the maps by ``bin_factor`` before
        # ``MapDataset`` sees them. Passing ``bin`` through as well would bin
        # the values twice, so the loader intentionally omits it below.
        # we recover the physical pixel scale explicitly here.
        # should probs create PR for that bug
        spacing = (
            float(dataset.ds_per_pixel) * bin_factor,
            float(dataset.ds_per_pixel) * bin_factor,
        )

    field = np.nan_to_num(field)
    if errors is not None:
        errors = np.nan_to_num(errors, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return SharpBoundary(
        field=field,
        errors=errors,
        spacing_Mm=spacing,
        files=files,
    )
