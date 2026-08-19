"""Hatchling build hook for the pinned, locally patched flhtools checkout."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from runpy import run_path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Must match the ``third_party/flhtools`` submodule gitlink. Bumping the pin
# means bumping this constant so image cache keys also change.
FLHTOOLS_REV = "67972f9fedd37e4ccf380a327173093abe9cbdd6"


_SUPPORT = run_path(Path(__file__).resolve().parents[1] / "_build_support.py")
_patches = _SUPPORT["_patches"]
_source = _SUPPORT["_source"]
_verify_pin = _SUPPORT["_verify_pin"]


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        source = _source("flhtools")
        source_file = source / "flhcart.py"
        license_file = source / "LICENSE"
        (patch,) = _patches("flhtools")
        for path in (source_file, license_file, patch):
            if not path.is_file():
                raise RuntimeError(f"missing flhtools build input: {path}")
        _verify_pin("flhtools", FLHTOOLS_REV)

        workdir = Path(tempfile.mkdtemp(prefix="flhtools-build-"))
        self._workdir = workdir
        try:
            shutil.copy2(source_file, workdir / source_file.name)
            shutil.copy2(license_file, workdir / license_file.name)
            subprocess.run(("git", "apply", str(patch)), cwd=workdir, check=True)
            build_data.setdefault("force_include", {}).update(
                {
                    str(workdir / "flhcart.py"): "flhtools/flhcart.py",
                    str(workdir / "LICENSE"): "flhtools/LICENSE",
                }
            )
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        shutil.rmtree(self._workdir, ignore_errors=True)
