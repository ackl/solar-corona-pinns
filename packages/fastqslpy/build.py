"""Hatchling build hook for the pinned, locally patched FastQSL checkout."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from runpy import run_path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Must match the ``third_party/FastQSL`` submodule gitlink. Bumping the pin
# means bumping this constant so image cache keys also change.
FASTQSL_REV = "ef98356ce6615323da51f5a3d95bd88427f2a9ad"

_SUPPORT = run_path(Path(__file__).resolve().parents[1] / "_build_support.py")
_patches = _SUPPORT["_patches"]
_source = _SUPPORT["_source"]
_verify_pin = _SUPPORT["_verify_pin"]


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        source = _source("FastQSL")
        package_source = source / "fastqslpy"
        license_file = source / "LICENSE"
        patches = _patches("fastqslpy")
        for path in (package_source, license_file, *patches):
            if not path.exists():
                raise RuntimeError(f"missing FastQSL build input: {path}")
        _verify_pin("FastQSL", FASTQSL_REV)

        workdir = Path(tempfile.mkdtemp(prefix="fastqsl-build-"))
        self._workdir = workdir
        try:
            package = workdir / "fastqslpy"
            shutil.copytree(package_source, package)
            shutil.copy2(license_file, package / "LICENSE")
            for patch in patches:
                subprocess.run(
                    ("git", "apply", "--ignore-space-change", str(patch)),
                    cwd=workdir,
                    check=True,
                )
            build_data.setdefault("force_include", {}).update(
                {
                    str(path): path.relative_to(workdir).as_posix()
                    for path in package.rglob("*")
                    if path.is_file()
                }
            )
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        shutil.rmtree(self._workdir, ignore_errors=True)
