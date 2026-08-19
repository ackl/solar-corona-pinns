"""Hatchling build hook: patch and compile the pinned UFiT checkout.

uv builds this package from the ``third_party/UFiT`` submodule, so a plain
``uv sync`` produces the patched shared library on every platform. The hook
copies the sources to a scratch directory, applies the repository-owned
``ufit-cfit.patch``, and compiles ``UFiT_Python_Callable.so`` with the system
``gfortran`` before hatchling assembles the wheel.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from runpy import run_path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Must match the ``third_party/UFiT`` submodule gitlink. Bumping the pin means
# bumping this constant, which also retriggers image rebuilds.
UFIT_REV = "7570df98810d34e3ba21c3d12c5b7f9ac361cb3f"

_SOURCES = (
    "UFiT_Definitions_Fortran.F90",
    "UFiT_User_Functions.F90",
    "UFiT_Functions_Fortran.F90",
    "UFiT_Python_Callable.F90",
    "UFiT_Functions_Python.py",
)

_SUPPORT = run_path(Path(__file__).resolve().parents[1] / "_build_support.py")
_patches = _SUPPORT["_patches"]
_source = _SUPPORT["_source"]
_verify_pin = _SUPPORT["_verify_pin"]


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def _compiler_args() -> list[str]:
    args = [
        "gfortran",
        "-shared",
        "-fPIC",
        "UFiT_Definitions_Fortran.F90",
        "UFiT_User_Functions.F90",
        "UFiT_Functions_Fortran.F90",
        "UFiT_Python_Callable.F90",
        "-O3",
        "-fopenmp",
    ]
    if sys.platform == "darwin":
        sdk_root = subprocess.run(
            ("xcrun", "--show-sdk-path"),
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        args.append(f"-Wl,-syslibroot,{sdk_root}")
    return args + ["-o", "UFiT_Python_Callable.so"]


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        source = _source("UFiT")
        (patch,) = _patches("ufit")
        license_file = source / "LICENSE"
        inputs = (*(source / name for name in _SOURCES), patch, license_file)
        for path in inputs:
            if not path.is_file():
                raise RuntimeError(f"missing UFiT build input: {path}")
        _verify_pin("UFiT", UFIT_REV)

        workdir = Path(tempfile.mkdtemp(prefix="ufit-build-"))
        self._workdir = workdir
        try:
            for name in _SOURCES:
                shutil.copy2(source / name, workdir / name)
            shutil.copy2(license_file, workdir / "LICENSE")
            _run(["git", "apply", str(patch)], cwd=workdir)
            _run(_compiler_args(), cwd=workdir)
            platform_tag = sysconfig.get_platform()
            if platform_tag.endswith("-universal2"):
                platform_tag = (
                    platform_tag.removesuffix("universal2") + platform.machine()
                )
            platform_tag = platform_tag.replace("-", "_").replace(".", "_")
            build_data["tag"] = f"py3-none-{platform_tag}"
            build_data["pure_python"] = False
            build_data.setdefault("force_include", {}).update(
                {
                    str(
                        workdir / "UFiT_Python_Callable.so"
                    ): "ufit/UFiT_Python_Callable.so",
                    str(
                        workdir / "UFiT_Functions_Python.py"
                    ): "ufit/UFiT_Functions_Python.py",
                    str(workdir / "LICENSE"): "ufit/LICENSE",
                }
            )
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        shutil.rmtree(self._workdir, ignore_errors=True)
