"""Shared plumbing for wheels built from pinned repository submodules."""

from __future__ import annotations

import subprocess
from pathlib import Path

_PACKAGES_ROOT = Path(__file__).resolve().parent


def _source(name: str) -> Path:
    return _PACKAGES_ROOT.parent / "third_party" / name


def _patches(name: str) -> tuple[Path, ...]:
    return tuple(sorted((_PACKAGES_ROOT / name / "patches").glob("*.patch")))


def _has_git_metadata(source: Path) -> bool:
    git = source / ".git"
    if git.is_dir():
        return True
    if not git.is_file():
        return False

    prefix = "gitdir: "
    contents = git.read_text().strip()
    if not contents.startswith(prefix):
        return True
    return (source / contents.removeprefix(prefix)).is_dir()


def _verify_pin(name: str, revision: str) -> None:
    source = _source(name)
    # Docker may copy a submodule's .git file without the superproject metadata
    # it points to. The gitlink and matching revision constant provide the pin
    # for that exported source tree.
    if not _has_git_metadata(source):
        return
    actual = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=source,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    submodule = f"third_party/{name}"
    if actual != revision:
        raise RuntimeError(
            f"{submodule} is at {actual}, expected pinned {revision}; "
            f"run `git submodule update --init {submodule}`"
        )
    dirty = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=source,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"{submodule} contains uncommitted changes")
