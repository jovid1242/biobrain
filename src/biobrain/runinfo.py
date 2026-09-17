"""Provenance attached to every result file: code version, environment, hardware, time."""

from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
import sys
from importlib import metadata

import psutil

from . import __version__, paths

_PACKAGES = ("numpy", "scipy", "pyarrow", "python-igraph", "matplotlib", "psutil", "numba", "llvmlite")


def _run(*cmd: str) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=paths.project_root(), capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _relative(arg: str) -> str:
    """Paths inside the project are recorded relative to it, so results carry no machine-specific home directory."""
    root = str(paths.project_root()) + os.sep
    return arg[len(root):] if arg.startswith(root) else arg


def collect(**extra) -> dict:
    commit = _run("git", "rev-parse", "HEAD")
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    cpu = _run("sysctl", "-n", "machdep.cpu.brand_string") if sys.platform == "darwin" else None
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "biobrain_version": __version__,
        "git_commit": commit,
        # dirty = code or data definitions differ from the commit (regenerated results/docs do not count)
        "git_dirty": bool(_run("git", "status", "--porcelain", "--", "src", "catalog", "pyproject.toml",
                               "requirements.lock")) if commit else None,
        "command": [_relative(arg) for arg in sys.argv],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu or platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "packages": versions,
        **extra,
    }
