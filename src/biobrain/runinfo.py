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

_PACKAGES = ("numpy", "scipy", "pyarrow", "python-igraph", "matplotlib", "psutil")


def _run(*cmd: str) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=paths.project_root(), capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


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
        "git_dirty": bool(_run("git", "status", "--porcelain")) if commit else None,
        "command": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu or platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "packages": versions,
        **extra,
    }
