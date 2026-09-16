"""Project locations and the optional .env file (no secrets are needed for Milestone 1)."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """BIOBRAIN_ROOT, else the nearest ancestor of this package that holds pyproject.toml."""
    if env := os.environ.get("BIOBRAIN_ROOT"):
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env without overriding variables already set."""
    path = path or project_root() / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def data_dir() -> Path:
    return Path(os.environ.get("BIOBRAIN_DATA_DIR", project_root() / "data"))


def results_dir() -> Path:
    return Path(os.environ.get("BIOBRAIN_RESULTS_DIR", project_root() / "results"))


def catalog_dir() -> Path:
    return project_root() / "catalog"
