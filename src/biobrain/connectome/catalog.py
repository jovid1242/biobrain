"""Dataset catalogs (`catalog/<id>.toml`): pinned file lists with official checksums."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .. import paths


@dataclass(frozen=True)
class FileEntry:
    id: str
    filename: str
    url: str
    size: int
    purpose: str
    format: str = ""
    subdir: str = ""
    required: bool = True
    large: bool = False
    md5: str | None = None
    git_blob_sha1: str | None = None


@dataclass(frozen=True)
class Catalog:
    id: str
    title: str
    version: str
    license: str
    citations: tuple[str, ...]
    files: tuple[FileEntry, ...]
    path: Path

    def file(self, file_id: str) -> FileEntry:
        for entry in self.files:
            if entry.id == file_id:
                return entry
        raise KeyError(f"{self.id}: no file with id {file_id!r}")

    @property
    def raw_dir(self) -> Path:
        return paths.data_dir() / "raw" / self.id

    def local_path(self, entry: FileEntry | str) -> Path:
        entry = self.file(entry) if isinstance(entry, str) else entry
        return self.raw_dir / entry.subdir / entry.filename

    @property
    def lock_path(self) -> Path:
        return self.path.with_suffix(".lock.json")

    def load_lock(self) -> dict:
        return json.loads(self.lock_path.read_text()) if self.lock_path.is_file() else {}

    def write_lock(self, lock: dict) -> None:
        self.lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def load_catalog(dataset: str | Path) -> Catalog:
    path = Path(dataset) if str(dataset).endswith(".toml") else paths.catalog_dir() / f"{dataset}.toml"
    raw = tomllib.loads(path.read_text())
    meta = raw["dataset"]
    files = tuple(FileEntry(**entry) for entry in raw["files"])
    ids = [f.id for f in files]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate file ids")
    return Catalog(meta["id"], meta["title"], meta["version"], meta["license"],
                   tuple(meta.get("citations", ())), files, path)


def list_catalogs() -> list[str]:
    return sorted(p.stem for p in paths.catalog_dir().glob("*.toml"))
