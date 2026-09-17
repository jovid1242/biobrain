"""Reproducible downloads: size policy, resume, retries, verification against official checksums.

Per file: if present, hash it and verify; otherwise stream into `<name>.part` (resuming with an
HTTP Range request after a failure), hashing while writing, then check the size and the official
md5 / git blob SHA-1 before renaming. A file that fails verification is kept as `<name>.corrupt`
and the command fails. The sha256 of every verified file goes into the catalog's lock file.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import paths
from ..telemetry import fmt_bytes
from .catalog import Catalog, FileEntry

CHUNK = 1 << 20
DEFAULT_MAX_FILE = 2 << 30
USER_AGENT = "biobrain-research/0.1 (+https://github.com/jovid1242/biobrain)"
RETRY_HTTP = {408, 425, 429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    pass


class Digest:
    """md5 + sha256, plus the git blob SHA-1 when the catalog pins one, fed incrementally."""

    def __init__(self, entry: FileEntry):
        self.md5 = hashlib.md5()
        self.sha256 = hashlib.sha256()
        self.blob = hashlib.sha1(b"blob %d\0" % entry.size) if entry.git_blob_sha1 else None

    def update(self, chunk: bytes) -> None:
        self.md5.update(chunk)
        self.sha256.update(chunk)
        if self.blob:
            self.blob.update(chunk)

    def result(self) -> dict:
        out = {"md5": self.md5.hexdigest(), "sha256": self.sha256.hexdigest()}
        if self.blob:
            out["git_blob_sha1"] = self.blob.hexdigest()
        return out


def hash_file(path: Path, entry: FileEntry) -> dict:
    digest = Digest(entry)
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.result()


def problems(entry: FileEntry, size: int, digests: dict) -> list[str]:
    found = []
    if size != entry.size:
        found.append(f"size {size} != catalog {entry.size}")
    if entry.md5 and digests["md5"] != entry.md5:
        found.append(f"md5 {digests['md5']} != official {entry.md5}")
    if entry.git_blob_sha1 and digests.get("git_blob_sha1") != entry.git_blob_sha1:
        found.append(f"git blob sha1 {digests.get('git_blob_sha1')} != official {entry.git_blob_sha1}")
    return found


def _stream(entry: FileEntry, part: Path, offset: int, digest: Digest, timeout: float, log) -> None:
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(entry.url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        if offset and getattr(resp, "status", None) != 206:
            part.unlink()
            raise ConnectionError("server ignored the Range request; restarting from byte 0")
        with open(part, "ab" if offset else "wb") as out:
            last_t, last_n = time.monotonic(), offset
            while chunk := resp.read(CHUNK):
                out.write(chunk)
                digest.update(chunk)
                offset += len(chunk)
                if (now := time.monotonic()) - last_t >= 10:
                    rate = (offset - last_n) / (now - last_t)
                    log(f"  {entry.id}: {fmt_bytes(offset)} / {fmt_bytes(entry.size)} "
                        f"({100 * offset / entry.size:.1f} %), {fmt_bytes(rate)}/s")
                    last_t, last_n = now, offset
    if offset < entry.size:
        raise ConnectionError(f"connection closed at byte {offset} of {entry.size}")


def fetch(entry: FileEntry, dest: Path, *, retries: int = 8, timeout: float = 60, log=print) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        if offset > entry.size:
            part.unlink()
            offset = 0
        digest = Digest(entry)
        try:
            if offset:
                with open(part, "rb") as fh:
                    while chunk := fh.read(CHUNK):
                        digest.update(chunk)
            if offset < entry.size:
                _stream(entry, part, offset, digest, timeout, log)
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_HTTP or attempt == retries:
                raise DownloadError(f"{entry.id}: HTTP {exc.code} from {entry.url}") from exc
            error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise DownloadError(f"{entry.id}: giving up after {retries} retries: {exc}") from exc
            error = exc
        wait = min(60, 2 ** attempt)
        log(f"  {entry.id}: {error!r}; retry {attempt + 1}/{retries} in {wait} s (resumes where it stopped)")
        time.sleep(wait)
    digests = digest.result()
    if bad := problems(entry, part.stat().st_size, digests):
        corrupt = dest.with_name(dest.name + ".corrupt")
        part.replace(corrupt)
        raise DownloadError(f"{entry.id}: verification failed ({'; '.join(bad)}); kept as {corrupt}")
    part.replace(dest)
    return digests


def select(catalog: Catalog, only: list[str] | None = None, include_optional: bool = False) -> list[FileEntry]:
    if only:
        return [catalog.file(file_id) for file_id in only]
    return [entry for entry in catalog.files if entry.required or include_optional]


def run(catalog: Catalog, *, only: list[str] | None = None, include_optional: bool = False,
        allow_large: bool = False, max_file: int = DEFAULT_MAX_FILE, dry_run: bool = False,
        retries: int = 8, parallel: int = 3, log=print) -> int:
    """Fetch/verify the selected files, up to `parallel` at a time (one thread owns one file). A failing
    file does not stop the others; all failures are reported together at the end (exit code 1)."""
    entries = select(catalog, only, include_optional)
    lock = catalog.load_lock()
    blocked = {e.id for e in entries if (e.large or e.size > max_file) and not allow_large}
    missing = [e for e in entries if e.id not in blocked and not catalog.local_path(e).exists()]

    log(f"dataset  {catalog.id} — {catalog.title}")
    log(f"license  {catalog.license}")
    log(f"target   {catalog.raw_dir}")
    log(f"{'file id':<22}{'size':>12}  {'need':<9}{'state':<24}purpose")
    for e in catalog.files:
        if e not in entries:
            state = "not selected"
        elif e.id in blocked:
            state = "blocked: --allow-large"
        elif catalog.local_path(e).exists():
            state = "present (will verify)"
        else:
            state = "to download"
        log(f"{e.id:<22}{fmt_bytes(e.size):>12}  {'required' if e.required else 'optional':<9}{state:<24}{e.purpose}")
    need = sum(e.size for e in missing)
    free = shutil.disk_usage(paths.data_dir()).free
    log(f"to download: {fmt_bytes(need)} in {len(missing)} file(s); free disk: {fmt_bytes(free)}")
    if dry_run:
        return 0
    if need and free < need * 1.1 + (1 << 30):
        raise DownloadError(f"not enough disk space: need {fmt_bytes(need)} + margin, have {fmt_bytes(free)}")

    failed = {}
    guard = threading.Lock()

    def process(entry: FileEntry) -> None:
        path = catalog.local_path(entry)
        if path.exists():
            digests = hash_file(path, entry)
            bad = problems(entry, path.stat().st_size, digests)
            if (locked := lock.get(entry.id, {}).get("sha256")) and locked != digests["sha256"]:
                bad.append(f"sha256 differs from lock file ({locked})")
            if not bad:
                log(f"ok       {entry.id}: verified ({'md5' if entry.md5 else 'git blob sha1'} + size)")
            else:
                log(f"invalid  {entry.id}: {'; '.join(bad)} — moving aside and downloading again")
                path.replace(path.with_name(path.name + ".corrupt"))
        if not path.exists():
            log(f"fetch    {entry.id}: {entry.url}")
            t0 = time.monotonic()
            try:
                digests = fetch(entry, path, retries=retries, log=log)
            except DownloadError as exc:
                log(f"FAILED   {entry.id}: {exc}")
                with guard:
                    failed[entry.id] = str(exc)
                return
            log(f"ok       {entry.id}: {fmt_bytes(entry.size)} in {time.monotonic() - t0:.0f} s, checksums verified")
        record = {"filename": str(path.relative_to(catalog.raw_dir)), "url": entry.url, "size": entry.size, **digests}
        with guard:
            if {k: v for k, v in lock.get(entry.id, {}).items() if k != "verified_utc"} != record:
                lock[entry.id] = {**record, "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
                catalog.write_lock(lock)

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        list(pool.map(process, [e for e in entries if e.id not in blocked]))
    if blocked:
        log(f"skipped (size policy): {', '.join(sorted(blocked))} — rerun with --allow-large if really needed")
    for file_id, reason in failed.items():
        log(f"not downloaded: {file_id} — {reason} (rerun resumes partial files)")
    return 1 if blocked or failed else 0
