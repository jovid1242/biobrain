import hashlib
import io
import json
import urllib.error

import pytest

from biobrain.connectome import download
from biobrain.connectome.catalog import FileEntry, load_catalog

PAYLOAD = bytes(range(256)) * 4096  # ~1 MiB, crosses the 1 MiB chunk boundary


def entry(url, data=PAYLOAD, **kw):
    return FileEntry(id="f", filename="f.bin", url=url, size=len(data), purpose="test",
                     md5=hashlib.md5(data).hexdigest(), **kw)


class FakeResponse(io.BytesIO):
    def __init__(self, data, status):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_git_blob_sha1_matches_git_hash_object(tmp_path):
    # `printf 'hello\n' | git hash-object --stdin` == ce013625030ba8dba906f756967f9e9ca394464a
    path = tmp_path / "hello"
    path.write_bytes(b"hello\n")
    e = FileEntry(id="h", filename="hello", url="", size=6, purpose="", git_blob_sha1="ce013625030ba8dba906f756967f9e9ca394464a")
    digests = download.hash_file(path, e)
    assert download.problems(e, 6, digests) == []


def test_problems_reports_each_mismatch():
    e = entry("", git_blob_sha1="0" * 40)
    bad = download.problems(e, e.size + 1, {"md5": "x", "sha256": "y", "git_blob_sha1": "z"})
    assert len(bad) == 3


def test_fetch_file_url_verifies_md5(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(PAYLOAD)
    dest = tmp_path / "out" / "f.bin"
    digests = download.fetch(entry(src.as_uri()), dest)
    assert dest.read_bytes() == PAYLOAD
    assert digests["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()


def test_fetch_checksum_mismatch_keeps_corrupt_copy(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(PAYLOAD)
    bad = FileEntry(id="f", filename="f.bin", url=src.as_uri(), size=len(PAYLOAD), purpose="", md5="0" * 32)
    dest = tmp_path / "f.bin"
    with pytest.raises(download.DownloadError, match="md5"):
        download.fetch(bad, dest)
    assert not dest.exists() and (tmp_path / "f.bin.corrupt").exists()


def test_fetch_resumes_with_range(tmp_path, monkeypatch):
    dest = tmp_path / "f.bin"
    half = len(PAYLOAD) // 2
    (tmp_path / "f.bin.part").write_bytes(PAYLOAD[:half])
    seen = []

    def urlopen(request, timeout):
        seen.append(request.get_header("Range"))
        return FakeResponse(PAYLOAD[half:], 206)

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    download.fetch(entry("http://example.invalid/f"), dest)
    assert seen == [f"bytes={half}-"] and dest.read_bytes() == PAYLOAD


def test_fetch_restarts_when_range_ignored_and_retries_5xx(tmp_path, monkeypatch):
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(PAYLOAD[:1000])
    replies = iter([FakeResponse(PAYLOAD, 200),  # Range ignored -> restart from 0
                    urllib.error.HTTPError("u", 503, "busy", {}, None),
                    FakeResponse(PAYLOAD, 200)])

    def urlopen(request, timeout):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    download.fetch(entry("http://example.invalid/f"), dest, log=lambda *_: None)
    assert dest.read_bytes() == PAYLOAD


def test_fetch_does_not_retry_404(tmp_path, monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError("u", 404, "missing", {}, None)

    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    with pytest.raises(download.DownloadError, match="HTTP 404"):
        download.fetch(entry("http://example.invalid/f"), tmp_path / "f.bin")


def write_catalog(tmp_path, small, big):
    toml = tmp_path / "toy.toml"
    toml.write_text(f"""
[dataset]
id = "toy"
title = "toy"
version = "1"
license = "test"

[[files]]
id = "small"
filename = "small.bin"
url = "{small.as_uri()}"
size = {small.stat().st_size}
md5 = "{hashlib.md5(small.read_bytes()).hexdigest()}"
purpose = "test"

[[files]]
id = "big"
filename = "big.bin"
url = "{big.as_uri()}"
size = {big.stat().st_size}
purpose = "test"
large = true
""")
    return load_catalog(toml)


def test_run_size_policy_lock_and_reverify(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOBRAIN_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    small, big = tmp_path / "small.src", tmp_path / "big.src"
    small.write_bytes(b"abc" * 1000)
    big.write_bytes(b"x" * 5000)
    cat = write_catalog(tmp_path, small, big)
    quiet = lambda *_: None

    assert download.run(cat, dry_run=True, log=quiet) == 0
    assert not cat.local_path("small").exists()

    assert download.run(cat, log=quiet) == 1  # "big" is flagged large -> blocked, reported
    assert cat.local_path("small").exists() and not cat.local_path("big").exists()
    lock = json.loads(cat.lock_path.read_text())
    assert lock["small"]["sha256"] == hashlib.sha256(small.read_bytes()).hexdigest()

    def no_network(*a, **k):
        raise AssertionError("a verified file must not be downloaded again")

    monkeypatch.setattr(download.urllib.request, "urlopen", no_network)
    assert download.run(cat, only=["small"], log=quiet) == 0


def test_run_continues_after_a_failed_file(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOBRAIN_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    small, big = tmp_path / "small.src", tmp_path / "big.src"
    small.write_bytes(b"abc")
    big.write_bytes(b"x" * 10)
    cat = write_catalog(tmp_path, small, big)
    small.unlink()  # first file now fails (URLError), the second must still be fetched
    assert download.run(cat, allow_large=True, retries=0, log=lambda *_: None) == 1
    assert cat.local_path("big").exists() and not cat.local_path("small").exists()
