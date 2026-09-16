"""Compact on-disk connectome: plain .npy arrays + manifest.json.

The manifest is the loader's only contract: every array is located, typed and checksummed
through it (never by file order or hard-coded offsets). Arrays are memory-mapped by default,
so loading costs almost no RSS until pages are touched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .. import paths
from ..telemetry import MemoryBudget

FORMAT_VERSION = 1


def processed_dir(dataset: str) -> Path:
    return paths.data_dir() / "processed" / dataset


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def smallest_uint(max_value: int) -> np.dtype:
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise OverflowError(max_value)


class StoreWriter:
    """Writes arrays into `<dir>.tmp/`, then swaps the finished store into place."""

    def __init__(self, final_dir: Path):
        self.final = final_dir
        self.tmp = final_dir.with_name(final_dir.name + ".tmp")
        shutil.rmtree(self.tmp, ignore_errors=True)
        (self.tmp / "arrays").mkdir(parents=True)
        self.arrays: dict[str, dict] = {}
        self.vocab: dict[str, list[str]] = {}

    def add(self, name: str, array: np.ndarray, description: str) -> None:
        array = np.ascontiguousarray(array)
        np.save(self.tmp / "arrays" / f"{name}.npy", array)
        self.arrays[name] = {"file": f"arrays/{name}.npy", "dtype": array.dtype.str, "shape": list(array.shape),
                             "bytes": int(array.nbytes), "description": description}

    def add_vocab(self, name: str, values: list[str]) -> None:
        self.vocab[name] = list(values)

    def commit(self, manifest: dict) -> Path:
        for meta in self.arrays.values():
            meta["sha256"] = sha256_file(self.tmp / meta["file"])
        manifest = {"format_version": FORMAT_VERSION, **manifest, "arrays": self.arrays, "vocab": self.vocab}
        (self.tmp / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        old = self.final.with_name(self.final.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if self.final.exists():
            self.final.rename(old)
        self.tmp.rename(self.final)
        shutil.rmtree(old, ignore_errors=True)
        return self.final


class Connectome:
    """Neuron-level directed graph (CSR out-edges + CSC in-edges) with annotations.

    Neuron index i is the position of its root id in ascending order. Edge e is a position in the
    CSR arrays (sorted by pre, then post); `in_edge` maps CSC slots back to CSR edge ids, so edge
    attributes (synapse counts, transmitter probabilities) are stored once.
    """

    def __init__(self, root: Path, manifest: dict, arrays: dict[str, np.ndarray]):
        self.root = root
        self.manifest = manifest
        self.arrays = arrays

    @classmethod
    def load(cls, dataset: str | Path = "flywire_fafb_v783", *, mmap: bool = True, verify: bool = False,
             only: list[str] | None = None, budget: MemoryBudget | None = None) -> "Connectome":
        root = Path(dataset) if Path(dataset).is_dir() else processed_dir(str(dataset))
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{root}: store format {manifest.get('format_version')} != {FORMAT_VERSION}; rerun preprocess")
        names = only or list(manifest["arrays"])
        if budget and not mmap:
            budget.check("load connectome arrays into RAM", sum(manifest["arrays"][n]["bytes"] for n in names))
        arrays = {}
        for name in names:
            meta = manifest["arrays"][name]
            path = root / meta["file"]
            if verify and sha256_file(path) != meta["sha256"]:
                raise ValueError(f"{path}: sha256 does not match the manifest")
            array = np.load(path, mmap_mode="r" if mmap else None)
            if array.dtype.str != meta["dtype"] or list(array.shape) != meta["shape"]:
                raise ValueError(f"{path}: {array.dtype.str}{array.shape} != manifest {meta['dtype']}{meta['shape']}")
            arrays[name] = array
        return cls(root, manifest, arrays)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    @property
    def n_neurons(self) -> int:
        return int(self.manifest["counts"]["neurons"])

    @property
    def n_edges(self) -> int:
        return int(self.manifest["counts"]["edges"])

    def vocab(self, name: str) -> list[str]:
        return self.manifest["vocab"][name]

    def index_of(self, root_ids) -> np.ndarray:
        ids = np.atleast_1d(np.asarray(root_ids, dtype=np.int64))
        root = self["root_id"]
        pos = np.minimum(np.searchsorted(root, ids), len(root) - 1)
        if not np.all(root[pos] == ids):
            raise KeyError(f"unknown root id(s): {ids[root[pos] != ids][:5].tolist()}")
        return pos

    def labels(self, column: str, index=None) -> np.ndarray:
        """Decoded annotation strings ('' = missing) for all neurons or the given indices."""
        codes = self[f"ann_{column}"]
        vocab = np.array(self.vocab(column), dtype=object)
        return vocab[codes if index is None else codes[index]]

    def out_degree(self) -> np.ndarray:
        return np.diff(self["out_indptr"])

    def in_degree(self) -> np.ndarray:
        return np.diff(self["in_indptr"])

    def out_synapses(self) -> np.ndarray:
        cum = np.concatenate([[0], np.cumsum(self["syn_count"], dtype=np.int64)])
        indptr = self["out_indptr"]
        return cum[indptr[1:]] - cum[indptr[:-1]]

    def in_synapses(self) -> np.ndarray:
        return np.bincount(self["out_indices"], weights=self["syn_count"], minlength=self.n_neurons).astype(np.int64)

    def edge_sources(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_neurons, dtype=np.int32), self.out_degree())

    def adjacency(self, min_synapses: int = 1, drop_self_loops: bool = False) -> sp.csr_array:
        """N x N CSR matrix of synapse counts, keeping edges with >= min_synapses."""
        syn = self["syn_count"]
        keep = syn >= min_synapses
        if drop_self_loops:
            keep &= self.edge_sources() != self["out_indices"]
        indptr = np.concatenate([[0], np.cumsum(keep, dtype=np.int64)])[self["out_indptr"]]
        return sp.csr_array((syn[keep].astype(np.int32), self["out_indices"][keep], indptr),
                            shape=(self.n_neurons, self.n_neurons))
