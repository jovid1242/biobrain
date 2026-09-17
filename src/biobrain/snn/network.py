"""Simulation network: a CSR of float weights built from observed synapse counts by explicit, configurable rules.

    weight(e) = sign(e) · gain · f(synapse_count(e)) / mean_over_connectome(f)

f (transform) and the sign table are ASSUMED; the normalisation constant is DERIVED from all 15.1 M
connectome edges, so the same neuron pair gets the same weight in every subgraph; gain is TUNED by the
calibration procedure. Edges whose sign is 0 (e.g. neuromodulators under the default table) are not fast
synapses and are removed from the simulation graph — they are counted in `meta`, not silently lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..connectome.store import Connectome
from .config import SIGN_KEYS, SignSpec, SimConfig, WeightSpec

EDGE_CLASSES = ("gaba", "acetylcholine", "glutamate", "octopamine", "serotonin", "dopamine")  # nt_prob_q column order


def transform(counts: np.ndarray, spec: WeightSpec) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64)
    if spec.transform == "linear":
        return c
    if spec.transform == "sqrt":
        return np.sqrt(c)
    if spec.transform == "log1p":
        return np.log1p(c)
    return np.minimum(c, spec.clip)


@lru_cache(maxsize=32)
def _normalization(root: str, transform_name: str, clip: float, min_synapses: int) -> float:
    conn = Connectome.load(root, only=["syn_count"])
    counts = np.asarray(conn["syn_count"])
    counts = counts[counts >= min_synapses]
    return float(transform(counts, WeightSpec(transform=transform_name, clip=clip)).mean())


def normalization(conn: Connectome, spec: WeightSpec) -> float:
    return _normalization(str(conn.root), spec.transform, spec.clip, spec.min_synapses)


@dataclass
class Network:
    n: int
    indptr: np.ndarray   # int64
    indices: np.ndarray  # int32
    weights: np.ndarray  # float32 / float64
    meta: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return int(self.indices.size)

    def out_degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def memory(self) -> dict:
        return {"topology": int(self.indptr.nbytes + self.indices.nbytes), "synapse_state": int(self.weights.nbytes)}

    @classmethod
    def from_edges(cls, n: int, src, dst, weights, dtype: str = "float64", meta: dict | None = None) -> "Network":
        """Test/null helper: duplicate (src, dst) pairs are summed; rows sorted by target."""
        src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
        w = np.asarray(weights, dtype=np.float64)
        if src.size == 0:
            return cls(n, np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=dtype), meta or {})
        keys, inverse = np.unique(src * n + dst, return_inverse=True)
        summed = np.bincount(inverse, weights=w)
        indptr = np.concatenate([[0], np.cumsum(np.bincount(keys // n, minlength=n), dtype=np.int64)])
        return cls(n, indptr, (keys % n).astype(np.int32), summed.astype(dtype), meta or {})


def signs(conn: Connectome, sub, spec: SignSpec) -> tuple[np.ndarray, dict]:
    """Per-edge sign from the configured table; returns the signs and what they were derived from."""
    table = np.array([spec.value(k) for k in SIGN_KEYS] + [spec.missing])
    neuron_nt = conn.labels("top_nt", sub.nodes)
    code = {k: i for i, k in enumerate(SIGN_KEYS)}
    neuron_code = np.array([code.get(x, len(SIGN_KEYS)) for x in neuron_nt])
    src = sub.sources()
    edge_code = neuron_code[src]
    info = {"source": spec.source}
    if spec.source == "edge":
        q = np.asarray(conn["nt_prob_q"][sub.edge_global])
        predicted = q.sum(axis=1) > 0
        dominant = np.array([code[c] for c in EDGE_CLASSES])[np.argmax(q, axis=1)]
        info["edges_without_prediction_used_neuron_label"] = int((~predicted).sum())
        info["edges_where_edge_class_differs_from_neuron_label"] = int(np.sum(predicted & (dominant != edge_code)))
        edge_code = np.where(predicted, dominant, edge_code)
    labels = np.array(list(SIGN_KEYS) + ["missing"], dtype=object)
    info["edges_by_transmitter"] = {str(labels[i]): int(c) for i, c in enumerate(np.bincount(edge_code, minlength=len(labels))) if c}
    info["neurons_by_transmitter"] = {str(labels[i]): int(c) for i, c in enumerate(np.bincount(neuron_code, minlength=len(labels))) if c}
    return table[edge_code], info


def build(conn: Connectome, sub, cfg: SimConfig) -> Network:
    spec = cfg.weights
    counts = np.asarray(sub.syn_count)
    sign, sign_info = signs(conn, sub, cfg.signs)
    norm = normalization(conn, spec)
    w = sign * spec.gain * transform(counts, spec) / norm
    keep = (counts >= spec.min_synapses) & (w != 0)
    src = sub.sources()
    dtype = cfg.neuron.dtype
    indptr = np.concatenate([[0], np.cumsum(np.bincount(src[keep], minlength=sub.n), dtype=np.int64)])
    weights = w[keep].astype(dtype)
    meta = {
        "subgraph": sub.name, "subgraph_sha256": sub.manifest["sha256"],
        "neurons": sub.n, "subgraph_edges": sub.m, "simulated_edges": int(keep.sum()),
        "removed_below_min_synapses": int(np.sum(counts < spec.min_synapses)),
        "removed_zero_sign": int(np.sum((counts >= spec.min_synapses) & (sign == 0))),
        "excitatory_edges": int(np.sum(weights > 0)), "inhibitory_edges": int(np.sum(weights < 0)),
        "weight_transform": spec.transform, "gain": spec.gain, "normalization": norm,
        "mean_abs_weight": float(np.abs(weights).mean()) if weights.size else 0.0,
        "max_abs_weight": float(np.abs(weights).max()) if weights.size else 0.0,
        "total_excitation": float(weights[weights > 0].sum()), "total_inhibition": float(weights[weights < 0].sum()),
        "signs": sign_info,
    }
    return Network(sub.n, indptr, sub.indices[keep], weights, meta)
