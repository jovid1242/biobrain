"""Connected, reproducible FlyWire subgraphs for simulation, each with a manifest.

Methods:
- expand: greedy expansion from a seed neuron — repeatedly add the neuron with the most synapses (both
  directions) to the already selected set. Prefixes are nested, so 100 ⊂ 1k ⊂ 10k ⊂ 50k.
- neuropil: all neurons whose home neuropil (M1 rule, analysis.regions) is the given neuropil.
Node lists are cached under data/processed/subgraphs/ (not in git); manifests go to results/milestone2/subgraphs/
with sha256 of nodes and edges, so a regenerated subgraph is verified against the recorded one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

from .. import paths, runinfo
from ..connectome.store import Connectome, sha256_file

SUBGRAPH_SEED = 20260917
CANONICAL = {
    "expand_100": {"method": "expand", "size": 100, "seed": SUBGRAPH_SEED},
    "expand_1k": {"method": "expand", "size": 1_000, "seed": SUBGRAPH_SEED},
    "expand_10k": {"method": "expand", "size": 10_000, "seed": SUBGRAPH_SEED},
    "expand_50k": {"method": "expand", "size": 50_000, "seed": SUBGRAPH_SEED},
    "neuropil_PB": {"method": "neuropil", "neuropil": "PB"},
    "neuropil_SPS_R": {"method": "neuropil", "neuropil": "SPS_R"},
    "neuropil_LO_R": {"method": "neuropil", "neuropil": "LO_R"},
    "full": {"method": "full"},  # Milestone 2.5: all 139,255 neurons
}


@dataclass
class Subgraph:
    name: str
    nodes: np.ndarray        # global neuron indices, ascending
    indptr: np.ndarray       # int64, local CSR by presynaptic neuron
    indices: np.ndarray      # int32 local post ids, ascending within a row
    syn_count: np.ndarray    # synapses per edge (OBSERVED)
    edge_global: np.ndarray  # position of each edge in the store CSR (transmitter lookup)
    manifest: dict

    @property
    def n(self) -> int:
        return int(self.nodes.size)

    @property
    def m(self) -> int:
        return int(self.indices.size)

    def sources(self) -> np.ndarray:
        return np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.indptr))


def induced(conn: Connectome, nodes: np.ndarray):
    """Induced subgraph on global `nodes`: local CSR, synapse counts and store edge positions."""
    nodes = np.unique(np.asarray(nodes, dtype=np.int64))
    local = np.full(conn.n_neurons, -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    ip = conn["out_indptr"]
    lo, counts = ip[nodes], ip[nodes + 1] - ip[nodes]
    total = int(counts.sum())
    idx = np.repeat(lo - (np.cumsum(counts) - counts), counts) + np.arange(total)
    target = local[conn["out_indices"][idx]]
    source = np.repeat(np.arange(nodes.size), counts)
    keep = (target >= 0) & (target != source)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(source[keep], minlength=nodes.size), dtype=np.int64)])
    return nodes, indptr, target[keep].astype(np.int32), np.asarray(conn["syn_count"][idx[keep]]), idx[keep]


def expansion_order(conn: Connectome, size: int, seed: int) -> tuple[np.ndarray, dict]:
    """Greedy weighted expansion (deterministic: ties go to the lower neuron index)."""
    n = conn.n_neurons
    out_ip, out_ix = np.asarray(conn["out_indptr"]), np.asarray(conn["out_indices"])
    syn = np.asarray(conn["syn_count"]).astype(np.int64)
    in_ip, in_ix = np.asarray(conn["in_indptr"]), np.asarray(conn["in_indices"])
    syn_in = syn[np.asarray(conn["in_edge"])]
    out_deg, in_deg = np.diff(out_ip), np.diff(in_ip)
    labels = lambda c: conn.labels(c)
    candidates = np.flatnonzero((labels("super_class") == "central") & (labels("flow") == "intrinsic")
                                & (out_deg >= np.median(out_deg)) & (in_deg >= np.median(in_deg)))
    seed_neuron = int(np.random.default_rng(seed).choice(candidates))
    score = np.zeros(n, dtype=np.int64)
    selected = np.zeros(n, dtype=bool)
    order = np.empty(size, dtype=np.int64)
    current = seed_neuron
    for k in range(size):
        order[k] = current
        selected[current] = True
        score[current] = -1
        for ip, ix, w in ((out_ip, out_ix, syn), (in_ip, in_ix, syn_in)):
            s, e = ip[current], ip[current + 1]
            t = ix[s:e]
            free = ~selected[t]
            score[t[free]] += w[s:e][free]
        if k + 1 < size:
            current = int(np.argmax(score))
            if score[current] <= 0:
                raise RuntimeError(f"expansion stopped at {k + 1} neurons: the component is exhausted")
    info = {"seed_neuron_index": seed_neuron, "seed_root_id": int(conn["root_id"][seed_neuron]),
            "seed_cell_type": conn.labels("cell_type", seed_neuron) or None,
            "seed_candidates": int(candidates.size),
            "seed_rule": "uniform draw among central intrinsic neurons with in- and out-degree >= the connectome median"}
    return order, info


def _composition(conn: Connectome, nodes: np.ndarray) -> dict:
    from ..analysis.regions import home_blocks

    def counts(values, top=None):
        v, c = np.unique(values, return_counts=True)
        o = np.argsort(c)[::-1][:top]
        return {str(v[i]) or "(none)": int(c[i]) for i in o}

    blocks, _ = home_blocks(conn)
    names = np.array(conn.vocab("neuropil"), dtype=object)
    return {"super_class": counts(conn.labels("super_class", nodes)),
            "top_nt": counts(conn.labels("top_nt", nodes)),
            "home_neuropil_top10": counts(names[blocks[nodes]], 10)}


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


M25_SUBGRAPHS = ("full",)


def manifests_dir(name: str | None = None) -> Path:
    """Milestone 2 subgraph manifests stay where M2 wrote them; subgraphs added in M2.5 get their own directory."""
    return paths.results_dir() / ("milestone25" if name in M25_SUBGRAPHS else "milestone2") / "subgraphs"


def cache_dir() -> Path:
    return paths.data_dir() / "processed" / "subgraphs"


def build(conn: Connectome, name: str, spec: dict) -> Subgraph:
    params = dict(spec)
    if spec["method"] == "expand":
        order, info = expansion_order(conn, spec["size"], spec["seed"])
        params.update(info)
        nodes = order
    elif spec["method"] == "full":
        nodes = np.arange(conn.n_neurons, dtype=np.int64)
        params["rule"] = "every neuron of the processed store"
    elif spec["method"] == "neuropil":
        from ..analysis.regions import home_blocks

        blocks, _ = home_blocks(conn)
        nodes = np.flatnonzero(blocks == conn.vocab("neuropil").index(spec["neuropil"]))
        params["rule"] = "home neuropil = neuropil with most presynapses (M1 analysis.regions)"
    else:
        raise ValueError(f"unknown subgraph method {spec['method']!r}")
    nodes, indptr, indices, syn, edge_global = induced(conn, nodes)
    adj = csr_array((np.ones(indices.size, dtype=np.int8), indices, indptr), shape=(nodes.size, nodes.size))
    weak_n, weak = connected_components(adj, directed=True, connection="weak")
    strong_n, strong = connected_components(adj, directed=True, connection="strong")
    manifest = {
        "name": name, "method": spec["method"], "params": params,
        "dataset": {"id": conn.manifest["dataset"], "store_manifest_sha256": sha256_file(conn.root / "manifest.json"),
                    "counts": conn.manifest["counts"]},
        "neurons": int(nodes.size), "edges": int(indices.size), "synapses": int(syn.astype(np.int64).sum()),
        "mean_out_degree": float(indices.size / max(nodes.size, 1)),
        "weak_components": int(weak_n), "largest_weak_component": int(np.bincount(weak).max()),
        "strong_components": int(strong_n), "largest_strong_component": int(np.bincount(strong).max()),
        "composition": _composition(conn, nodes),
        "sha256": {"root_ids": _digest(np.asarray(conn["root_id"][nodes], dtype=np.int64)),
                   "edges": _digest(indptr, indices, syn)},
        "created": runinfo.collect(),
    }
    return Subgraph(name, nodes.astype(np.int32), indptr, indices, syn, edge_global, manifest)


def save(sub: Subgraph) -> None:
    manifests_dir(sub.name).mkdir(parents=True, exist_ok=True)
    cache_dir().mkdir(parents=True, exist_ok=True)
    np.save(cache_dir() / f"{sub.name}.npy", sub.nodes)
    (manifests_dir(sub.name) / f"{sub.name}.json").write_text(json.dumps(sub.manifest, indent=1) + "\n")


def load(conn: Connectome, name: str) -> Subgraph:
    """Rebuild from the cached node list (or re-extract) and verify against the recorded manifest."""
    manifest_path, cache_path = manifests_dir(name) / f"{name}.json", cache_dir() / f"{name}.npy"
    if not manifest_path.is_file():
        sub = build(conn, name, CANONICAL[name])
        save(sub)
        return sub
    manifest = json.loads(manifest_path.read_text())
    if cache_path.is_file():
        nodes = np.load(cache_path)
    else:
        nodes = build(conn, name, CANONICAL[name]).nodes
        cache_dir().mkdir(parents=True, exist_ok=True)
        np.save(cache_path, nodes)
    nodes, indptr, indices, syn, edge_global = induced(conn, nodes)
    got = {"root_ids": _digest(np.asarray(conn["root_id"][nodes], dtype=np.int64)), "edges": _digest(indptr, indices, syn)}
    if got != manifest["sha256"]:
        raise ValueError(f"subgraph {name}: regenerated nodes/edges do not match the recorded manifest")
    return Subgraph(name, nodes.astype(np.int32), indptr, indices, syn, edge_global, manifest)
