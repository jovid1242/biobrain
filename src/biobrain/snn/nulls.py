"""Null topologies for the same simulation interface (Milestone 2: infrastructure only, no scientific claims).

- degree_preserving: Maslov–Sneppen rewiring of the simulated edges (every in- and out-degree kept).
- reciprocity_preserving: mutual pairs are rewired among themselves as an undirected graph and one-way edges
  among themselves as a directed graph, so in/out-degrees and the number of mutual pairs are kept (collisions
  between the two parts are dropped and counted).
In both, each presynaptic neuron keeps its own multiset of outgoing weights (so its sign and output strength are
preserved), assigned to its new targets in random order.
"""

from __future__ import annotations

import random

import igraph as ig
import numpy as np

from .network import Network


def _edges(net: Network) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(np.arange(net.n, dtype=np.int64), net.out_degree()), net.indices.astype(np.int64)


def _rewire(n: int, src: np.ndarray, dst: np.ndarray, directed: bool, seed: int) -> np.ndarray:
    ig.set_random_number_generator(random.Random(seed))
    g = ig.Graph(n=n, edges=np.stack([src, dst], axis=1), directed=directed)
    g.rewire(n=10 * max(g.ecount(), 1), allowed_edge_types="simple")
    return np.array(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)


def _assign_weights(net: Network, new_src: np.ndarray, new_dst: np.ndarray, seed: int, variant: str, info: dict) -> Network:
    rng = np.random.default_rng(seed)
    order = np.lexsort((new_dst, new_src))
    new_src, new_dst = new_src[order], new_dst[order]
    new_deg = np.bincount(new_src, minlength=net.n)
    old_deg = net.out_degree()
    weights = np.empty(new_src.size, dtype=net.weights.dtype)
    new_ptr = np.concatenate([[0], np.cumsum(new_deg)])
    for i in np.flatnonzero(new_deg):
        own = net.weights[net.indptr[i]:net.indptr[i + 1]]
        k = new_deg[i]
        weights[new_ptr[i]:new_ptr[i] + k] = rng.permutation(own)[:k] if k <= own.size else rng.choice(own, size=k)
    info.update({"variant": variant, "seed": seed, "edges": int(new_src.size),
                 "neurons_with_changed_out_degree": int(np.sum(new_deg != old_deg))})
    indptr = np.concatenate([[0], np.cumsum(new_deg, dtype=np.int64)])
    meta = {**net.meta, "topology": info}
    return Network(net.n, indptr, new_dst.astype(np.int32), weights, meta)


def reciprocity(net: Network) -> float:
    src, dst = _edges(net)
    keys = src * net.n + dst
    rev = dst * net.n + src
    return float(np.isin(rev, keys).mean()) if keys.size else 0.0


def degree_preserving(net: Network, seed: int) -> Network:
    src, dst = _edges(net)
    new = _rewire(net.n, src, dst, directed=True, seed=seed)
    return _assign_weights(net, new[:, 0], new[:, 1], seed, "degree_preserving", {"reciprocity_real": reciprocity(net)})


def reciprocity_preserving(net: Network, seed: int) -> Network:
    src, dst = _edges(net)
    keys = src * net.n + dst
    mutual = np.isin(dst * net.n + src, keys)
    lower = mutual & (src < dst)  # one undirected edge per mutual pair
    und = _rewire(net.n, src[lower], dst[lower], directed=False, seed=seed) if lower.any() else np.zeros((0, 2), np.int64)
    one_way = _rewire(net.n, src[~mutual], dst[~mutual], directed=True, seed=seed + 1) if (~mutual).any() else np.zeros((0, 2), np.int64)
    both = np.concatenate([np.concatenate([und, und[:, ::-1]]), one_way])
    new_keys, first = np.unique(both[:, 0] * net.n + both[:, 1], return_index=True)
    info = {"reciprocity_real": reciprocity(net), "mutual_pairs_real": int(lower.sum()),
            "duplicate_edges_dropped": int(both.shape[0] - new_keys.size)}
    kept = both[np.sort(first)]
    return _assign_weights(net, kept[:, 0], kept[:, 1], seed, "reciprocity_preserving", info)


def make(net: Network, variant: str, seed: int) -> Network:
    if variant == "real":
        return net
    if variant == "degree_preserving":
        out = degree_preserving(net, seed)
    elif variant == "reciprocity_preserving":
        out = reciprocity_preserving(net, seed)
    else:
        raise ValueError(f"unknown topology variant {variant!r}")
    out.meta["topology"]["reciprocity_null"] = reciprocity(out)
    return out
