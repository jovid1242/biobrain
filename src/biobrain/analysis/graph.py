"""Directed graphs as CSR arrays, simple undirected views, and random null models."""

from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import numpy as np
import scipy.sparse as sp

from ..connectome.store import Connectome


@dataclass(frozen=True)
class Graph:
    """Directed graph: out-neighbours of i are `indices[indptr[i]:indptr[i+1]]`, sorted, no duplicates."""

    n: int
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    name: str = ""

    @property
    def m(self) -> int:
        return int(self.indices.size)

    def sources(self) -> np.ndarray:
        return np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.indptr))

    def out_degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.indices, minlength=self.n)

    def keys(self) -> np.ndarray:
        """src * n + dst per edge; ascending because the CSR is sorted."""
        return self.sources().astype(np.int64) * self.n + self.indices

    def scipy(self) -> sp.csr_array:
        return sp.csr_array((np.ones(self.m, dtype=np.int8), self.indices, self.indptr), shape=(self.n, self.n))

    def igraph(self) -> ig.Graph:
        return ig.Graph(n=self.n, edges=np.stack([self.sources(), self.indices], axis=1), directed=True)

    @classmethod
    def from_edges(cls, n: int, src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None,
                   name: str = "") -> "Graph":
        """Build from an edge list; duplicate (src, dst) pairs are merged, weights summed."""
        weights = np.ones(src.size, dtype=np.int64) if weights is None else np.asarray(weights, dtype=np.int64)
        keys = np.asarray(src, dtype=np.int64) * n + np.asarray(dst, dtype=np.int64)
        keys, inverse = np.unique(keys, return_inverse=True)
        summed = np.bincount(inverse, weights=weights).astype(np.int64)
        counts = np.bincount((keys // n).astype(np.int64), minlength=n)
        indptr = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
        return cls(n, indptr, (keys % n).astype(np.int32), summed, name)

    @classmethod
    def from_connectome(cls, conn: Connectome, min_synapses: int = 1, drop_self_loops: bool = True) -> "Graph":
        syn = conn["syn_count"]
        keep = syn >= min_synapses
        if drop_self_loops:
            keep &= conn.edge_sources() != conn["out_indices"]
        indptr = np.concatenate([[0], np.cumsum(keep, dtype=np.int64)])[conn["out_indptr"]]
        name = f">={min_synapses} synapse{'s' if min_synapses > 1 else ''}"
        return cls(conn.n_neurons, indptr, np.asarray(conn["out_indices"][keep]), syn[keep].astype(np.int64), name)

    def subgraph(self, nodes: np.ndarray) -> tuple["Graph", np.ndarray]:
        """Induced subgraph on `nodes` (relabelled 0..k-1 in ascending original order) + the node map."""
        nodes = np.unique(nodes)
        new_id = np.full(self.n, -1, dtype=np.int64)
        new_id[nodes] = np.arange(nodes.size)
        src, dst = new_id[self.sources()], new_id[self.indices]
        keep = (src >= 0) & (dst >= 0)
        return Graph.from_edges(nodes.size, src[keep], dst[keep], self.weights[keep], self.name), nodes


def undirected_edges(g: Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple undirected view: one (lo, hi) edge per connected pair, weight = synapses in both directions."""
    src, dst = g.sources().astype(np.int64), g.indices.astype(np.int64)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    keys, inverse = np.unique(lo * g.n + hi, return_inverse=True)
    weight = np.bincount(inverse, weights=g.weights).astype(np.int64)
    return (keys // g.n).astype(np.int32), (keys % g.n).astype(np.int32), weight


def undirected_igraph(g: Graph) -> tuple[ig.Graph, np.ndarray]:
    lo, hi, w = undirected_edges(g)
    return ig.Graph(n=g.n, edges=np.stack([lo, hi], axis=1), directed=False), w


def erdos_renyi(n: int, m: int, rng: np.random.Generator, name: str = "ER") -> Graph:
    """Directed G(n, m) without self-loops or duplicate edges."""
    pairs = np.empty(0, dtype=np.int64)
    while pairs.size < m:
        draw = rng.integers(0, n * (n - 1), size=int((m - pairs.size) * 1.02) + 16)
        pairs = np.unique(np.concatenate([pairs, draw]))
    pairs = rng.permutation(pairs)[:m]
    src = pairs // (n - 1)
    dst = pairs % (n - 1)
    dst = dst + (dst >= src)  # skip the diagonal
    return Graph.from_edges(n, src, dst, name=name)


def degree_preserving_null(g: Graph, seed: int, swaps_per_edge: int = 10) -> Graph:
    """Maslov-Sneppen rewiring (igraph): in- and out-degree of every node preserved, simple graph, unweighted."""
    import random

    ig.set_random_number_generator(random.Random(seed))
    h = g.igraph()
    h.rewire(n=swaps_per_edge * h.ecount(), allowed_edge_types="simple")
    edges = np.array(h.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    return Graph.from_edges(g.n, edges[:, 0], edges[:, 1], name=f"degree-preserving null (seed {seed})")
