"""Region-level views: neuropil 'home' blocks, group-to-group matrices, synapse locations."""

from __future__ import annotations

import numpy as np

from ..connectome.store import Connectome
from .graph import Graph

BLOCK_RULE = ("home neuropil = the neuropil holding most of the neuron's presynapses (all partners, incl. fragments), "
              "as in Lin et al. 2024's neuropil blocks; neurons without presynapses fall back to their postsynapses")


def primary_neuropil(conn: Connectome, side: str) -> np.ndarray:
    """Per neuron, the neuropil code with the most `side` ('pre'/'post') synapses; 0 if it has none.
    Ties go to the lower neuropil code (alphabetical)."""
    indptr = conn[f"np_{side}_indptr"]
    counts = conn[f"np_{side}_count"].astype(np.int64)
    rows = np.repeat(np.arange(conn.n_neurons), np.diff(indptr))
    order = np.lexsort((-counts, rows))  # rows keep their CSR positions; inside a row: largest count first
    has = np.diff(indptr) > 0
    best = np.zeros(conn.n_neurons, dtype=np.int64)
    best[has] = conn[f"np_{side}_neuropil"][order[indptr[:-1][has]]]
    return best


def home_blocks(conn: Connectome) -> tuple[np.ndarray, dict]:
    pre, post = primary_neuropil(conn, "pre"), primary_neuropil(conn, "post")
    block = np.where(pre > 0, pre, post)
    return block, {"rule": BLOCK_RULE, "from_presynapses": int(np.sum(pre > 0)),
                   "from_postsynapse_fallback": int(np.sum((pre == 0) & (post > 0))), "unassigned": int(np.sum(block == 0))}


def group_matrices(g: Graph, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k x k synapse and connection counts between label groups (labels are integers in [0, k))."""
    key = labels[g.sources()].astype(np.int64) * k + labels[g.indices]
    synapses = np.bincount(key, weights=g.weights, minlength=k * k).reshape(k, k).astype(np.int64)
    connections = np.bincount(key, minlength=k * k).reshape(k, k)
    return synapses, connections


def neuropil_synapses(conn: Connectome) -> np.ndarray:
    """Synapses between proofread neurons located in each neuropil (index = neuropil code)."""
    return np.bincount(conn["row_neuropil"], weights=conn["row_syn_count"],
                       minlength=len(conn.vocab("neuropil"))).astype(np.int64)
