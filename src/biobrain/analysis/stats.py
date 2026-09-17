"""Graph statistics on `Graph` objects. Every function returns plain JSON-able numbers.

Exact where the cost is linear; estimates (with the sample size and an uncertainty) where it is not.
Definitions follow Lin et al. 2024 where one exists, so values can be compared.
"""

from __future__ import annotations

import random
import time

import igraph as ig
import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import pearsonr, spearmanr

from .graph import Graph, undirected_edges

MAN_TRIADS = ("003", "012", "102", "021D", "021U", "021C", "111D", "111U",
              "030T", "030C", "201", "120D", "120U", "120C", "210", "300")
CONNECTED_TRIADS = MAN_TRIADS[3:]  # the 13 connected three-node motifs


def summary(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)), "max": float(v.max()),
            "zeros": int(np.sum(v == 0)), "gini": gini(v)}


def gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64))
    if v.sum() == 0:
        return 0.0
    ranks = np.arange(1, v.size + 1)
    return float((2 * np.sum(ranks * v) / (v.size * v.sum())) - (v.size + 1) / v.size)


def top_share(values: np.ndarray, fraction: float) -> float:
    """Share of the total held by the top `fraction` of items."""
    v = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    k = max(1, int(round(fraction * v.size)))
    return float(v[:k].sum() / v.sum())


def degrees(g: Graph, mask: np.ndarray | None = None) -> dict:
    kin, kout = g.in_degree(), g.out_degree()
    win = np.bincount(g.indices, weights=g.weights, minlength=g.n)
    wout = np.bincount(g.sources(), weights=g.weights, minlength=g.n)
    sel = slice(None) if mask is None else mask
    pr = pearsonr(kin[sel], kout[sel])
    sr = spearmanr(kin[sel], kout[sel])
    return {
        "in_degree": summary(kin[sel]), "out_degree": summary(kout[sel]),
        "in_synapses": summary(win[sel]), "out_synapses": summary(wout[sel]),
        "in_out_degree_pearson": float(pr.statistic), "in_out_degree_spearman": float(sr.statistic),
        "top1pct_share_of_total_degree": top_share((kin + kout)[sel], 0.01),
    }


def reciprocity(g: Graph) -> dict:
    """P(j->i | i->j) over directed edges without self-loops, with ER and configuration-model baselines."""
    keys = g.keys()
    rev = g.indices.astype(np.int64) * g.n + g.sources()
    pos = np.minimum(np.searchsorted(keys, rev), keys.size - 1)
    mutual = keys[pos] == rev
    r = float(mutual.mean())
    density = g.m / (g.n * (g.n - 1))
    kin, kout = g.in_degree().astype(np.float64), g.out_degree().astype(np.float64)
    cfg = float(np.sum(kin * kout) ** 2 / g.m ** 3)  # sum_ij p_ij p_ji with p_ij = kout_i kin_j / m
    return {"reciprocity": r, "reciprocal_edges": int(mutual.sum()), "density": density,
            "er_expected": density, "ratio_to_er": r / density,
            "cfg_expected_analytic": cfg, "ratio_to_cfg_analytic": r / cfg,
            "neurons_with_a_reciprocal_partner": int(np.unique(g.sources()[mutual]).size)}


def components(g: Graph) -> tuple[dict, np.ndarray, np.ndarray]:
    """Weak and strong components; also returns the giant-WCC and giant-SCC node masks."""
    a = g.scipy()
    active = (g.in_degree() + g.out_degree()) > 0
    out, masks = {"neurons": g.n, "neurons_with_edges": int(active.sum()), "isolated": int((~active).sum())}, []
    for kind in ("weak", "strong"):
        count, labels = connected_components(a, directed=True, connection=kind)
        sizes = np.bincount(labels)
        giant = int(np.argmax(sizes))
        rest = np.delete(sizes, giant)
        out[kind] = {"components": int(count), "largest": int(sizes[giant]),
                     "largest_fraction_of_all": float(sizes[giant] / g.n),
                     "largest_fraction_of_neurons_with_edges": float(sizes[giant] / max(1, active.sum())),
                     "second_largest": int(rest.max()) if rest.size else 0,
                     "singletons": int(np.sum(sizes == 1))}
        masks.append(labels == giant)
    return out, masks[0], masks[1]


def clustering(g: Graph) -> dict:
    """Global clustering (transitivity) of the simple undirected view: 3 x triangles / connected triples."""
    lo, hi, _ = undirected_edges(g)
    und = ig.Graph(n=g.n, edges=np.stack([lo, hi], axis=1), directed=False)
    d = np.bincount(np.concatenate([lo, hi]), minlength=g.n).astype(np.float64)
    wedges = float(np.sum(d * (d - 1) / 2))
    t0 = time.monotonic()
    c = und.transitivity_undirected()
    avg_local = und.transitivity_avglocal_undirected(mode="zero")
    m_u = lo.size
    return {"global_clustering": float(c), "average_local_clustering": float(avg_local),
            "undirected_edges": int(m_u), "wedges": wedges, "triangles": float(c * wedges / 3),
            "er_expected": 2 * m_u / (g.n * (g.n - 1)), "seconds": round(time.monotonic() - t0, 2)}


def wedge_count(g: Graph) -> float:
    lo, hi, _ = undirected_edges(g)
    d = np.bincount(np.concatenate([lo, hi]), minlength=g.n).astype(np.float64)
    return float(np.sum(d * (d - 1) / 2))


def path_lengths(g: Graph, nodes: np.ndarray, k: int, seed: int, directed: bool, batch: int = 16) -> dict:
    """Shortest-path hops from k random sources to every other node of `nodes` (a mask of a connected set)."""
    rng = np.random.default_rng(seed)
    a = g.scipy()
    if not directed:
        a = (a + a.T).tocsr()
    candidates = np.flatnonzero(nodes)
    sources = np.sort(rng.choice(candidates, size=min(k, candidates.size), replace=False))
    hist = np.zeros(128, dtype=np.int64)
    per_source, unreachable = [], 0
    t0 = time.monotonic()
    for start in range(0, sources.size, batch):
        d = shortest_path(a, directed=directed, unweighted=True, indices=sources[start:start + batch])[:, nodes]
        finite = np.isfinite(d)
        unreachable += int((~finite).sum())
        for row, ok in zip(d, finite):
            hops = row[ok & (row > 0)].astype(np.int64)
            hist += np.bincount(hops, minlength=hist.size)[:hist.size]
            per_source.append(hops.mean())
    per_source = np.array(per_source)
    lengths = np.arange(hist.size)
    mean = float(np.sum(lengths * hist) / hist.sum())
    return {"directed": directed, "sources": int(sources.size), "targets_per_source": int(nodes.sum() - 1),
            "mean": mean, "sem_over_sources": float(per_source.std(ddof=1) / np.sqrt(per_source.size)) if per_source.size > 1 else None,
            "max_observed": int(np.max(np.flatnonzero(hist))), "unreachable_pairs": unreachable,
            "histogram": {int(h): int(c) for h, c in enumerate(hist) if c}, "seconds": round(time.monotonic() - t0, 1)}


def triad_census(g: Graph) -> dict:
    t0 = time.monotonic()
    tc = g.igraph().triad_census()
    return {"counts": {name: int(tc[name]) for name in MAN_TRIADS}, "seconds": round(time.monotonic() - t0, 1)}


def rich_club(g: Graph, ks: np.ndarray) -> np.ndarray:
    """phi(k) = E_k / (N_k (N_k - 1)) over nodes with total degree > k (directed edges among them)."""
    deg = g.in_degree() + g.out_degree()
    edge_min = np.sort(np.minimum(deg[g.sources()], deg[g.indices]))
    node_deg = np.sort(deg)
    e_k = edge_min.size - np.searchsorted(edge_min, ks, side="right")
    n_k = node_deg.size - np.searchsorted(node_deg, ks, side="right")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n_k > 1, e_k / (n_k * (n_k - 1.0)), np.nan)


def rich_club_regime(ks, normalized, neurons_above, ratio: float = 1.01, min_neurons: int = 10) -> dict:
    """Describe a normalized rich-club curve instead of reducing it to one cut-off.
    Only points with at least `min_neurons` neurons above k are considered."""
    rows = [(int(k), float(v), int(n)) for k, v, n in zip(ks, normalized, neurons_above)
            if v is not None and np.isfinite(v) and n >= min_neurons]
    above = [r for r in rows if r[1] > ratio]
    peak = max(rows, key=lambda r: r[1]) if rows else None
    after_peak = [r for r in rows if peak and r[0] > peak[0] and r[1] < 1.0]
    lowest = min(rows, key=lambda r: r[1]) if rows else None
    return {
        "rule": f"normalized coefficient > {ratio} (Lin et al. 2024 criterion); points with N_k >= {min_neurons}",
        "first_k_above": above[0][0] if above else None,
        "neurons_above_first_k": above[0][2] if above else None,
        "k_range_above": [above[0][0], above[-1][0]] if above else None,
        "contiguous": bool(above) and all(r[1] > ratio for r in rows if above[0][0] <= r[0] <= above[-1][0]),
        "max": {"k": peak[0], "normalized": peak[1], "neurons_above": peak[2]} if peak else None,
        "first_k_below_1_after_max": after_peak[0][0] if after_peak else None,
        "min": {"k": lowest[0], "normalized": lowest[1], "neurons_above": lowest[2]} if lowest else None,
    }


def leiden(n: int, lo: np.ndarray, hi: np.ndarray, weights: np.ndarray | None, seed: int,
           max_iterations: int = 20, min_gain: float = 1e-4, time_limit_s: float = 900) -> dict:
    """Modularity Leiden on an undirected graph, one iteration at a time until the gain stalls."""
    ig.set_random_number_generator(random.Random(seed))
    g = ig.Graph(n=n, edges=np.stack([lo, hi], axis=1), directed=False)
    w = None if weights is None else weights.astype(np.float64)
    membership, quality, history, stopped = None, -1.0, [], "max iterations"
    t0 = time.monotonic()
    for it in range(max_iterations):
        part = g.community_leiden(objective_function="modularity", weights=w, n_iterations=1,
                                  initial_membership=membership)
        q = g.modularity(part.membership, weights=w)
        history.append({"iteration": it + 1, "modularity": q, "communities": len(part), "t": round(time.monotonic() - t0, 1)})
        gain, quality, membership = q - quality, q, part.membership
        if gain < min_gain:
            stopped = f"gain < {min_gain}"
            break
        if time.monotonic() - t0 > time_limit_s:
            stopped = "time limit"
            break
    return {"modularity": quality, "membership": np.asarray(membership, dtype=np.int32), "history": history,
            "stopped": stopped}


def modularity(n: int, lo: np.ndarray, hi: np.ndarray, weights: np.ndarray | None, membership: np.ndarray) -> float:
    g = ig.Graph(n=n, edges=np.stack([lo, hi], axis=1), directed=False)
    return float(g.modularity(membership.tolist(), weights=None if weights is None else weights.astype(np.float64)))


def partition_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Normalized mutual information (arithmetic mean normalisation) and adjusted Rand index."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    n = a.size
    joint = np.unique(a.astype(np.int64) * (b.max() + 1) + b, return_counts=True)[1].astype(np.float64)
    pa = np.bincount(a).astype(np.float64)
    pb = np.bincount(b).astype(np.float64)
    ent = lambda c: -np.sum((c / n) * np.log(c / n))
    ha, hb, hab = ent(pa), ent(pb), ent(joint)
    mi = ha + hb - hab
    nmi = float(2 * mi / (ha + hb)) if ha + hb > 0 else 1.0
    comb = lambda c: np.sum(c * (c - 1) / 2)
    index, expected = comb(joint), comb(pa) * comb(pb) / (n * (n - 1) / 2)
    maximum = (comb(pa) + comb(pb)) / 2
    ari = float((index - expected) / (maximum - expected)) if maximum != expected else 1.0
    return {"nmi": nmi, "ari": ari}


def sampled_betweenness(g: Graph, k: int, seed: int) -> tuple[np.ndarray, dict]:
    """Brandes betweenness from k random sources, scaled by n/k (an unbiased estimate of the full sum)."""
    rng = np.random.default_rng(seed)
    sources = np.sort(rng.choice(g.n, size=min(k, g.n), replace=False))
    t0 = time.monotonic()
    scores = np.asarray(g.igraph().betweenness(directed=True, sources=sources.tolist()), dtype=np.float64)
    return scores * (g.n / sources.size), {"sources": int(sources.size), "seed": seed, "seconds": round(time.monotonic() - t0, 1)}
