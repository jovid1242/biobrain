"""Graph statistics against brute force on small random graphs."""

import itertools

import numpy as np
import pytest

from biobrain.analysis import stats
from biobrain.analysis.graph import Graph, degree_preserving_null, erdos_renyi, undirected_edges


def random_graph(n=24, p=0.15, seed=0):
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < p) & ~np.eye(n, dtype=bool)
    src, dst = np.nonzero(a)
    return Graph.from_edges(n, src, dst, rng.integers(1, 9, src.size)), a


def test_from_edges_merges_duplicates_and_sorts():
    g = Graph.from_edges(3, np.array([2, 0, 0, 2]), np.array([1, 2, 2, 0]), np.array([1, 5, 7, 2]))
    assert g.indptr.tolist() == [0, 1, 1, 3]
    assert g.indices.tolist() == [2, 0, 1] and g.weights.tolist() == [12, 2, 1]


def test_subgraph_and_undirected_view():
    g, a = random_graph()
    nodes = np.array([3, 1, 7, 12, 20])
    sub, kept = g.subgraph(nodes)
    assert kept.tolist() == sorted(nodes.tolist())
    assert np.array_equal(sub.scipy().toarray().astype(bool), a[np.ix_(kept, kept)])
    lo, hi, _ = undirected_edges(g)
    expect = {(min(i, j), max(i, j)) for i, j in zip(*np.nonzero(a))}
    assert set(zip(lo.tolist(), hi.tolist())) == expect


def test_null_models_keep_what_they_promise():
    g, _ = random_graph(n=60, p=0.1, seed=3)
    null = degree_preserving_null(g, seed=1)
    assert np.array_equal(null.in_degree(), g.in_degree()) and np.array_equal(null.out_degree(), g.out_degree())
    assert np.all(null.sources() != null.indices) and np.unique(null.keys()).size == null.m
    assert not np.array_equal(null.keys(), g.keys())
    er = erdos_renyi(50, 400, np.random.default_rng(0))
    assert er.m == 400 and np.all(er.sources() != er.indices) and np.unique(er.keys()).size == 400


def test_reciprocity_bruteforce():
    g, a = random_graph(seed=1)
    r = stats.reciprocity(g)
    assert r["reciprocal_edges"] == int(np.sum(a & a.T))
    assert r["reciprocity"] == pytest.approx(np.sum(a & a.T) / a.sum())
    assert r["neurons_with_a_reciprocal_partner"] == int(np.sum((a & a.T).any(axis=1)))


def reach(a):
    r = a | np.eye(len(a), dtype=bool)
    for _ in range(len(a)):
        r = r | (r.astype(int) @ r.astype(int) > 0)
    return r


def test_components_bruteforce():
    g, a = random_graph(n=30, p=0.05, seed=2)
    out, weak, strong = stats.components(g)
    r = reach(a)
    scc_ids = {frozenset(np.flatnonzero(r[i] & r[:, i])) for i in range(30)}
    wr = reach(a | a.T)
    wcc_ids = {frozenset(np.flatnonzero(wr[i])) for i in range(30)}
    assert out["strong"]["components"] == len(scc_ids) and out["weak"]["components"] == len(wcc_ids)
    assert strong.sum() == max(len(c) for c in scc_ids) and weak.sum() == max(len(c) for c in wcc_ids)


def test_clustering_bruteforce():
    g, a = random_graph(n=30, p=0.3, seed=4)
    u = (a | a.T).astype(np.int64)
    triangles = np.trace(u @ u @ u) / 6
    d = u.sum(axis=1)
    assert stats.clustering(g)["global_clustering"] == pytest.approx(3 * triangles / np.sum(d * (d - 1) / 2))


def test_path_lengths_bruteforce():
    g, a = random_graph(n=25, p=0.2, seed=5)
    _, _, strong = stats.components(g)
    res = stats.path_lengths(g, strong, k=25, seed=0, directed=True)
    n = len(a)
    dist = np.where(a, 1.0, np.inf)
    np.fill_diagonal(dist, 0)
    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    sub = dist[np.ix_(strong, strong)]
    assert res["mean"] == pytest.approx(sub[sub > 0].mean())
    assert res["max_observed"] == int(sub.max()) and res["unreachable_pairs"] == 0


# igraph's documented triad definitions (A, B, C in order), used as an independent reference
TRIAD_EDGES = {
    "003": [], "012": [("A", "B")], "102": [("A", "B"), ("B", "A")],
    "021D": [("B", "A"), ("B", "C")], "021U": [("A", "B"), ("C", "B")], "021C": [("A", "B"), ("B", "C")],
    "111D": [("A", "B"), ("B", "A"), ("C", "B")], "111U": [("A", "B"), ("B", "A"), ("B", "C")],
    "030T": [("A", "B"), ("C", "B"), ("A", "C")], "030C": [("B", "A"), ("C", "B"), ("A", "C")],
    "201": [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")],
    "120D": [("B", "A"), ("B", "C"), ("A", "C"), ("C", "A")], "120U": [("A", "B"), ("C", "B"), ("A", "C"), ("C", "A")],
    "120C": [("A", "B"), ("B", "C"), ("A", "C"), ("C", "A")],
    "210": [("A", "B"), ("B", "C"), ("C", "B"), ("A", "C"), ("C", "A")],
    "300": [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"), ("A", "C"), ("C", "A")],
}


def triad_signature(edges):
    """Isomorphism-invariant signature of a 3-node digraph: sorted (out, in, mutual) per node + edge count."""
    nodes = "ABC"
    e = set(edges)
    sig = []
    for x in nodes:
        out = sum((x, y) in e for y in nodes if y != x)
        inn = sum((y, x) in e for y in nodes if y != x)
        mut = sum((x, y) in e and (y, x) in e for y in nodes if y != x)
        sig.append((out, inn, mut))
    return tuple(sorted(sig)), len(e)


def test_triad_census_bruteforce():
    g, a = random_graph(n=14, p=0.25, seed=6)
    reference = {triad_signature(edges): name for name, edges in TRIAD_EDGES.items()}
    assert len(reference) == 16  # the signature separates all 16 classes
    expected = dict.fromkeys(stats.MAN_TRIADS, 0)
    for i, j, k in itertools.combinations(range(14), 3):
        label = dict(zip((i, j, k), "ABC"))
        edges = [(label[x], label[y]) for x, y in itertools.permutations((i, j, k), 2) if a[x, y]]
        expected[reference[triad_signature(edges)]] += 1
    assert stats.triad_census(g)["counts"] == expected


def test_rich_club_bruteforce():
    g, a = random_graph(n=40, p=0.12, seed=7)
    deg = a.sum(0) + a.sum(1)
    ks = np.array([0, 3, 6, 9, 12])
    got = stats.rich_club(g, ks)
    for k, phi in zip(ks, got):
        rich = deg > k
        nk = rich.sum()
        want = a[np.ix_(rich, rich)].sum() / (nk * (nk - 1)) if nk > 1 else np.nan
        assert (np.isnan(phi) and np.isnan(want)) or phi == pytest.approx(want)


def test_partition_agreement():
    x = np.repeat(np.arange(5), 20)
    assert stats.partition_agreement(x, x) == {"nmi": 1.0, "ari": 1.0}
    assert stats.partition_agreement(x, (x + 3) % 5) == pytest.approx({"nmi": 1.0, "ari": 1.0})
    rng = np.random.default_rng(0)
    noise = stats.partition_agreement(x, rng.integers(0, 5, x.size))
    assert abs(noise["ari"]) < 0.05 and noise["nmi"] < 0.1


def test_leiden_finds_planted_communities_deterministically():
    rng = np.random.default_rng(0)
    blocks = np.repeat([0, 1, 2], 30)
    pairs = [(i, j) for i in range(90) for j in range(i + 1, 90)
             if rng.random() < (0.4 if blocks[i] == blocks[j] else 0.01)]
    lo, hi = (np.array(x) for x in zip(*pairs))
    first = stats.leiden(90, lo, hi, None, seed=11)
    again = stats.leiden(90, lo, hi, None, seed=11)
    assert np.array_equal(first["membership"], again["membership"])
    assert stats.partition_agreement(first["membership"], blocks)["ari"] == 1.0 and first["modularity"] > 0.5


def test_sampled_betweenness_with_all_sources_is_exact():
    g, _ = random_graph(n=30, p=0.15, seed=8)
    est, meta = stats.sampled_betweenness(g, k=30, seed=0)
    assert np.allclose(est, g.igraph().betweenness(directed=True)) and meta["sources"] == 30


def test_gini_and_summary():
    assert stats.gini(np.ones(10)) == pytest.approx(0)
    assert stats.gini(np.r_[np.zeros(9), 1]) == pytest.approx(0.9)
    assert stats.summary(np.array([0, 1, 2, 3]))["zeros"] == 1
