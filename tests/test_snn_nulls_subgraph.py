"""Null topologies and subgraph extraction (synthetic connectome store)."""

import numpy as np
import pytest

from biobrain.connectome import preprocess
from biobrain.connectome.store import Connectome
from biobrain.snn import engine, network, nulls, subgraph
from biobrain.snn.config import InputSpec, NeuronParams, SimConfig
from biobrain.snn.inputs import generate


def random_net(n=300, p=0.04, seed=0):
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < p) & ~np.eye(n, dtype=bool)
    a |= a.T & (rng.random((n, n)) < 0.3)  # some reciprocity
    np.fill_diagonal(a, False)
    src, dst = np.nonzero(a)
    sign = np.where(rng.random(n) < 0.3, -1.0, 1.0)
    return network.Network.from_edges(n, src, dst, sign[src] * rng.random(src.size), dtype="float32")


def out_weight_multisets(net):
    return [np.sort(net.weights[net.indptr[i]:net.indptr[i + 1]]).tolist() for i in range(net.n)]


def test_degree_preserving_null_keeps_degrees_and_per_neuron_weights():
    net = random_net()
    null = nulls.make(net, "degree_preserving", seed=3)
    assert np.array_equal(null.out_degree(), net.out_degree())
    assert np.array_equal(np.bincount(null.indices, minlength=net.n), np.bincount(net.indices, minlength=net.n))
    assert out_weight_multisets(null) == out_weight_multisets(net)
    assert not np.array_equal(null.indices, net.indices)
    assert null.meta["topology"]["reciprocity_null"] < null.meta["topology"]["reciprocity_real"]


def test_reciprocity_preserving_null_keeps_mutual_pairs_approximately():
    net = random_net(seed=1)
    null = nulls.make(net, "reciprocity_preserving", seed=5)
    topo = null.meta["topology"]
    assert abs(topo["reciprocity_null"] - topo["reciprocity_real"]) < 0.05
    assert abs(null.m - net.m) <= topo["duplicate_edges_dropped"]


@pytest.mark.parametrize("variant", ["real", "degree_preserving", "reciprocity_preserving"])
def test_null_topologies_run_through_the_same_engine_interface(variant):
    net = nulls.make(random_net(seed=2), variant, seed=1)
    p = NeuronParams()
    sched = generate(InputSpec(rate=0.02), net.n, 200, 1)
    ts = engine.run(net, sched, p, 200, "time_step")
    ed = engine.run(net, sched, p, 200, "event_driven")
    assert ts.counters["spikes"] == ed.counters["spikes"] > 0


@pytest.fixture
def store(synthetic_catalog, budget):
    return Connectome.load(preprocess.run(synthetic_catalog, budget=budget, log=lambda *_: None))


def test_expansion_is_connected_nested_and_reproducible(store, monkeypatch):
    specs = {"e50": {"method": "expand", "size": 50, "seed": 4}, "e300": {"method": "expand", "size": 300, "seed": 4}}
    monkeypatch.setattr(subgraph, "CANONICAL", specs)
    small, big = subgraph.build(store, "e50", specs["e50"]), subgraph.build(store, "e300", specs["e300"])
    assert small.n == 50 and big.n == 300 and set(small.nodes) <= set(big.nodes)
    assert small.manifest["weak_components"] == 1 and big.manifest["largest_weak_component"] == 300
    subgraph.save(big)
    again = subgraph.load(store, "e300")
    assert again.manifest["sha256"] == big.manifest["sha256"]
    assert np.array_equal(again.indices, big.indices) and again.m == big.manifest["edges"]


def test_induced_subgraph_matches_bruteforce(store):
    rng = np.random.default_rng(0)
    nodes = np.sort(rng.choice(store.n_neurons, 400, replace=False))
    nodes, indptr, indices, syn, edge_global = subgraph.induced(store, nodes)
    local = {int(g): i for i, g in enumerate(nodes)}
    expected = set()
    src = store.edge_sources()
    for e in range(store.n_edges):
        a, b = int(src[e]), int(store["out_indices"][e])
        if a in local and b in local and a != b:
            expected.add((local[a], local[b], int(store["syn_count"][e])))
    got = {(int(i), int(indices[k]), int(syn[k])) for i in range(nodes.size) for k in range(indptr[i], indptr[i + 1])}
    assert got == expected and np.array_equal(store["syn_count"][edge_global], syn)


def test_network_weights_signs_and_transforms(store, monkeypatch):
    monkeypatch.setattr(subgraph, "CANONICAL", {"e200": {"method": "expand", "size": 200, "seed": 1}})
    sub = subgraph.build(store, "e200", subgraph.CANONICAL["e200"])
    counts = sub.syn_count.astype(float)
    for transform, f in (("linear", counts), ("sqrt", np.sqrt(counts)), ("log1p", np.log1p(counts)), ("clipped", np.minimum(counts, 3))):
        cfg = SimConfig().replace(weights={"transform": transform, "gain": 0.5, "clip": 3.0},
                                  signs={"missing": 1.0, "dopamine": 1.0, "serotonin": 1.0, "octopamine": 1.0})
        net = network.build(store, sub, cfg)
        norm = network.normalization(store, cfg.weights)
        assert net.m == sub.m  # no zero signs with this table
        assert np.allclose(np.abs(net.weights), 0.5 * f / norm, rtol=1e-6)
    zero = network.build(store, sub, SimConfig().replace(signs={k: 0.0 for k in ("acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine", "missing")}))
    assert zero.m == 0 and zero.meta["removed_zero_sign"] == sub.m
    edge_signs = network.build(store, sub, SimConfig().replace(signs={"source": "edge"}))
    assert "edges_where_edge_class_differs_from_neuron_label" in edge_signs.meta["signs"]
