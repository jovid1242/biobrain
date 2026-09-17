from collections import defaultdict

import numpy as np
import pytest

from biobrain.connectome import preprocess
from biobrain.connectome.store import Connectome, sha256_file
from conftest import FRAGMENT, IDS, ROWS, nt_probs


@pytest.fixture
def store(toy_catalog, budget):
    path = preprocess.run(toy_catalog, budget=budget, log=lambda *_: None)
    return Connectome.load(path, verify=True)


def expected_edges():
    """(pre, post) -> [synapses, synapse-weighted probabilities over predicted rows, predicted synapses]"""
    ids = sorted(IDS)
    edges = defaultdict(lambda: [0, np.zeros(6), 0])
    for pre, post, neuropil, syn, dom in ROWS:
        if pre in ids and post in ids:
            e = edges[(ids.index(pre), ids.index(post))]
            e[0] += syn
            if dom is not None:
                e[1] += syn * np.array(nt_probs(dom))
                e[2] += syn
    return dict(sorted(edges.items()))


def test_neurons_sorted_and_counts(store):
    assert store["root_id"].tolist() == sorted(IDS)
    assert store.n_neurons == 6
    exp = expected_edges()
    assert store.n_edges == len(exp)
    assert store.manifest["counts"]["synapses"] == sum(v[0] for v in exp.values())


def test_csr_matches_bruteforce_aggregation(store):
    exp = expected_edges()
    pre = store.edge_sources()
    got = {(int(pre[e]), int(store["out_indices"][e])): e for e in range(store.n_edges)}
    assert list(got) == list(exp)  # CSR order is (pre, post)
    for pair, e in got.items():
        syn, weighted, predicted = exp[pair]
        assert store["syn_count"][e] == syn
        want = np.rint(weighted / predicted * 255).astype(np.uint8) if predicted else np.zeros(6, dtype=np.uint8)
        assert np.array_equal(store["nt_prob_q"][e], want)


def test_csc_is_a_permutation_of_csr(store):
    pre = store.edge_sources()
    post_of_slot = np.repeat(np.arange(store.n_neurons), store.in_degree())
    e = store["in_edge"]
    assert sorted(e.tolist()) == list(range(store.n_edges))
    assert np.array_equal(store["out_indices"][e], post_of_slot)
    assert np.array_equal(store["in_indices"], pre[e])


def test_rows_keep_neuropil_breakdown(store):
    names = store.vocab("neuropil")
    assert names[0] == "" and names[1:] == sorted(names[1:])
    ids = sorted(IDS)
    e01 = store.index_of(IDS[0])[0]
    first_edge = store["out_indptr"][e01]
    rows = np.flatnonzero(store["row_edge"] == first_edge)
    assert [names[c] for c in store["row_neuropil"][rows]] == ["AL_L", "LH_L", "SMP_L"]
    assert store["row_syn_count"][rows].tolist() == [3, 7, 5]
    assert np.bincount(store["row_edge"], weights=store["row_syn_count"]).astype(int).tolist() == store["syn_count"].tolist()
    assert ids  # neuron order used above is the sorted one


def test_observations(store):
    obs = store.manifest["observed"]
    assert obs["connections"]["rows"] == len(ROWS)
    assert obs["connections"]["rows_pre_not_in_root_ids"] == 1 and obs["connections"]["rows_dropped"] == 1
    assert obs["connections"]["rows_unassigned_neuropil"] == 1
    assert obs["connections"]["unassigned_neuropil_labels_rows"] == {"UNASGD": 1, "None": 0}
    assert obs["connections"]["neuropil_names_in_table"] == 5 and "UNASGD" not in store.vocab("neuropil")
    assert obs["edges"]["self_connection_pairs"] == 1
    assert obs["connections"]["duplicate_pre_post_neuropil_rows"] == 0
    tr = obs["transmitters"]
    assert (tr["rows_without_prediction"], tr["synapses_without_prediction"]) == (2, 17)
    assert (tr["edges_without_prediction"], tr["edges_partially_predicted"]) == (1, 1)
    assert tr["predicted_rows_sum_dev_gt_1e-3"] == 0 and tr["predicted_values_outside_0_1"] == 0


def test_degrees_synapses_adjacency(store):
    i = store.index_of(IDS[:2])
    assert store.out_synapses()[i[0]] == 15 and store.in_synapses()[i[1]] == 15
    assert store.out_degree().sum() == store.in_degree().sum() == store.n_edges
    adj = store.adjacency(min_synapses=5, drop_self_loops=True)
    assert adj.nnz == 3 and adj[i[0], i[1]] == 15
    with pytest.raises(KeyError):
        store.index_of(FRAGMENT)


def test_neuropil_counts_keep_only_proofread_neurons(store):
    obs = store.manifest["observed"]["neuropil_counts_pre"]
    assert obs["synapses_all_segments"] == 166 and obs["synapses_proofread_neurons"] == 66
    assert obs["synapses_unassigned_neuropil_all_segments"] == 12 and obs["unassigned_labels"] == {"UNASGD": 0, "None": 12}
    assert obs["synapses_proofread_neurons_assigned_neuropil"] == 54
    i = store.index_of(IDS[0])[0]
    lo, hi = store["np_pre_indptr"][i:i + 2]
    assert store["np_pre_count"][lo:hi].tolist() == [30, 7, 5]


def test_annotations_aligned_to_neuron_index(store):
    obs = store.manifest["observed"]["annotations"]
    assert obs["neurons_without_row"] == 1 and obs["neurons_without_row_ids"] == [IDS[5]]
    idx = store.index_of(IDS)
    assert store.labels("super_class", idx).tolist() == ["central", "optic", "central", "sensory", "optic", ""]
    assert store.labels("cell_type", idx).tolist() == ["T0", "T1", "T0", "T1", "", ""]
    assert np.isnan(store["ann_soma_x"][idx[2]]) and store["ann_soma_x"][idx[0]] == 10
    assert store["ann_nucleus_id"][idx[2]] == -1 and store["ann_supervoxel_id"][idx[1]] == 78112261444987078
    assert store["ann_nucleus_id"][idx[3]] == 2453927  # parsed from "2453927.0"
    assert "ann_vfb_id" not in store.arrays and not store["ann_has_row"][idx[5]]


def test_neuron_lookup(store):
    from biobrain.analysis import lookup

    info = lookup.describe(store, IDS[0])
    assert info["out_degree"] == 1 and info["out_synapses_to_proofread"] == 15 and info["in_degree"] == 1
    assert info["strongest_outgoing"][0]["root_id"] == IDS[1] and info["strongest_incoming"][0]["synapses"] == 2
    nt = info["outgoing_transmitter_mean_probability"]
    assert nt["gaba"] == pytest.approx(0.268, abs=0.003) and nt["ach"] == pytest.approx(0.572, abs=0.003)
    assert info["annotations"]["super_class"] == "central"
    assert info["presynapses_all_partners_by_neuropil"] == {"AL_L": 30, "LH_L": 7, "SMP_L": 5}
    assert "strongest outgoing" in lookup.render(info)


def test_manifest_checksums_and_rebuild_is_deterministic(toy_catalog, budget):
    first = preprocess.run(toy_catalog, budget=budget, log=lambda *_: None)
    digests = {name: meta["sha256"] for name, meta in Connectome.load(first).manifest["arrays"].items()}
    assert all(sha256_file(first / f"arrays/{name}.npy") == d for name, d in digests.items())
    second = preprocess.run(toy_catalog, budget=budget, log=lambda *_: None)
    assert {name: meta["sha256"] for name, meta in Connectome.load(second).manifest["arrays"].items()} == digests
