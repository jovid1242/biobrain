"""LIF engine: model semantics on hand-built networks, both modes, and their equivalence."""

import math

import numpy as np
import pytest

from biobrain.snn import metrics
from biobrain.snn.config import InputSpec, NeuronParams
from biobrain.snn.engine import gather, run
from biobrain.snn.inputs import InputSchedule, generate
from biobrain.snn.network import Network

MODES = ("time_step", "event_driven")
P64 = NeuronParams(dtype="float64")


def schedule(steps, events: dict[int, list[int]], weight=2.0):
    per = [np.array(sorted(events.get(t, [])), dtype=np.int32) for t in range(steps)]
    indptr = np.concatenate([[0], np.cumsum([a.size for a in per])]).astype(np.int64)
    return InputSchedule(indptr, np.concatenate(per).astype(np.int32) if per else np.zeros(0, np.int32), weight, {})


def spike_times(res, neuron):
    indptr, neurons = res.raster
    return [t for t in range(res.steps) if neuron in neurons[indptr[t]:indptr[t + 1]]]


@pytest.mark.parametrize("mode", MODES)
def test_leak_follows_exact_exponential(mode):
    net = Network.from_edges(1, [], [], [])
    res = run(net, schedule(11, {0: [0]}, weight=0.5), P64, 11, mode, record_spikes=True)
    assert res.counters["spikes"] == 0
    assert res.final_v[0] == pytest.approx(0.5 * math.exp(-10 / 20), rel=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_threshold_reset_and_refractory(mode):
    p = NeuronParams(dtype="float64", refractory_ms=3)
    net = Network.from_edges(1, [], [], [])
    # forced at 0 -> spike; inputs at 1..3 fall in the refractory window; input at 4 spikes again
    res = run(net, schedule(8, {0: [0], 1: [0], 2: [0], 3: [0], 4: [0]}), p, 8, mode, record_spikes=True)
    assert spike_times(res, 0) == [0, 4]
    assert res.final_v[0] == 0.0  # reset value, no input afterwards


@pytest.mark.parametrize("mode", MODES)
def test_exact_threshold_crossing_spikes(mode):
    net = Network.from_edges(1, [], [], [])
    res = run(net, schedule(3, {0: [0]}, weight=1.0), P64, 3, mode, record_spikes=True)
    assert spike_times(res, 0) == [0]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("delay", [1, 2, 5])
def test_single_synapse_arrives_after_delay(mode, delay):
    p = NeuronParams(dtype="float64", delay_ms=delay)
    net = Network.from_edges(2, [0], [1], [1.5])
    res = run(net, schedule(12, {2: [0]}), p, 12, mode, record_spikes=True)
    assert spike_times(res, 0) == [2] and spike_times(res, 1) == [2 + delay]
    assert res.counters["synaptic_events"] == 2 - 1  # only neuron 0 has an outgoing edge


@pytest.mark.parametrize("mode", MODES)
def test_excitatory_and_inhibitory_weights_add(mode):
    # 0 -> 2 (+0.7), 1 -> 2 (-0.5), 3 -> 2 (+0.7): one E is subthreshold, E+E spikes, E+E+I = 0.9 does not
    net = Network.from_edges(4, [0, 1, 3], [2, 2, 2], [0.7, -0.5, 0.7])
    res = run(net, schedule(10, {0: [0], 3: [0, 3], 6: [0, 1, 3]}), P64, 10, mode, record_spikes=True)
    assert spike_times(res, 2) == [4]
    # reset at 4; inputs arrive at 7 (0.9); two more leak steps until the last step 9
    assert res.final_v[2] == pytest.approx(0.9 * math.exp(-2 / 20), rel=1e-12)


@pytest.mark.parametrize("mode", MODES)
def test_multi_synapse_weights_sum_within_a_step(mode):
    net = Network.from_edges(3, [0, 1], [2, 2], [0.5, 0.5])
    res = run(net, schedule(4, {0: [0, 1]}), P64, 4, mode, record_spikes=True)
    assert spike_times(res, 2) == [1]


@pytest.mark.parametrize("mode", MODES)
def test_edge_cases_single_neuron_empty_edges_no_input(mode):
    for n in (1, 5):
        res = run(Network.from_edges(n, [], [], []), schedule(20, {}), P64, 20, mode)
        assert res.counters["spikes"] == 0 and res.counters["synaptic_events"] == 0
        if mode == "event_driven":
            assert res.counters["neuron_updates"] == 0 and res.counters["skipped_neuron_updates"] == n * 20


@pytest.mark.parametrize("mode", MODES)
def test_cycle_reciprocal_pair_isolated_neuron(mode):
    # cycle 0->1->2->0 with suprathreshold weights keeps one spike circulating; 3<->4 reciprocal; 5 isolated
    net = Network.from_edges(6, [0, 1, 2, 3, 4], [1, 2, 0, 4, 3], [1.5] * 5)
    p = NeuronParams(dtype="float64", refractory_ms=1)
    res = run(net, schedule(12, {0: [0, 3]}), p, 12, mode, record_spikes=True)
    assert spike_times(res, 0) == [0, 3, 6, 9] and spike_times(res, 1) == [1, 4, 7, 10]
    assert spike_times(res, 3) == [0, 2, 4, 6, 8, 10] and spike_times(res, 4) == [1, 3, 5, 7, 9, 11]
    assert spike_times(res, 5) == []


@pytest.mark.parametrize("mode", MODES)
def test_all_neurons_forced_every_step_hit_the_refractory_ceiling(mode):
    n, steps = 50, 30
    p = NeuronParams(dtype="float64", refractory_ms=2)
    res = run(Network.from_edges(n, [], [], []), schedule(steps, {t: list(range(n)) for t in range(steps)}), p, steps, mode)
    assert res.counters["spikes"] == n * 10  # spike at 0, 3, 6, ... (every ref + 1 steps)
    assert res.counters["external_events"] == n * steps


@pytest.mark.parametrize("mode", MODES)
def test_high_degree_neuron_counts_every_delivery(mode):
    k = 3000
    net = Network.from_edges(k + 1, np.zeros(k, int), np.arange(1, k + 1), np.full(k, 0.1))
    res = run(net, schedule(6, {0: [0], 3: [0]}), P64, 6, mode)  # step 3: after the 2 ms refractory window
    assert res.counters["synaptic_events"] == 2 * k and res.counters["spikes"] == 2


def test_gather_matches_rows():
    net = Network.from_edges(4, [0, 0, 2, 3, 3, 3], [1, 2, 0, 0, 1, 2], [1, 2, 3, 4, 5, 6])
    t, w = gather(net.indptr, net.indices, net.weights, np.array([3, 0]))
    assert t.tolist() == [0, 1, 2, 1, 2] and w.tolist() == [4, 5, 6, 1, 2]


def random_network(n, p_edge, seed, dtype="float64", inhibitory=0.3, scale=0.6):
    rng = np.random.default_rng(seed)
    a = (rng.random((n, n)) < p_edge) & ~np.eye(n, dtype=bool)
    src, dst = np.nonzero(a)
    sign = np.where(rng.random(n) < inhibitory, -1.0, 1.0)
    return Network.from_edges(n, src, dst, sign[src] * rng.random(src.size) * scale, dtype=dtype)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_time_step_and_event_driven_are_equivalent(seed, dtype):
    n, steps = 200, 400
    p = NeuronParams(dtype=dtype, tau_m_ms=[10, 20, 40][seed % 3], refractory_ms=seed % 4, delay_ms=1 + seed % 3,
                     v_reset=[0.0, -0.2][seed % 2])
    net = random_network(n, 0.05, seed, dtype=dtype)
    sched = generate(InputSpec(rate=[0.002, 0.02, 0.1][seed % 3], weight=[1.5, 0.6][seed % 2]), n, steps, seed)
    ts = run(net, sched, p, steps, "time_step", record_spikes=True)
    ed = run(net, sched, p, steps, "event_driven", record_spikes=True)
    cmp = metrics.compare(ts, ed, metrics.V_TOLERANCE[dtype])
    assert ts.counters["spikes"] > 0
    assert cmp["equivalent"], cmp
    assert ed.counters["neuron_updates"] <= ts.counters["neuron_updates"]


@pytest.mark.parametrize("seed", range(3))
def test_auto_aggregation_matches_sparse_and_time_step(seed):
    n, steps = 150, 300
    p = NeuronParams(dtype="float64")
    net = random_network(n, 0.15, seed)
    sched = generate(InputSpec(rate=0.05), n, steps, seed)
    ts = run(net, sched, p, steps, "time_step", record_spikes=True)
    sparse = run(net, sched, p, steps, "event_driven", record_spikes=True, aggregation="sparse")
    auto = run(net, sched, p, steps, "event_driven", record_spikes=True, aggregation="auto", dense_ratio=0.05)
    assert auto.extra["dense_aggregation_steps"] > 0
    assert metrics.compare(ts, auto, 1e-9)["equivalent"] and metrics.compare(sparse, auto, 1e-12)["equivalent"]
    assert auto.counters["neuron_updates"] == sparse.counters["neuron_updates"]


def test_first_divergent_step_of_rasters():
    ptr, ids = np.array([0, 1, 1, 3]), np.array([4, 0, 2])
    assert metrics.first_divergent_step((ptr, ids), (ptr, ids.copy())) is None
    assert metrics.first_divergent_step((ptr, ids), (ptr, np.array([4, 0, 3]))) == 2  # same counts, other neuron
    assert metrics.first_divergent_step((ptr, ids), (np.array([0, 1, 2, 3]), np.array([4, 1, 2]))) == 1  # count differs
    assert metrics.first_divergent_step((ptr, ids), (np.array([0, 0, 0, 2]), np.array([0, 2]))) == 0

def test_deterministic_given_seed_and_different_across_seeds():
    net = random_network(150, 0.05, 1)
    a = run(net, generate(InputSpec(rate=0.02), 150, 300, 7), P64, 300, "event_driven", record_spikes=True)
    b = run(net, generate(InputSpec(rate=0.02), 150, 300, 7), P64, 300, "event_driven", record_spikes=True)
    c = run(net, generate(InputSpec(rate=0.02), 150, 300, 8), P64, 300, "event_driven", record_spikes=True)
    assert metrics.compare(a, b, 0.0)["equivalent"]
    assert not np.array_equal(a.raster[1], c.raster[1])


def test_counters_and_memory_accounting():
    net = random_network(120, 0.08, 3, dtype="float32")
    p = NeuronParams()
    sched = generate(InputSpec(rate=0.03), 120, 200, 3)
    for mode in MODES:
        res = run(net, sched, p, 200, mode, record_spikes=True)
        indptr, neurons = res.raster
        assert res.counters["synaptic_events"] == int(net.out_degree()[neurons].sum())
        assert res.counters["spikes"] == neurons.size == res.spike_counts.sum()
        assert res.counters["neuron_updates"] + res.counters["skipped_neuron_updates"] == 120 * 200
        assert res.memory["topology"] == net.indptr.nbytes + net.indices.nbytes
        assert res.memory["synapse_state"] == net.weights.nbytes
        assert res.counters["external_events"] == sched.events
    ts = run(net, sched, p, 200, "time_step")
    assert ts.memory["event_buffers"] == p.delay_steps * 120 * 4 and ts.memory["neuron_state"] == 120 * 8
    ed = run(net, sched, p, 200, "event_driven")
    assert ed.memory["neuron_state"] == 120 * 12


def test_metrics_summary_and_regimes():
    p = NeuronParams(refractory_ms=2)
    silent = run(Network.from_edges(10, [], [], []), schedule(100, {}), p, 100, "time_step")
    s = metrics.summarize(silent, p)
    assert s["population_rate_hz"] == 0 and s["silent_neurons"] == 10 and s["refractory_occupancy"] == 0
    forced = run(Network.from_edges(10, [], [], []), schedule(60, {t: list(range(10)) for t in range(30)}), p, 60, "time_step")
    assert metrics.classify(forced, p, 30, external_on=300)["regime"] == "SATURATED"
    feed = run(Network.from_edges(10, [], [], []), schedule(60, {t: [t % 10] for t in range(30)}), p, 60, "time_step")
    assert metrics.classify(feed, p, 30, external_on=30)["regime"] == "DEAD"
