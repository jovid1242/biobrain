"""Compiled (Numba) backend: the conceptual engine tests on every compiled kernel, cross-backend bit-exactness,
cross-mode equivalence, recording, profiling and no allocations inside the simulation loop."""

import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("numba")

import test_snn_engine as conceptual  # noqa: E402  (same directory; its tests are reused, not re-collected)

from biobrain.snn import compiled, engine, metrics  # noqa: E402
from biobrain.snn.config import InputSpec, NeuronParams  # noqa: E402
from biobrain.snn.inputs import generate  # noqa: E402
from biobrain.snn.network import Network  # noqa: E402

COMPILED = [("time_step", "touched"), ("event_driven", "touched"), ("event_driven", "dense"), ("event_driven", "copy")]
CONCEPTUAL = {
    "test_leak_follows_exact_exponential": [{}],
    "test_threshold_reset_and_refractory": [{}],
    "test_exact_threshold_crossing_spikes": [{}],
    "test_single_synapse_arrives_after_delay": [{"delay": 1}, {"delay": 2}, {"delay": 5}],
    "test_excitatory_and_inhibitory_weights_add": [{}],
    "test_multi_synapse_weights_sum_within_a_step": [{}],
    "test_edge_cases_single_neuron_empty_edges_no_input": [{}],
    "test_cycle_reciprocal_pair_isolated_neuron": [{}],
    "test_all_neurons_forced_every_step_hit_the_refractory_ceiling": [{}],
    "test_high_degree_neuron_counts_every_delivery": [{}],
}


def test_every_mode_parametrised_conceptual_test_is_reused():
    with_mode = {name for name, fn in vars(conceptual).items()
                 if name.startswith("test_") and callable(fn) and "mode" in inspect.signature(fn).parameters}
    assert with_mode == set(CONCEPTUAL)


@pytest.mark.parametrize("mode,variant", COMPILED)
@pytest.mark.parametrize("name", sorted(CONCEPTUAL))
def test_conceptual_suite_on_compiled_kernels(name, mode, variant, monkeypatch):
    def compiled_run(net, sched, p, steps, m, **kw):
        kw.pop("aggregation", None)
        kw.pop("dense_ratio", None)
        return engine.run(net, sched, p, steps, m, backend="numba", variant=variant, **kw)

    monkeypatch.setattr(conceptual, "run", compiled_run)
    for extra in CONCEPTUAL[name]:
        getattr(conceptual, name)(mode=mode, **extra)


def _bit_identical(a, b) -> None:
    assert np.array_equal(a.raster[0], b.raster[0]) and np.array_equal(a.raster[1], b.raster[1])
    for field in ("spikes_per_step", "updates_per_step", "events_per_step", "spike_counts"):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field
    assert a.final_v.dtype == b.final_v.dtype and a.final_v.tobytes() == b.final_v.tobytes()
    assert a.counters == b.counters
    assert a.voltage == b.voltage


def _case(seed, dtype, n=200, steps=400):
    p = NeuronParams(dtype=dtype, tau_m_ms=[10, 20, 40][seed % 3], refractory_ms=seed % 4, delay_ms=1 + seed % 3,
                     v_reset=[0.0, -0.2][seed % 2])
    net = conceptual.random_network(n, 0.05, seed, dtype=dtype)
    sched = generate(InputSpec(rate=[0.002, 0.02, 0.1][seed % 3], weight=[1.5, 0.6][seed % 2]), n, steps, seed)
    return net, sched, p, steps


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_compiled_is_bit_identical_to_numpy_in_the_same_mode(seed, dtype):
    net, sched, p, steps = _case(seed, dtype)
    kw = dict(record_spikes=True, record_voltage_every=7)
    ts_np = engine.run(net, sched, p, steps, "time_step", **kw)
    ts_nb = engine.run(net, sched, p, steps, "time_step", backend="numba", **kw)
    assert ts_np.counters["spikes"] > 0
    _bit_identical(ts_np, ts_nb)
    ed_np = engine.run(net, sched, p, steps, "event_driven", **kw)
    for variant in compiled.VARIANTS:
        _bit_identical(ed_np, engine.run(net, sched, p, steps, "event_driven", backend="numba", variant=variant, **kw))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_compiled_time_step_and_event_driven_are_equivalent(seed, dtype):
    net, sched, p, steps = _case(seed, dtype)
    ts = engine.run(net, sched, p, steps, "time_step", backend="numba", record_spikes=True)
    ed = engine.run(net, sched, p, steps, "event_driven", backend="numba", record_spikes=True)
    cmp = metrics.compare(ts, ed, metrics.V_TOLERANCE[dtype])
    assert cmp["equivalent"], cmp
    assert ed.counters["neuron_updates"] <= ts.counters["neuron_updates"]


def test_raster_buffer_growth_resumes_without_changing_the_run():
    n, steps = 60, 300
    net = conceptual.random_network(n, 0.3, 4)
    sched = generate(InputSpec(rate=0.5), n, steps, 4)  # thousands of spikes: the 1,024-slot buffer must grow
    for mode in ("time_step", "event_driven"):
        a = engine.run(net, sched, conceptual.P64, steps, mode, record_spikes=True)
        b = engine.run(net, sched, conceptual.P64, steps, mode, backend="numba", record_spikes=True)
        assert b.counters["spikes"] > 1024
        _bit_identical(a, b)


def test_compiled_deterministic_and_seed_dependent():
    net = conceptual.random_network(150, 0.05, 1)
    runs = [engine.run(net, generate(InputSpec(rate=0.02), 150, 300, s), conceptual.P64, 300, "event_driven", backend="numba",
                       record_spikes=True) for s in (7, 7, 8)]
    _bit_identical(runs[0], runs[1])
    assert not np.array_equal(runs[0].raster[1], runs[2].raster[1])


def test_profile_mode_reports_phases_and_changes_nothing():
    net, sched, p, steps = _case(1, "float32")
    for mode, phases in (("time_step", compiled.TS_PHASES), ("event_driven", compiled.ED_PHASES)):
        plain = engine.run(net, sched, p, steps, mode, backend="numba", record_spikes=True)
        prof = engine.run(net, sched, p, steps, mode, backend="numba", record_spikes=True, profile=True)
        _bit_identical(plain, prof)
        assert set(prof.phases) == set(phases) and all(v >= 0 for v in prof.phases.values())
        assert sum(prof.phases.values()) <= prof.wall_s * 1.05


def test_compiled_counters_and_memory_accounting():
    net = conceptual.random_network(120, 0.08, 3, dtype="float32")
    p = NeuronParams()
    sched = generate(InputSpec(rate=0.03), 120, 200, 3)
    ed = engine.run(net, sched, p, 200, "event_driven", backend="numba", record_spikes=True)
    indptr, neurons = ed.raster
    assert ed.counters["synaptic_events"] == int(net.out_degree()[neurons].sum())
    assert 0 < ed.extra["unique_targets"] <= ed.counters["synaptic_events"]
    assert ed.memory["neuron_state"] == 120 * 12 and ed.memory["synapse_state"] == net.weights.nbytes
    ts = engine.run(net, sched, p, 200, "time_step", backend="numba")
    assert ts.memory["neuron_state"] == 120 * 8 and ts.memory["event_buffers"] == p.delay_steps * 120 * 8


@pytest.mark.parametrize("k", [0, 1, 5, 48, 49, 700, 5000])
@pytest.mark.parametrize("n", [100, 139_255])
def test_allocation_free_sort(k, n):
    rng = np.random.default_rng(k)
    a = rng.integers(0, n, max(k, 1)).astype(np.int32)
    expected = np.sort(a[:k])
    compiled._sort_ids(a, k, np.zeros_like(a), np.zeros(513, np.int64), compiled._radix_passes(n))
    assert np.array_equal(a[:k], expected)


_ALLOCATIONS = """
import json, numpy as np
from numba.core.runtime import rtsys
from biobrain.snn import compiled
from biobrain.snn.config import InputSpec, NeuronParams
from biobrain.snn.inputs import generate
from biobrain.snn.network import Network
rng = np.random.default_rng(0)
n = 400
net = Network.from_edges(n, rng.integers(0, n, 12000), rng.integers(0, n, 12000), rng.normal(0.05, 0.3, 12000), dtype="float32")
p = NeuronParams()
out = {}
for mode, variant in (("time_step", "touched"), ("event_driven", "touched"), ("event_driven", "dense"), ("event_driven", "copy")):
    counts = []
    for steps in (50, 50, 800):
        sched = generate(InputSpec(rate=0.05), n, steps, 1)
        compiled.run(net, sched, p, steps, mode, variant=variant)
        before = rtsys.get_allocation_stats().alloc
        res = compiled.run(net, sched, p, steps, mode, variant=variant)
        counts.append([rtsys.get_allocation_stats().alloc - before, res.counters["spikes"]])
    out[mode + "/" + variant] = counts
print(json.dumps(out))
"""


def test_no_allocation_inside_the_simulation_loop():
    """Numba NRT allocation counts of a whole run must not grow with the number of steps (50 vs 800)."""
    env = {**os.environ, "NUMBA_NRT_STATS": "1"}
    proc = subprocess.run([sys.executable, "-c", _ALLOCATIONS], capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name, ((a50, s50), _, (a800, s800)) in json.loads(proc.stdout.strip().splitlines()[-1]).items():
        assert s800 > s50 > 0, name
        assert a800 == a50, (name, a50, a800)
