"""One LIF model, two execution modes: time-step and event-driven.

Per step t, identically in both modes:
  1. leak       v ← v·α                     (event-driven: only neurons that receive input, as v·α^k)
  2. synaptic   v ← v + Σ weights of spikes emitted at t − d
  3. external   v ← v + w_in for neurons with an external event at t
  4. refractory neurons (t − last_spike ≤ ref) are held at v_reset
  5. v ≥ θ → spike: v ← v_reset, last_spike ← t, events scheduled for t + d

Potentials are dimensionless (θ = 1 by default), rest is 0. Without input a neuron only decays towards 0,
so it cannot cross θ: this is what lets the event-driven mode skip it exactly.

Counters (both modes): neuron_updates = neurons whose state was computed in a step (time-step: all N);
synaptic_events = presynaptic spike × postsynaptic edge deliveries (one weight accumulation each, i.e. the
synaptic operations); external_events = input events. No FLOPs are claimed.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import NeuronParams
from .inputs import InputSchedule
from .network import Network

SENTINEL = -(2 ** 30)


def gather(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray, rows: np.ndarray):
    """Targets and weights of all outgoing edges of `rows`, in row order."""
    if rows.size == 1:
        s, e = indptr[rows[0]], indptr[rows[0] + 1]
        return indices[s:e], weights[s:e]
    lo = indptr[rows]
    counts = indptr[rows + 1] - lo
    total = int(counts.sum())
    idx = np.repeat(lo - (np.cumsum(counts) - counts), counts) + np.arange(total)
    return indices[idx], weights[idx]


class _Phases:
    def __init__(self, enabled: bool):
        self.acc: dict[str, float] = defaultdict(float)
        self.t = time.perf_counter()
        self.lap = self._lap if enabled else (lambda name: None)

    def _lap(self, name: str) -> None:
        now = time.perf_counter()
        self.acc[name] += now - self.t
        self.t = now


@dataclass
class RunResult:
    mode: str
    n: int
    m: int
    steps: int
    counters: dict
    spikes_per_step: np.ndarray
    updates_per_step: np.ndarray
    events_per_step: np.ndarray
    spike_counts: np.ndarray
    final_v: np.ndarray
    wall_s: float
    memory: dict
    raster: tuple[np.ndarray, np.ndarray] | None = None
    voltage: list[dict] | None = None
    phases: dict | None = None
    extra: dict = field(default_factory=dict)


def _raster(chunks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([c.size for c in chunks], dtype=np.int64)
    return np.concatenate([[0], np.cumsum(counts)]), (np.concatenate(chunks).astype(np.int32) if chunks else np.zeros(0, np.int32))


def _finish(mode, net, steps, spikes, updates, events, spike_counts, final_v, wall, memory, chunks, voltage, phases):
    counters = {"steps": steps, "neuron_updates": int(updates.sum()), "spikes": int(spikes.sum()),
                "synaptic_events": int(events.sum()), "external_events": memory.pop("_external_events"),
                "neuron_step_slots": steps * net.n}
    counters["skipped_neuron_updates"] = counters["neuron_step_slots"] - counters["neuron_updates"]
    return RunResult(mode, net.n, net.m, steps, counters, spikes, updates, events, spike_counts, final_v, wall, memory,
                     _raster(chunks) if chunks is not None else None, voltage, dict(phases.acc) if phases else None)


def run_time_step(net: Network, schedule: InputSchedule, p: NeuronParams, steps: int, *, record_spikes: bool = False,
                  record_voltage_every: int = 0, profile: bool = False) -> RunResult:
    n, dt = net.n, np.dtype(p.dtype)
    ref, d = p.ref_steps, p.delay_steps
    alpha, theta, v_reset, w_in = dt.type(p.alpha), dt.type(p.v_threshold), dt.type(p.v_reset), dt.type(schedule.weight)
    v = np.zeros(n, dtype=dt)
    last_spike = np.full(n, SENTINEL, dtype=np.int32)
    ring = np.zeros((d, n), dtype=dt)  # ring[t % d]: synaptic input due at step t
    recent: list[np.ndarray] = []      # spikes of the last `ref` steps (they are refractory now)
    spikes = np.zeros(steps, dtype=np.int32)
    events = np.zeros(steps, dtype=np.int64)
    spike_counts = np.zeros(n, dtype=np.int32)
    chunks = [] if record_spikes else None
    voltage = [] if record_voltage_every else None
    indptr, indices, weights = net.indptr, net.indices, net.weights
    external = 0
    ph = _Phases(profile)
    t0 = time.perf_counter()
    ph.t = t0
    for t in range(steps):
        slot = ring[t % d]
        v *= alpha
        ph.lap("leak")
        v += slot
        slot[:] = 0
        ph.lap("synaptic_input")
        ext = schedule.neurons[schedule.indptr[t]:schedule.indptr[t + 1]]
        if ext.size:
            v[ext] += w_in
            external += ext.size
        ph.lap("external_input")
        if recent:
            v[recent[0] if len(recent) == 1 else np.concatenate(recent)] = v_reset
        ph.lap("refractory")
        spk = np.flatnonzero(v >= theta)
        ph.lap("threshold")
        if ref:
            recent.append(spk)
            if len(recent) > ref:
                recent.pop(0)
        if spk.size:
            v[spk] = v_reset
            last_spike[spk] = t
            spike_counts[spk] += 1
            spikes[t] = spk.size
            targets, w = gather(indptr, indices, weights, spk)
            ph.lap("reset_and_gather")
            if targets.size:
                events[t] = targets.size
                ring[(t + d) % d] += np.bincount(targets, weights=w, minlength=n)
            ph.lap("deliver")
        if chunks is not None:
            chunks.append(spk)
        if voltage is not None and t % record_voltage_every == 0:
            voltage.append({"step": t, "mean": float(v.mean()), "max": float(v.max())})
        ph.lap("record")
    wall = time.perf_counter() - t0
    memory = {"neuron_state": int(v.nbytes + last_spike.nbytes), "event_buffers": int(ring.nbytes),
              "input_schedule": schedule.nbytes, "instrumentation": int(spikes.nbytes + events.nbytes + spike_counts.nbytes),
              **net.memory(), "_external_events": external}
    updates = np.full(steps, n, dtype=np.int64)
    return _finish("time_step", net, steps, spikes, updates, events, spike_counts, v.copy(), wall, memory, chunks, voltage,
                   ph if profile else None)


def _decay_table(p: NeuronParams, steps: int) -> np.ndarray:
    """α^k for k = 0..K, then 0: beyond K the remaining potential is below 1e-30 of its value."""
    k_max = min(steps + 1, int(np.ceil(70 * p.tau_m_ms / p.dt_ms)))
    table = np.power(p.alpha, np.arange(k_max + 1, dtype=np.float64))
    table[-1] = 0.0
    return table.astype(p.dtype)


def _lazy(v, t_last, last_spike, t, p: NeuronParams, decay: np.ndarray) -> np.ndarray:
    """Potential at step t (after the leak of step t, before its inputs) from the stored state."""
    ref_end = last_spike.astype(np.int64) + p.ref_steps
    from_reset = ref_end >= t_last
    k = np.where(from_reset, t - ref_end, t - t_last)
    k = np.clip(k, 0, decay.size - 1)
    base = np.where(from_reset, v.dtype.type(p.v_reset), v)
    return base * decay[k]


def run_event_driven(net: Network, schedule: InputSchedule, p: NeuronParams, steps: int, *, record_spikes: bool = False,
                     record_voltage_every: int = 0, profile: bool = False, aggregation: str = "sparse",
                     dense_ratio: float = 0.125) -> RunResult:
    """aggregation="sparse": arriving events are grouped by sorting (no O(N) work in a step).
    aggregation="auto": steps with more than dense_ratio·N arriving events use one O(N) bincount + mask instead;
    neuron updates stay limited to neurons that receive input in both variants."""
    if aggregation not in ("sparse", "auto"):
        raise ValueError("aggregation must be 'sparse' or 'auto'")
    n, dt = net.n, np.dtype(p.dtype)
    ref, d = p.ref_steps, p.delay_steps
    theta, v_reset, w_in = dt.type(p.v_threshold), dt.type(p.v_reset), dt.type(schedule.weight)
    decay = _decay_table(p, steps)
    v = np.zeros(n, dtype=dt)
    last_spike = np.full(n, SENTINEL, dtype=np.int32)
    t_last = np.zeros(n, dtype=np.int32)
    ring: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(d)]
    spikes = np.zeros(steps, dtype=np.int32)
    updates = np.zeros(steps, dtype=np.int64)
    events = np.zeros(steps, dtype=np.int64)
    spike_counts = np.zeros(n, dtype=np.int32)
    chunks = [] if record_spikes else None
    voltage = [] if record_voltage_every else None
    indptr, indices, weights = net.indptr, net.indices, net.weights
    empty = np.zeros(0, dtype=np.int32)
    external, pending, peak_pending, dense_steps = 0, 0, 0, 0
    dense_threshold = dense_ratio * n if aggregation == "auto" else np.inf
    ph = _Phases(profile)
    t0 = time.perf_counter()
    ph.t = t0
    for t in range(steps):
        batches = ring[t % d]
        ring[t % d] = []
        ext = schedule.neurons[schedule.indptr[t]:schedule.indptr[t + 1]]
        external += ext.size
        if batches:
            pending -= sum(b[0].size for b in batches)
            targets = batches[0][0] if len(batches) == 1 else np.concatenate([b[0] for b in batches])
            w = batches[0][1] if len(batches) == 1 else np.concatenate([b[1] for b in batches])
            if targets.size > dense_threshold:
                dense_steps += 1
                hit = np.zeros(n, dtype=bool)
                hit[targets] = True
                syn_u = np.flatnonzero(hit)
                syn_in = np.bincount(targets, weights=w, minlength=n)[syn_u].astype(dt)
            else:
                syn_u, inverse = np.unique(targets, return_inverse=True)
                syn_in = np.bincount(inverse, weights=w).astype(dt)
        else:
            syn_u = empty
        ph.lap("aggregate")
        if not syn_u.size and not ext.size:
            if chunks is not None:
                chunks.append(empty)
            if voltage is not None and t % record_voltage_every == 0:
                full = _lazy(v, t_last, last_spike, t, p, decay)
                voltage.append({"step": t, "mean": float(full.mean()), "max": float(full.max())})
            continue
        touched = syn_u if not ext.size else (ext if not syn_u.size else np.union1d(syn_u, ext))
        ls = last_spike[touched]
        vv = _lazy(v[touched], t_last[touched], ls, t, p, decay)
        ph.lap("leak")
        if syn_u.size:
            vv[np.searchsorted(touched, syn_u)] += syn_in
        if ext.size:
            vv[np.searchsorted(touched, ext)] += w_in
        ph.lap("input")
        if ref:
            vv[(t - ls) <= ref] = v_reset
        local = np.flatnonzero(vv >= theta)
        ph.lap("refractory_threshold")
        spk = touched[local]
        vv[local] = v_reset
        v[touched] = vv
        t_last[touched] = t
        updates[t] = touched.size
        if spk.size:
            last_spike[spk] = t
            spike_counts[spk] += 1
            spikes[t] = spk.size
            targets, w = gather(indptr, indices, weights, spk)
            if targets.size:
                events[t] = targets.size
                ring[(t + d) % d].append((targets, w))
                pending += targets.size
                peak_pending = max(peak_pending, pending)
        ph.lap("reset_and_schedule")
        if chunks is not None:
            chunks.append(spk)
        if voltage is not None and t % record_voltage_every == 0:
            full = _lazy(v, t_last, last_spike, t, p, decay)
            voltage.append({"step": t, "mean": float(full.mean()), "max": float(full.max())})
        ph.lap("record")
    wall = time.perf_counter() - t0
    final_v = _lazy(v, t_last, last_spike, steps - 1, p, decay) if steps else v
    # after the last step, neurons updated at steps-1 already hold their post-step value
    final_v = np.where(t_last == steps - 1, v, final_v).astype(dt)
    event_bytes = np.dtype(np.int32).itemsize + dt.itemsize
    memory = {"neuron_state": int(v.nbytes + last_spike.nbytes + t_last.nbytes), "decay_table": int(decay.nbytes),
              "event_buffers": int(peak_pending * event_bytes), "input_schedule": schedule.nbytes,
              "instrumentation": int(spikes.nbytes + events.nbytes + updates.nbytes + spike_counts.nbytes),
              **net.memory(), "_external_events": external}
    result = _finish("event_driven", net, steps, spikes, updates, events, spike_counts, final_v, wall, memory, chunks, voltage,
                     ph if profile else None)
    result.extra = {"aggregation": aggregation, "dense_aggregation_steps": dense_steps}
    return result


def run(net: Network, schedule: InputSchedule, p: NeuronParams, steps: int, mode: str, backend: str = "numpy", **kw) -> RunResult:
    """backend "numpy": the two loops above; "numba": compiled.py (same model, same arithmetic order)."""
    if backend == "numba":
        from . import compiled

        return compiled.run(net, schedule, p, steps, mode, **kw)
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    kw.pop("variant", None)
    if mode == "time_step":
        kw.pop("aggregation", None)
        return run_time_step(net, schedule, p, steps, **kw)
    return run_event_driven(net, schedule, p, steps, **kw)
