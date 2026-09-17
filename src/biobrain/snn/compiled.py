"""Compiled execution (Numba: CPU, one thread, no fastmath) of exactly the model in engine.py.

The kernels reproduce the NumPy engine operation by operation, so a compiled run is meant to be bit-identical to the
NumPy run of the same mode (checked by tests and by experiments, never assumed):

- time-step: v·α in the state dtype on every step; synaptic input summed per target in float64 in event order (what
  np.bincount does), cast to the state dtype once; then external input, refractory hold, threshold.
- event-driven: v = base·α^k from the same decay table; the same per-target float64 sums in the same event order,
  because the spikes of a step are kept in ascending neuron order, as np.flatnonzero / sorted unions give them.

No array is allocated inside the simulation loop: every buffer exists before a kernel is called. A kernel only
returns early when a spike raster is being recorded and its buffer may overflow; the caller grows the buffer and
resumes at the same step, whose state has not been touched yet.

Event-driven accumulation variants (all produce the same result):
  touched: accumulator[N] + list of first-touched targets; only those neurons are updated and cleared (default)
  dense:   accumulator[N] + one O(N) scan per step with input (no list, no sort)
  copy:    like `touched`, but the (target, weight) events are copied into a ring buffer at spike time instead of
           walking the CSR rows of the spiking neurons at delivery time
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit, types
from numba.extending import overload

from .config import NeuronParams
from .engine import SENTINEL, RunResult, _decay_table, _lazy
from .inputs import InputSchedule
from .network import Network

VARIANTS = ("touched", "dense", "copy")
TS_PHASES = ("external_mark", "neuron_pass", "delivery", "record")
ED_PHASES = ("delivery", "external_mark", "neuron_update", "spike_sort", "schedule", "record")

_clock = types.ExternalFunction("clock_gettime_nsec_np", types.uint64(types.int32))  # macOS; profile timers only
_CLOCK_MONOTONIC_RAW = 4


@njit(cache=True)
def _now():
    return np.int64(_clock(_CLOCK_MONOTONIC_RAW))


def _to_state(value, like):
    raise NotImplementedError  # compiled-only helper


@overload(_to_state, inline="always")
def _to_state_impl(value, like):
    """Cast a float64 sum to the state dtype, as `.astype(dt)` / a float32 ring slot does in the NumPy engine."""
    if like.dtype == types.float32:
        return lambda value, like: np.float32(value)
    return lambda value, like: np.float64(value)


@njit(cache=True)
def clock_cost_ns(reps):
    t0 = _now()
    for _ in range(reps):
        _now()
    return (_now() - t0) / reps


@njit(cache=True)
def _sort_ids(a, k, tmp, counts, passes):
    """Ascending sort of a[:k] without allocation: insertion sort for short runs, else LSD radix sort (9-bit digits)."""
    if k <= 48:
        for i in range(1, k):
            x = a[i]
            j = i - 1
            while j >= 0 and a[j] > x:
                a[j + 1] = a[j]
                j -= 1
            a[j + 1] = x
        return
    src, dst = a, tmp
    shift = 0
    for _ in range(passes):
        for b in range(513):
            counts[b] = 0
        for i in range(k):
            counts[((src[i] >> shift) & 511) + 1] += 1
        for b in range(1, 513):
            counts[b] += counts[b - 1]
        for i in range(k):
            digit = (src[i] >> shift) & 511
            dst[counts[digit]] = src[i]
            counts[digit] += 1
        src, dst = dst, src
        shift += 9
    if passes % 2 == 1:
        for i in range(k):
            a[i] = src[i]


def _radix_passes(n: int) -> int:
    return max(1, -(-max(n - 1, 1).bit_length() // 9))


# ---- time-step -------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _ts_kernel(t_start, t_end, indptr, indices, weights, sched_indptr, sched_neurons, d, ref, alpha, theta, v_reset, w_in,
               v, last_spike, acc, ext_add, spk, spikes, events, spike_counts,
               record, rec_ptr, rec_ids, rec_used, volt_every, volt_buf, profile, timers):
    n = v.size
    for t in range(t_start, t_end):
        if record and rec_used[0] + n > rec_ids.size:
            return t
        c0 = _now() if profile else 0
        slot = t % d
        e0, e1 = sched_indptr[t], sched_indptr[t + 1]
        for q in range(e0, e1):
            ext_add[sched_neurons[q]] = w_in
        c1 = _now() if profile else 0
        ns = 0
        for i in range(n):
            x = v[i] * alpha
            x = x + _to_state(acc[slot, i], v)
            acc[slot, i] = 0.0
            x = x + ext_add[i]
            x = v_reset if (ref > 0) & (t - last_spike[i] <= ref) else x
            fired = x >= theta
            spk[ns] = i  # branch-free append: the slot is overwritten unless the neuron fired
            ns += fired
            v[i] = v_reset if fired else x
        c2 = _now() if profile else 0
        for q in range(e0, e1):
            ext_add[sched_neurons[q]] = 0.0
        c2b = _now() if profile else 0
        ev = 0
        for q in range(ns):
            i = spk[q]
            last_spike[i] = t
            spike_counts[i] += 1
            for e in range(indptr[i], indptr[i + 1]):
                acc[slot, indices[e]] += weights[e]
            ev += indptr[i + 1] - indptr[i]
        spikes[t] = ns
        events[t] = ev
        c3 = _now() if profile else 0
        if record:
            used = rec_used[0]
            for q in range(ns):
                rec_ids[used + q] = spk[q]
            rec_used[0] = used + ns
            rec_ptr[t + 1] = used + ns
        if volt_every > 0 and t % volt_every == 0:
            row = t // volt_every
            for i in range(n):
                volt_buf[row, i] = v[i]
        if profile:
            c4 = _now()
            timers[0] += (c1 - c0) + (c2b - c2)  # external marks are set before and cleared after the pass
            timers[1] += c2 - c1
            timers[2] += c3 - c2b
            timers[3] += c4 - c3
    return t_end


# ---- event-driven ------------------------------------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _ed_update(j, t, ref, kmax, v, t_last, last_spike, acc, flags, decay, v_reset, w_in, zero, theta):
    """One touched neuron at step t, as engine._lazy + the input/refractory/threshold lines of run_event_driven.

    Written with selects instead of branches. Adding the cleared accumulator (+0.0) or `zero` to a neuron without that
    input leaves the value bit-identical: the only float x with x + 0.0 != x is -0.0, which cannot reach these
    additions except as +/-0.0 before a non-zero external weight (tests check bit identity with the NumPy engine)."""
    ls = last_spike[j]
    ref_end = np.int64(ls) + ref
    tl = t_last[j]
    from_reset = ref_end >= tl
    k = t - ref_end if from_reset else t - tl
    k = min(max(k, 0), kmax)
    x = (v_reset if from_reset else v[j]) * decay[k]
    x = x + _to_state(acc[j], v)
    acc[j] = 0.0
    x = x + (w_in if flags[j] & 2 else zero)
    flags[j] = 0
    x = v_reset if (ref > 0) & (t - ls <= ref) else x
    fired = x >= theta
    v[j] = v_reset if fired else x
    t_last[j] = t
    return fired


@njit(cache=True)
def _ed_voltage_row(t, ref, kmax, v, t_last, last_spike, decay, v_reset, out):
    """engine._lazy for every neuron (only on voltage-recording steps)."""
    for j in range(v.size):
        ref_end = np.int64(last_spike[j]) + ref
        if ref_end >= t_last[j]:
            k = t - ref_end
            x = v_reset
        else:
            k = t - t_last[j]
            x = v[j]
        if k < 0:
            k = 0
        elif k > kmax:
            k = kmax
        out[j] = x * decay[k]


@njit(cache=True)
def _ed_kernel(t_start, t_end, variant, indptr, indices, weights, sched_indptr, sched_neurons, d, ref, theta, v_reset, w_in, zero,
               decay, v, last_spike, t_last, acc, flags, touched, ring_ids, ring_count, ev_tgt, ev_w, ev_count, spk,
               sort_tmp, sort_counts, radix_passes, spikes, updates, events, unique_targets, spike_counts, pending,
               record, rec_ptr, rec_ids, rec_used, volt_every, volt_buf, profile, timers):
    """variant: 0 = touched (CSR walk at delivery), 1 = dense scan, 2 = touched with copied events."""
    n = v.size
    kmax = decay.size - 1
    for t in range(t_start, t_end):
        if record and rec_used[0] + n > rec_ids.size:
            return t
        c0 = _now() if profile else 0
        slot = t % d
        na = 0
        n_syn = 0
        if variant == 0:
            for q in range(ring_count[slot]):
                s = ring_ids[slot, q]
                for e in range(indptr[s], indptr[s + 1]):
                    j = indices[e]
                    touched[na] = j
                    na += flags[j] == 0
                    flags[j] = 1
                    acc[j] += weights[e]
                pending[0] -= indptr[s + 1] - indptr[s]
            ring_count[slot] = 0
            n_syn = na
        elif variant == 1:
            for q in range(ring_count[slot]):
                s = ring_ids[slot, q]
                for e in range(indptr[s], indptr[s + 1]):
                    j = indices[e]
                    n_syn += flags[j] == 0
                    flags[j] = 1
                    acc[j] += weights[e]
                pending[0] -= indptr[s + 1] - indptr[s]
            ring_count[slot] = 0
        else:
            ne = ev_count[slot]
            for q in range(ne):
                j = ev_tgt[slot, q]
                touched[na] = j
                na += flags[j] == 0
                flags[j] = 1
                acc[j] += ev_w[slot, q]
            pending[0] -= ne
            ev_count[slot] = 0
            n_syn = na
        unique_targets[t] = n_syn
        c1 = _now() if profile else 0
        e0, e1 = sched_indptr[t], sched_indptr[t + 1]
        n_ext_new = 0
        for q in range(e0, e1):
            j = sched_neurons[q]
            new = flags[j] == 0
            touched[na] = j  # harmless for the dense variant (buffer has N slots, na stays 0 there)
            na += new and variant != 1
            n_ext_new += new
            flags[j] |= 2
        c2 = _now() if profile else 0
        total = na if variant != 1 else n_syn + n_ext_new
        if total == 0:
            if volt_every > 0 and t % volt_every == 0:
                _ed_voltage_row(t, ref, kmax, v, t_last, last_spike, decay, v_reset, volt_buf[t // volt_every])
            if record:
                rec_ptr[t + 1] = rec_used[0]
            if profile:
                c5 = _now()
                timers[0] += c1 - c0
                timers[1] += c2 - c1
                timers[5] += c5 - c2
            continue
        ns = 0
        if variant == 1:
            for j in range(n):
                if flags[j] != 0:
                    fired = _ed_update(j, t, ref, kmax, v, t_last, last_spike, acc, flags, decay, v_reset, w_in, zero, theta)
                    spk[ns] = j
                    ns += fired
        else:
            for q in range(na):
                j = touched[q]
                fired = _ed_update(j, t, ref, kmax, v, t_last, last_spike, acc, flags, decay, v_reset, w_in, zero, theta)
                spk[ns] = j
                ns += fired
        updates[t] = total
        c3 = _now() if profile else 0
        if variant != 1:
            _sort_ids(spk, ns, sort_tmp, sort_counts, radix_passes)
        c4 = _now() if profile else 0
        ev = 0
        if variant == 2:
            ne = 0
            for q in range(ns):
                j = spk[q]
                last_spike[j] = t
                spike_counts[j] += 1
                for e in range(indptr[j], indptr[j + 1]):
                    ev_tgt[slot, ne] = indices[e]
                    ev_w[slot, ne] = weights[e]
                    ne += 1
            ev_count[slot] = ne
            ev = ne
        else:
            for q in range(ns):
                j = spk[q]
                last_spike[j] = t
                spike_counts[j] += 1
                ring_ids[slot, q] = j
                ev += indptr[j + 1] - indptr[j]
            ring_count[slot] = ns
        spikes[t] = ns
        events[t] = ev
        pending[0] += ev
        if pending[0] > pending[1]:
            pending[1] = pending[0]
        c5 = _now() if profile else 0
        if record:
            used = rec_used[0]
            for q in range(ns):
                rec_ids[used + q] = spk[q]
            rec_used[0] = used + ns
            rec_ptr[t + 1] = used + ns
        if volt_every > 0 and t % volt_every == 0:
            _ed_voltage_row(t, ref, kmax, v, t_last, last_spike, decay, v_reset, volt_buf[t // volt_every])
        if profile:
            c6 = _now()
            timers[0] += c1 - c0
            timers[1] += c2 - c1
            timers[2] += c3 - c2
            timers[3] += c4 - c3
            timers[4] += c5 - c4
            timers[5] += c6 - c5
    return t_end


# ---- Python wrappers ---------------------------------------------------------------------------------------------------
def _grow(buf: np.ndarray, needed: int) -> np.ndarray:
    out = np.zeros(max(2 * buf.size, needed), dtype=buf.dtype)
    out[:buf.size] = buf
    return out


def _voltage_list(volt_buf: np.ndarray, every: int, steps: int) -> list[dict] | None:
    if not every:
        return None
    return [{"step": t, "mean": float(volt_buf[t // every].mean()), "max": float(volt_buf[t // every].max())}
            for t in range(0, steps, every)]


def _result(mode, net, steps, spikes, updates, events, spike_counts, final_v, wall, memory, external, raster, voltage,
            phases, extra) -> RunResult:
    counters = {"steps": steps, "neuron_updates": int(updates.sum()), "spikes": int(spikes.sum()),
                "synaptic_events": int(events.sum()), "external_events": external, "neuron_step_slots": steps * net.n}
    counters["skipped_neuron_updates"] = counters["neuron_step_slots"] - counters["neuron_updates"]
    return RunResult(mode, net.n, net.m, steps, counters, spikes, updates, events, spike_counts, final_v, wall, memory,
                     raster, voltage, phases, extra)


def _schedule_arrays(schedule: InputSchedule, steps: int):
    return np.ascontiguousarray(schedule.indptr[:steps + 1], dtype=np.int64), np.ascontiguousarray(schedule.neurons, dtype=np.int32)


def run_time_step(net: Network, schedule: InputSchedule, p: NeuronParams, steps: int, *, record_spikes: bool = False,
                  record_voltage_every: int = 0, profile: bool = False) -> RunResult:
    n, dt = net.n, np.dtype(p.dtype)
    d, ref = p.delay_steps, p.ref_steps
    size = max(n, 1)
    v = np.zeros(n, dtype=dt)
    last_spike = np.full(n, SENTINEL, dtype=np.int32)
    acc = np.zeros((d, size), dtype=np.float64)  # float64 per-target sums, cast once when consumed
    ext_add = np.zeros(size, dtype=dt)
    spk = np.zeros(size, dtype=np.int32)
    spikes = np.zeros(steps, dtype=np.int32)
    events = np.zeros(steps, dtype=np.int64)
    spike_counts = np.zeros(n, dtype=np.int32)
    rec_ptr = np.zeros(steps + 1 if record_spikes else 1, dtype=np.int64)
    rec_ids = np.zeros(max(1024, 2 * size) if record_spikes else 1, dtype=np.int32)
    rec_used = np.zeros(1, dtype=np.int64)
    every = int(record_voltage_every)
    volt_buf = np.zeros(((steps - 1) // every + 1 if every and steps else 0, size), dtype=dt)
    timers = np.zeros(len(TS_PHASES), dtype=np.int64)
    sched_indptr, sched_neurons = _schedule_arrays(schedule, steps)
    args = (net.indptr, net.indices, net.weights, sched_indptr, sched_neurons, d, ref, dt.type(p.alpha), dt.type(p.v_threshold),
            dt.type(p.v_reset), dt.type(schedule.weight))
    t, t0 = 0, time.perf_counter()
    while t < steps:
        t = _ts_kernel(t, steps, *args, v, last_spike, acc, ext_add, spk, spikes, events, spike_counts,
                       record_spikes, rec_ptr, rec_ids, rec_used, every, volt_buf, profile, timers)
        if t < steps:
            rec_ids = _grow(rec_ids, int(rec_used[0]) + size)
    wall = time.perf_counter() - t0
    memory = {"neuron_state": int(v.nbytes + last_spike.nbytes), "event_buffers": int(acc.nbytes),
              "work_buffers": int(ext_add.nbytes + spk.nbytes), "input_schedule": schedule.nbytes,
              "instrumentation": int(spikes.nbytes + events.nbytes + spike_counts.nbytes), **net.memory()}
    raster = (rec_ptr, rec_ids[:int(rec_used[0])].copy()) if record_spikes else None
    phases = {k: float(v_) / 1e9 for k, v_ in zip(TS_PHASES, timers)} if profile else None
    updates = np.full(steps, n, dtype=np.int64)
    external = int(sched_indptr[steps] - sched_indptr[0]) if steps else 0
    return _result("time_step", net, steps, spikes, updates, events, spike_counts, v.copy(), wall, memory, external, raster,
                   _voltage_list(volt_buf, every, steps), phases, {"backend": "numba", "variant": "dense_pass"})


def run_event_driven(net: Network, schedule: InputSchedule, p: NeuronParams, steps: int, *, record_spikes: bool = False,
                     record_voltage_every: int = 0, profile: bool = False, variant: str = "touched") -> RunResult:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    n, dt = net.n, np.dtype(p.dtype)
    d, ref = p.delay_steps, p.ref_steps
    size = max(n, 1)
    decay = _decay_table(p, steps)
    v = np.zeros(n, dtype=dt)
    last_spike = np.full(n, SENTINEL, dtype=np.int32)
    t_last = np.zeros(n, dtype=np.int32)
    acc = np.zeros(size, dtype=np.float64)
    flags = np.zeros(size, dtype=np.uint8)
    touched = np.zeros(size, dtype=np.int32)
    code = VARIANTS.index(variant)
    ring_ids = np.zeros((d, size) if code != 2 else (d, 1), dtype=np.int32)
    ring_count = np.zeros(d, dtype=np.int64)
    ev_cap = max(net.m, 1) if code == 2 else 1
    ev_tgt = np.zeros((d, ev_cap), dtype=np.int32)
    ev_w = np.zeros((d, ev_cap), dtype=net.weights.dtype)
    ev_count = np.zeros(d, dtype=np.int64)
    spk = np.zeros(size, dtype=np.int32)
    sort_tmp = np.zeros(size, dtype=np.int32)
    sort_counts = np.zeros(513, dtype=np.int64)
    spikes = np.zeros(steps, dtype=np.int32)
    updates = np.zeros(steps, dtype=np.int64)
    events = np.zeros(steps, dtype=np.int64)
    unique_targets = np.zeros(steps, dtype=np.int64)
    spike_counts = np.zeros(n, dtype=np.int32)
    pending = np.zeros(2, dtype=np.int64)  # current, peak
    rec_ptr = np.zeros(steps + 1 if record_spikes else 1, dtype=np.int64)
    rec_ids = np.zeros(max(1024, 2 * size) if record_spikes else 1, dtype=np.int32)
    rec_used = np.zeros(1, dtype=np.int64)
    every = int(record_voltage_every)
    volt_buf = np.zeros(((steps - 1) // every + 1 if every and steps else 0, size), dtype=dt)
    timers = np.zeros(len(ED_PHASES), dtype=np.int64)
    sched_indptr, sched_neurons = _schedule_arrays(schedule, steps)
    t, t0 = 0, time.perf_counter()
    while t < steps:
        t = _ed_kernel(t, steps, code, net.indptr, net.indices, net.weights, sched_indptr, sched_neurons, d, ref,
                       dt.type(p.v_threshold), dt.type(p.v_reset), dt.type(schedule.weight), dt.type(0.0), decay, v, last_spike,
                       t_last, acc,
                       flags, touched, ring_ids, ring_count, ev_tgt, ev_w, ev_count, spk, sort_tmp, sort_counts,
                       _radix_passes(size), spikes, updates, events, unique_targets, spike_counts, pending,
                       record_spikes, rec_ptr, rec_ids, rec_used, every, volt_buf, profile, timers)
        if t < steps:
            rec_ids = _grow(rec_ids, int(rec_used[0]) + size)
    wall = time.perf_counter() - t0
    final_v = _lazy(v, t_last, last_spike, steps - 1, p, decay) if steps else v
    final_v = np.where(t_last == steps - 1, v, final_v).astype(dt)
    event_buffers = acc.nbytes + flags.nbytes + touched.nbytes + ring_ids.nbytes + ring_count.nbytes + ev_tgt.nbytes + ev_w.nbytes
    memory = {"neuron_state": int(v.nbytes + last_spike.nbytes + t_last.nbytes), "decay_table": int(decay.nbytes),
              "event_buffers": int(event_buffers), "work_buffers": int(spk.nbytes + sort_tmp.nbytes + sort_counts.nbytes),
              "input_schedule": schedule.nbytes,
              "instrumentation": int(spikes.nbytes + events.nbytes + updates.nbytes + unique_targets.nbytes + spike_counts.nbytes),
              **net.memory()}
    raster = (rec_ptr, rec_ids[:int(rec_used[0])].copy()) if record_spikes else None
    phases = {k: float(v_) / 1e9 for k, v_ in zip(ED_PHASES, timers)} if profile else None
    external = int(sched_indptr[steps] - sched_indptr[0]) if steps else 0
    extra = {"backend": "numba", "variant": variant, "unique_targets": int(unique_targets.sum()),
             "unique_targets_max_per_step": int(unique_targets.max()) if steps else 0,
             "peak_pending_events": int(pending[1]), "_unique_targets_per_step": unique_targets}
    return _result("event_driven", net, steps, spikes, updates, events, spike_counts, final_v, wall, memory, external, raster,
                   _voltage_list(volt_buf, every, steps), phases, extra)


def run(net: Network, schedule: InputSchedule, p: NeuronParams, steps: int, mode: str, *, variant: str = "touched",
        aggregation: str | None = None, **kw) -> RunResult:
    if mode == "time_step":
        return run_time_step(net, schedule, p, steps, **kw)
    return run_event_driven(net, schedule, p, steps, variant=variant, **kw)


def warmup(dtype: str = "float32", modes=("time_step", "event_driven"), variants=("touched",)) -> float:
    """Compile (or load from the on-disk cache) the kernels for this dtype on a tiny network; returns seconds."""
    t0 = time.perf_counter()
    net = Network.from_edges(4, [0, 1, 2, 3], [1, 2, 3, 0], [1.5, 1.5, 1.5, -0.5], dtype=dtype)
    sched = InputSchedule(np.array([0, 1, 1, 1, 1], dtype=np.int64), np.array([0], dtype=np.int32), 2.0, {})
    p = NeuronParams(dtype=dtype)
    for mode in modes:
        for variant in (variants if mode == "event_driven" else ("touched",)):
            for profile in (False, True):
                run(net, sched, p, 4, mode, variant=variant, profile=profile, record_spikes=True, record_voltage_every=2)
                run(net, sched, p, 4, mode, variant=variant, profile=profile)
    return time.perf_counter() - t0
