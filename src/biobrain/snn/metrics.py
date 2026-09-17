"""Derived metrics, activity-regime classification and mode-equivalence comparison.

Regime rules are definitions we chose (ASSUMED), written down before any calibration run:
- SATURATED: firing during input >= 50 % of the refractory-limited maximum rate, or firing in the last third of
  the no-input window >= 10 % of that maximum (runaway).
- DEAD: spikes during input < 1.05 x external input events (recurrent propagation adds < 5 %) and no spikes in
  the last third of the no-input window.
- STABLE: everything else; `self_sustained` flags activity that persists without input.
"""

from __future__ import annotations

import numpy as np

from .config import NeuronParams
from .engine import RunResult

RULES = {"dead_amplification": 1.05, "saturated_fraction_of_max_rate": 0.5, "runaway_fraction_of_max_rate": 0.1}


def summarize(res: RunResult, p: NeuronParams) -> dict:
    c = res.counters
    n, steps = res.n, res.steps
    sim_s = steps * p.dt_ms / 1000
    slots = max(n * steps, 1)
    tail = np.minimum(p.ref_steps, steps - 1 - np.arange(steps))
    active = res.spike_counts > 0
    wall = max(res.wall_s, 1e-12)
    out = {
        "neurons": n, "edges": res.m, "steps": steps, "sim_time_s": sim_s, "wall_s": res.wall_s,
        "realtime_factor": sim_s / wall, **c,
        "update_fraction": c["neuron_updates"] / slots, "skipped_update_fraction": c["skipped_neuron_updates"] / slots,
        "population_rate_hz": c["spikes"] / max(n * sim_s, 1e-12),
        "spike_fraction_per_step": c["spikes"] / slots,
        "fraction_neurons_active": float(active.mean()) if n else 0.0, "silent_neurons": int((~active).sum()),
        "refractory_occupancy": float(np.sum(res.spikes_per_step * tail) / slots),
        "synaptic_events_per_spike": c["synaptic_events"] / max(c["spikes"], 1),
        "max_spike_fraction_in_a_step": float(res.spikes_per_step.max() / max(n, 1)) if steps else 0.0,
        "neuron_updates_per_wall_s": c["neuron_updates"] / wall,
        "synaptic_events_per_wall_s": c["synaptic_events"] / wall,
        "spikes_per_wall_s": c["spikes"] / wall,
    }
    if res.voltage:
        out["membrane_mean_max"] = float(max(s["mean"] for s in res.voltage))
        out["membrane_max"] = float(max(s["max"] for s in res.voltage))
    return out


def classify(res: RunResult, p: NeuronParams, on_steps: int, external_on: int) -> dict:
    n = max(res.n, 1)
    max_rate = 1.0 / (p.ref_steps + 1)
    on = res.spikes_per_step[:on_steps]
    off = res.spikes_per_step[on_steps:]
    late = off[len(off) - max(len(off) // 3, 1):] if len(off) else off
    rate_on = float(on.sum() / (n * max(on_steps, 1)))
    rate_late = float(late.sum() / (n * max(len(late), 1))) if len(late) else 0.0
    amplification = float(on.sum() / max(external_on, 1))
    if rate_on >= RULES["saturated_fraction_of_max_rate"] * max_rate or rate_late >= RULES["runaway_fraction_of_max_rate"] * max_rate:
        regime = "SATURATED"
    elif amplification < RULES["dead_amplification"] and rate_late == 0:
        regime = "DEAD"
    else:
        regime = "STABLE"
    return {"regime": regime, "rate_on_per_step": rate_on, "rate_late_off_per_step": rate_late,
            "amplification": amplification, "self_sustained": rate_late > 0, "max_rate_per_step": max_rate,
            "rules": RULES}


def compare(a: RunResult, b: RunResult, v_tolerance: float) -> dict:
    """Equivalence of two runs of the same model and input: identical spikes, potentials within tolerance."""
    same_steps = np.array_equal(a.spikes_per_step, b.spikes_per_step)
    first = int(np.flatnonzero(a.spikes_per_step != b.spikes_per_step)[0]) if not same_steps else None
    raster_equal = None
    if a.raster is not None and b.raster is not None:
        raster_equal = bool(np.array_equal(a.raster[0], b.raster[0]) and np.array_equal(a.raster[1], b.raster[1]))
    dv = float(np.max(np.abs(a.final_v.astype(np.float64) - b.final_v.astype(np.float64)))) if a.n else 0.0
    spikes_equal = bool(same_steps and np.array_equal(a.spike_counts, b.spike_counts)
                        and np.array_equal(a.events_per_step, b.events_per_step))
    return {"spikes_equal": spikes_equal, "raster_equal": raster_equal, "first_divergent_step": first,
            "spike_count_difference": int(a.counters["spikes"] - b.counters["spikes"]),
            "max_abs_final_v_difference": dv, "v_tolerance": v_tolerance,
            "equivalent": bool(spikes_equal and (raster_equal is not False) and dv <= v_tolerance)}


def first_divergent_step(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> int | None:
    """First step whose spike set differs between two rasters (per-step indptr, neuron ids ascending within a step)."""
    (pa, na), (pb, nb) = a, b
    count_diff = np.flatnonzero(np.diff(pa) != np.diff(pb))
    limit = int(count_diff[0]) if count_diff.size else pa.size - 1
    end = int(pa[limit])  # steps before `limit` have equal counts, so positions up to `end` are aligned
    idx = np.flatnonzero(na[:end] != nb[:end])
    if idx.size:
        return int(np.searchsorted(pa, idx[0], side="right") - 1)
    return limit if count_diff.size else None

V_TOLERANCE = {"float64": 1e-9, "float32": 1e-4}
