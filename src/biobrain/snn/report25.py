"""Milestone 2.5 summary and figures — built only from raw files in results/milestone25 and the frozen M2 baseline.

Two speedups are never mixed:
  compiled_speedup        = NumPy wall / compiled wall, same mode (implementation effect)
  event_driven_advantage  = time-step wall / event-driven wall, same backend (execution-strategy effect)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from .. import runinfo
from . import experiments, m25

MIB = 2 ** 20
LABELS = tuple(m25.BACKENDS)
SIZE_ORDER = ("expand_1k", "expand_10k", "expand_50k", "full")
REALISTIC_RATES = (0.001, 0.01)  # MILESTONE25_PLAN §5: "realistic activity" for the A/B/C classification


def _dir():
    return m25.results_dir()


def _load(name: str) -> list[dict]:
    path = _dir() / "benchmarks" / f"{name}.jsonl"
    return experiments.load_jsonl(path) if path.is_file() else []


def _json(relative: str) -> dict | None:
    path = _dir() / relative
    return json.loads(path.read_text()) if path.is_file() else None


def _records() -> list[dict]:
    return _load("matrix") + _load("full_smoke")


def _rate(r: dict) -> float:
    return r["config"]["inputs"]["rate"]


def _per_step_us(r: dict) -> float:
    return 1e6 * r["metrics"]["wall_s"] / r["metrics"]["steps"]


def _stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def _significant(s: dict) -> bool:
    """MILESTONE25_PLAN §5: a win/loss only if |median - 1| >= 5 % and the seed range does not include 1."""
    return abs(s["median"] - 1) >= 0.05 and not (s["min"] <= 1 <= s["max"])


def _grouped(records) -> dict:
    """(subgraph, rate) -> label -> {seed: record}"""
    out = defaultdict(lambda: defaultdict(dict))
    for r in records:
        out[(r["subgraph"]["name"], _rate(r))][r["backend_label"]][r["seed"]] = r
    return out


def _ratio(cell: dict, num: str, den: str) -> dict | None:
    seeds = sorted(set(cell.get(num, {})) & set(cell.get(den, {})))
    if not seeds:
        return None
    s = _stats([cell[num][k]["metrics"]["wall_s"] / cell[den][k]["metrics"]["wall_s"] for k in seeds])
    s["significant"] = _significant(s)
    return s


def table(records) -> list[dict]:
    rows = []
    for (name, rate), cell in sorted(_grouped(records).items(), key=lambda kv: (SIZE_ORDER.index(kv[0][0]), kv[0][1])):
        any_rec = next(iter(next(iter(cell.values())).values()))
        row = {"subgraph": name, "neurons": any_rec["metrics"]["neurons"], "edges": any_rec["metrics"]["edges"], "input_rate": rate,
               "steps": any_rec["metrics"]["steps"], "seeds": sorted({s for per in cell.values() for s in per})}
        for label, per in cell.items():
            recs = list(per.values())
            m = [x["metrics"] for x in recs]
            row[label] = {
                "us_per_step": _stats([_per_step_us(x) for x in recs]),
                "wall_s_per_sim_s": _stats([x["wall_s"] / x["sim_time_s"] for x in m]),
                "peak_rss_mib": _stats([x["process"]["peak_rss"] / MIB for x in recs]),
                "spikes": _stats([x["spikes"] for x in m]), "synaptic_events": _stats([x["synaptic_events"] for x in m]),
                "neuron_updates": _stats([x["neuron_updates"] for x in m]),
                "skipped_update_fraction": _stats([x["skipped_update_fraction"] for x in m]),
                "million_events_per_wall_s": _stats([x["synaptic_events"] / x["wall_s"] / 1e6 for x in m]),
                "ns_per_event_wall": _stats([1e9 * x["wall_s"] / x["synaptic_events"] for x in m]) if min(x["synaptic_events"] for x in m) else None,
                "unique_targets_per_step": _stats([x["engine_extra"]["unique_targets"] / x["metrics"]["steps"] for x in recs])
                if label == "compiled_event_driven" else None,
                "power_sources": sorted({str(x["process"].get("power_source")) for x in recs}),
                "thermal_warnings": sorted({w for x in recs for w in x["process"].get("thermal_warnings") or []}),
                "compile_or_cache_load_s": _stats([x["process"]["compile_or_cache_load_s"] for x in recs])
                if label.startswith("compiled") else None,
            }
        ts = next(iter(cell.get("numpy_time_step", cell.get("compiled_time_step")).values()))["metrics"]
        row["spikes_per_neuron_per_s"] = ts["population_rate_hz"]
        row["events_per_neuron_per_step"] = ts["synaptic_events"] / (ts["neurons"] * ts["steps"])
        row["fan_out_per_spike"] = ts["synaptic_events_per_spike"]
        row["identical_spike_counts_across_backends"] = all(
            len({per[s]["metrics"]["spikes"] for per in cell.values() if s in per}) == 1 for s in row["seeds"])
        row["compiled_speedup"] = {"time_step": _ratio(cell, "numpy_time_step", "compiled_time_step"),
                                   "event_driven": _ratio(cell, "numpy_event_driven", "compiled_event_driven")}
        row["event_driven_advantage"] = {"compiled": _ratio(cell, "compiled_time_step", "compiled_event_driven"),
                                         "numpy": _ratio(cell, "numpy_time_step", "numpy_event_driven")}
        rows.append(row)
    return rows


def classify(rows) -> dict:
    cells = {}
    for r in rows:
        if r["subgraph"] in m25.BENCH_SIZES and r["input_rate"] in REALISTIC_RATES and r["event_driven_advantage"]["compiled"]:
            s = r["event_driven_advantage"]["compiled"]
            case = "A" if s["median"] >= 1.25 and s["min"] > 1.05 else "C" if s["median"] <= 0.8 and s["max"] < 0.95 else "B"
            cells[f"{r['subgraph']} @ {r['input_rate']:g}"] = {"case": case, **s}
    counts = {c: sum(v["case"] == c for v in cells.values()) for c in "ABC"}
    majority = [c for c, k in counts.items() if k > len(cells) / 2]
    return {"rule": "MILESTONE25_PLAN §5", "cells": cells, "counts": counts, "case": majority[0] if majority else None}


def crossover(rows, name: str, key: str = "compiled") -> dict:
    pts = sorted((r["input_rate"], r["events_per_neuron_per_step"], r["event_driven_advantage"][key]["median"],
                  r["compiled_event_driven" if key == "compiled" else "numpy_event_driven"]["skipped_update_fraction"]["median"])
                 for r in rows if r["subgraph"] == name and r["event_driven_advantage"][key])
    if not pts:
        return {"status": "no data"}
    above = [p for p in pts if p[2] > 1]
    if not above:
        return {"status": "event-driven slower at every measured input", "best": max(p[2] for p in pts)}
    if len(above) == len(pts):
        return {"status": "event-driven faster at every measured input", "worst": min(p[2] for p in pts)}
    crossings = [(a, b) for a, b in zip(pts, pts[1:]) if (a[2] > 1) != (b[2] > 1)]
    (r0, e0, s0, k0), (r1, e1, s1, k1) = crossings[0]
    f = math.log(s0) / (math.log(s0) - math.log(s1))
    geo = lambda a, b: float(math.exp(math.log(max(a, 1e-12)) + f * (math.log(max(b, 1e-12)) - math.log(max(a, 1e-12)))))
    return {"status": "crossover" if len(crossings) == 1 else "multiple crossings (first reported)", "input_rate": geo(r0, r1),
            "events_per_neuron_per_step": geo(e0, e1), "skipped_update_fraction": float(k0 + f * (k1 - k0)),
            "between_inputs": [r0, r1], "faster_below": bool(s0 > 1)}


def m2_reproduction(records) -> dict:
    """NumPy runs of M2.5 against the frozen M2 runs with the same config hash (same machine, other day time)."""
    base = {(r["config_hash"], r["mode"], r["aggregation"]): r for r in (_json("baseline_m2.json") or {"benchmark_rows": []})["benchmark_rows"]}
    pairs = []
    for r in records:
        if r["backend"] != "numpy":
            continue
        b = base.get((r["config_hash"], r["mode"], r["config"]["run"]["aggregation"] if r["mode"] == "event_driven" else None))
        if b:
            pairs.append({"subgraph": r["subgraph"]["name"], "input_rate": _rate(r), "mode": r["mode"], "seed": r["seed"],
                          "m25_over_m2_wall": r["metrics"]["wall_s"] / b["wall_s"],
                          "spikes_identical": r["metrics"]["spikes"] == b["spikes"],
                          "synaptic_events_identical": r["metrics"]["synaptic_events"] == b["synaptic_events"]})
    ratios = [p["m25_over_m2_wall"] for p in pairs]
    return {"pairs": len(pairs), "wall_ratio": _stats(ratios) if ratios else None,
            "all_spikes_identical": all(p["spikes_identical"] and p["synaptic_events_identical"] for p in pairs), "rows": pairs}


GROUPS = {  # MILESTONE25_PLAN §4
    "neuron_update": {"numpy_time_step": ("leak", "synaptic_input", "external_input", "refractory", "threshold"),
                      "numpy_event_driven": ("leak", "input", "refractory_threshold"),
                      "compiled_time_step": ("neuron_pass", "external_mark"), "compiled_event_driven": ("neuron_update", "external_mark")},
    "event_delivery_and_management": {"numpy_time_step": ("reset_and_gather", "deliver"), "numpy_event_driven": ("aggregate", "reset_and_schedule"),
                                      "compiled_time_step": ("delivery",), "compiled_event_driven": ("delivery", "spike_sort", "schedule")},
    "bookkeeping": {"numpy_time_step": ("record",), "numpy_event_driven": ("record",), "compiled_time_step": ("record",),
                    "compiled_event_driven": ("record",)},
}


def phase_table() -> list[dict]:
    before, after = _json("profile/profile_numpy.json"), _json("profile/profile_compiled.json")
    if not before or not after:
        return []
    numpy_rows = {(r["subgraph"], r["input_rate"], r["mode"]): r for r in before["rows"]
                  if r["mode"] == "time_step" or r["aggregation"] == "auto"}
    rows = []
    for c in after["rows"]:
        n = numpy_rows.get((c["subgraph"], c["input_rate"], c["mode"]))
        if not n:
            continue
        label = "time_step" if c["mode"] == "time_step" else "event_driven"
        t_np, t_c = sum(n["phases_s"].values()), sum(c["phases_s"].values())
        for group, members in GROUPS.items():
            s_np = sum(n["phases_s"].get(k, 0.0) for k in members[f"numpy_{label}"]) / n["steps"]
            s_c = sum(c["phases_s"].get(k, 0.0) for k in members[f"compiled_{label}"]) / c["steps"]
            rows.append({"subgraph": c["subgraph"], "input_rate": c["input_rate"], "mode": c["mode"], "phase": group,
                         "numpy_us_per_step": 1e6 * s_np, "compiled_us_per_step": 1e6 * s_c,
                         "speedup": s_np / s_c if s_c > 0 else None,
                         "share_before": s_np * n["steps"] / t_np if t_np else None,
                         "share_after": s_c * c["steps"] / t_c if t_c else None})
    return rows


def throughput(rows) -> dict:
    out = {}
    for label in LABELS:
        vals = [(r["subgraph"], r["input_rate"], r[label]["million_events_per_wall_s"]["median"], r[label]["ns_per_event_wall"]["median"])
                for r in rows if label in r and r[label]["ns_per_event_wall"] and r["events_per_neuron_per_step"] >= 0.1]
        if vals:
            out[label] = {"million_events_per_wall_s": _stats([v[2] for v in vals]), "ns_per_event_wall": _stats([v[3] for v in vals]),
                          "note": "wall time / synaptic events over runs with >= 0.1 arriving events per neuron per step; includes neuron updates"}
    prof = _json("profile/profile_compiled.json")
    if prof:
        for mode in ("time_step", "event_driven"):
            ns = [1e9 * r["phases_s"]["delivery"] / r["synaptic_events"] for r in prof["rows"] if r["mode"] == mode and r["synaptic_events"] > 1000]
            if ns:
                out[f"compiled_{mode}_delivery_phase"] = {"ns_per_event": _stats(ns),
                                                          "note": "delivery phase timer / events (compiled profile runs, 500 steps)"}
    return out


def full_connectome(rows, records) -> dict | None:
    cal = _json(f"{m25.FULL_DIR}/calibration.json")
    if not cal:
        return None
    out = {"build": _json(f"{m25.FULL_DIR}/build.json"), "estimate_before_run": _json(f"{m25.FULL_DIR}/estimate_before_run.json"),
           "calibration": {k: cal[k] for k in ("baseline_gain", "widest_stable_range", "stable_gains", "gains", "calibration_wall_s",
                                              "numpy_spot_check")},
           "calibration_regimes": {f"{g:.4g}": sorted({r["regime"] for r in cal["rows"] if r["gain"] == g}) for g in cal["gains"]}}
    regime = _json(f"{m25.FULL_DIR}/baseline_regime.json")
    if regime:
        out["baseline_regime"] = [{k: r[k] for k in ("seed", "regime", "self_sustained", "amplification", "population_rate_hz",
                                                     "rate_on_per_step", "rate_late_off_per_step")} for r in regime["rows"]]
    probe = _json(f"{m25.FULL_DIR}/probe.json")
    if probe:
        out["probe"] = [{k: r[k] for k in ("input_rate", "seed", "population_rate_hz_on", "population_rate_hz_last_third_off",
                                           "first_silent_step_after_input_off")}
                        | {"regime": r["classification"]["regime"], "self_sustained": r["classification"]["self_sustained"],
                           **{k: r["summary"][k] for k in ("fraction_neurons_active", "silent_neurons", "refractory_occupancy",
                                                           "synaptic_events_per_spike", "population_rate_hz")},
                           "unique_targets_per_step_mean": float(np.mean(r["unique_targets_per_step"]))}
                        for r in probe["rows"]]
    full_rows = [r for r in rows if r["subgraph"] == m25.FULL]
    out["smoke"] = full_rows
    # memory: measured worker peak RSS vs the M2 ESTIMATE (NumPy time-step worker at 1 % spikes) and the M2.5 estimate
    m2 = (_json("baseline_m2.json") or {}).get("m2_full_connectome_estimate", {})
    empty = (_json("experiments/empty_process_rss.json") or {}).get("median_bytes", {})
    mem = []
    for r in records:
        if r["subgraph"]["name"] != m25.FULL:
            continue
        acc = r["memory"]
        accounted = sum(v for k, v in acc.items() if isinstance(v, int))
        base = empty.get(r["backend"], 0)
        mem.append({"backend_label": r["backend_label"], "input_rate": _rate(r), "seed": r["seed"],
                    "spike_fraction_per_step": r["metrics"]["spike_fraction_per_step"], "peak_rss": r["process"]["peak_rss"],
                    "base_process": base, "topology": acc["topology"], "weights": acc["synapse_state"],
                    "neuron_state": acc["neuron_state"], "event_buffers": acc["event_buffers"] + acc.get("work_buffers", 0),
                    "input_schedule": acc["input_schedule"], "instrumentation": acc["instrumentation"],
                    "decay_table": acc.get("decay_table", 0), "accounted_arrays": accounted,
                    "temporary_and_allocator": r["process"]["peak_rss"] - base - accounted})
    out["memory_runs"] = mem
    if m2 and mem:
        ts = sorted((x for x in mem if x["backend_label"] == "numpy_time_step"), key=lambda x: abs(math.log10(max(x["spike_fraction_per_step"], 1e-9)) + 2))
        if ts:
            est = m2["ram"]["time_step_total_mib_at_1pct_spikes"] * MIB
            out["m2_ram_estimate_vs_measured"] = {"estimated_bytes": est, "measured_bytes": ts[0]["peak_rss"],
                                                  "measured_at_spike_fraction": ts[0]["spike_fraction_per_step"],
                                                  "error_percent": 100 * (est - ts[0]["peak_rss"]) / ts[0]["peak_rss"],
                                                  "what": "NumPy time-step worker, run closest to 1 % spikes per step"}
        pred = []
        for r in records:
            if r["subgraph"]["name"] != m25.FULL or r["backend"] != "numpy":
                continue
            mode = "time_step" if r["mode"] == "time_step" else "event_driven_auto"
            model = m2["cost_models"].get(mode)
            if not model:
                continue
            m = r["metrics"]
            c = model["seconds"]
            per = m["neurons"] if mode == "time_step" else m["neuron_updates"] / m["steps"]
            predicted = c[0] + c[1] * per + c[2] * m["synaptic_events"] / m["steps"] + c[3] * m["spikes"] / m["steps"]
            measured = m["wall_s"] / m["steps"]
            pred.append({"mode": mode, "input_rate": _rate(r), "seed": r["seed"], "predicted_s_per_step": predicted,
                         "measured_s_per_step": measured, "error_percent": 100 * (predicted - measured) / measured})
        if pred:
            out["m2_runtime_model_vs_measured"] = {"rows": pred, "abs_error_percent": _stats([abs(p["error_percent"]) for p in pred]),
                                                   "what": "M2 cost model evaluated at the measured events, spikes and updates of each full-brain NumPy run"}
    return out


def ideal_event_driven_bound(rows) -> dict:
    """Upper bound on the time-step / event-driven ratio for ANY event-driven implementation on this CPU: it must still
    deliver the same synaptic events, so it cannot take less than the time-step delivery phase.
    profile-based: 1 / (delivery share of compiled time-step), from the compiled profile runs;
    model-based: (c_N·N + c_E·E) / (c_E·E) with the compiled time-step cost model (base term 0)."""
    out = {"profile_based": [], "model_based": []}
    prof = _json("profile/profile_compiled.json")
    if prof:
        for r in prof["rows"]:
            if r["mode"] == "time_step" and r["phases_s"]["delivery"] > 0:
                share = r["phases_s"]["delivery"] / sum(r["phases_s"].values())
                out["profile_based"].append({"subgraph": r["subgraph"], "input_rate": r["input_rate"], "delivery_share": share,
                                             "bound": 1 / share})
    model = m25.cost_models([r for r in _records() if r["subgraph"]["name"] != m25.FULL]).get("compiled_time_step")
    if model:
        _, c_n, c_e = model["seconds"]
        for r in rows:
            e = r["events_per_neuron_per_step"] * r["neurons"]
            if e > 0:
                measured = r["event_driven_advantage"]["compiled"]
                out["model_based"].append({"subgraph": r["subgraph"], "input_rate": r["input_rate"],
                                           "events_per_neuron_per_step": r["events_per_neuron_per_step"],
                                           "bound": (c_n * r["neurons"] + c_e * e) / (c_e * e),
                                           "measured_compiled_advantage": measured["median"] if measured else None})
    return out


def summary() -> str:
    records = _records()
    rows = table(records)
    names = [n for n in SIZE_ORDER if any(r["subgraph"] == n for r in rows)]
    out = {
        "m2_raw_results_unchanged": m25.m2_unchanged(),
        "environment": _json("environment.json"),
        "classification": classify(rows),
        "crossover": {n: {"compiled": crossover(rows, n, "compiled"), "numpy": crossover(rows, n, "numpy")} for n in names},
        "throughput": throughput(rows),
        "phases_before_after": phase_table(),
        "m2_reproduction": m2_reproduction(records),
        "cost_models": m25.cost_models([r for r in records if r["subgraph"]["name"] != m25.FULL]),
        "cost_models_with_full": m25.cost_models(records),
        "ideal_event_driven_bound": ideal_event_driven_bound(rows),
        "equivalence": _equivalence_summary(),
        "full_connectome": full_connectome(rows, records),
        "empty_worker_peak_rss": _json("experiments/empty_process_rss.json"),
        "rows": rows,
        "run": runinfo.collect(),
    }
    return str(experiments._write(_dir() / "summary.json", out))


def _equivalence_summary() -> dict:
    eq, long_eq = _json("experiments/equivalence.json"), _json("experiments/long_equivalence.json")
    out = {}
    if eq:
        same = [(r["subgraph"], r["input_rate"], r["dtype"], mode, v["identical"]) for r in eq["rows"] for mode, v in r["same_mode_numpy_vs_compiled"].items()]
        cross = [(r["subgraph"], r["input_rate"], r["dtype"], v) for r in eq["rows"] for v in [r["cross_mode"]["compiled_time_step_vs_compiled_event_driven"]]]
        out["same_mode_bit_identical"] = {"comparisons": len(same), "identical": sum(s[4] for s in same),
                                          "not_identical": [s[:4] for s in same if not s[4]]}
        out["voltage_records_identical"] = all(r["voltage_records_identical"] for r in eq["rows"])
        out["compiled_ts_vs_ed"] = {"comparisons": len(cross), "spikes_equal": sum(c[3]["spikes_equal"] for c in cross),
                                    "within_m2_absolute_tolerance": sum(c[3]["within_m2_absolute_tolerance"] for c in cross),
                                    "within_m25_relative_tolerance": sum(c[3]["within_m25_relative_tolerance"] for c in cross),
                                    "max_abs_final_v_difference": {dt: max((c[3]["max_abs_final_v_difference"] for c in cross if c[2] == dt), default=None)
                                                                   for dt in ("float32", "float64")},
                                    "exceptions": [{"subgraph": c[0], "input_rate": c[1], "dtype": c[2], "spikes_equal": c[3]["spikes_equal"],
                                                    "max_abs_final_v_difference": c[3]["max_abs_final_v_difference"]}
                                                   for c in cross if not c[3]["within_m25_relative_tolerance"]]}
    if long_eq:
        out["long_run"] = long_eq["rows"]
    ts, ed = _json("experiments/check_compiled_time_step.json"), _json("experiments/check_compiled_event_driven.json")
    if ts:
        out["check_ts_identical"] = [sum(r["identical"] for r in ts["rows"]), len(ts["rows"])]
    if ed:
        flags = [x["identical"] for r in ed["rows"] if "identical_to_numpy_event_driven" in r for x in r["identical_to_numpy_event_driven"].values()]
        out["check_ed_identical"] = [sum(flags), len(flags)]
    return out


# ---- figures ------------------------------------------------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


STYLE = {"numpy_time_step": ("#6baed6", "--", "o"), "numpy_event_driven": ("#fdae6b", "--", "s"),
         "compiled_time_step": ("#08519c", "-", "o"), "compiled_event_driven": ("#d94801", "-", "s")}


def _plain_log_axes(ax) -> None:
    from matplotlib.ticker import NullFormatter

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def figures() -> list[str]:
    plt = _plt()
    fig_dir = _dir() / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = table(_records())
    names = [n for n in SIZE_ORDER if any(r["subgraph"] == n for r in rows)]
    colors = dict(zip(SIZE_ORDER, ("tab:green", "tab:purple", "tab:red", "black")))
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    def series(name, key_fn):
        pts = sorted((r["input_rate"], *key_fn(r)) for r in rows if r["subgraph"] == name and key_fn(r) is not None)
        return list(zip(*pts)) if pts else None

    # 1. activity vs event-driven advantage (compiled solid, NumPy dashed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, xkey in ((axes[0], "input_rate"), (axes[1], "events_per_neuron_per_step")):
        for name in names:
            for key, ls in (("compiled", "-"), ("numpy", ":")):
                pts = sorted((r[xkey], r["event_driven_advantage"][key]["median"], r["event_driven_advantage"][key]["min"],
                              r["event_driven_advantage"][key]["max"]) for r in rows
                             if r["subgraph"] == name and r["event_driven_advantage"][key])
                if pts:
                    x, y, lo, hi = map(np.array, zip(*pts))
                    ax.errorbar(x, y, yerr=[y - lo, hi - y], ls=ls, marker="o", ms=3, capsize=2, color=colors[name],
                                label=f"{name}, {key}")
        ax.axhline(1, color="k", lw=0.8)
        ax.set(xscale="log", yscale="log", ylabel="time-step wall / event-driven wall (same backend)",
               xlabel="external input probability per neuron per step" if xkey == "input_rate" else "synaptic events arriving per neuron per step")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Event-driven advantage: compiled (solid) vs NumPy (dotted); median over seeds, bars = min–max")
    save(fig, "1_event_driven_advantage_vs_activity.png")

    # 2. size vs runtime; 3. size vs throughput; 4. size vs RAM
    for fname, metric, ylabel in (("2_runtime_vs_size.png", "us_per_step", "µs per simulated step (median)"),
                                  ("3_synaptic_throughput_vs_size.png", "million_events_per_wall_s", "million synaptic events per wall second"),
                                  ("4_peak_rss_vs_size.png", "peak_rss_mib", "peak RSS of the worker (MiB)")):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
        for ax, rate in zip(axes, (0.0001, 0.01, 0.1)):
            for label in LABELS:
                pts = sorted((r["neurons"], r[label][metric]["median"]) for r in rows if r["input_rate"] == rate and label in r)
                if pts:
                    color, ls, marker = STYLE[label]
                    ax.plot(*zip(*pts), ls=ls, marker=marker, color=color, ms=4, label=label)
            if fname.startswith("4"):
                empty = (_json("experiments/empty_process_rss.json") or {}).get("median_bytes", {})
                for backend, ls in (("numpy", ":"), ("numba", "-.")):
                    if backend in empty:
                        ax.axhline(empty[backend] / MIB, color="gray", ls=ls, lw=1, label=f"empty {backend} worker")
            ax.set(xscale="log", yscale="linear" if fname.startswith("4") else "log", xlabel="neurons", title=f"input {rate:g}")
            _plain_log_axes(ax)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=6)
        save(fig, fname)

    # 5. NumPy vs compiled speedup
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name in names:
        for mode, ls in (("time_step", "-"), ("event_driven", "--")):
            pts = sorted((r["input_rate"], r["compiled_speedup"][mode]["median"]) for r in rows if r["subgraph"] == name and r["compiled_speedup"][mode])
            if pts:
                ax.plot(*zip(*pts), ls=ls, marker="o", ms=3, color=colors[name], label=f"{name}, {mode}")
    ax.axhline(1, color="k", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="external input probability per neuron per step",
           ylabel="NumPy wall / compiled wall (same mode)", title="Compilation speedup (implementation effect only)")
    ax.legend(fontsize=6, ncol=2)
    save(fig, "5_numpy_vs_compiled_speedup.png")

    # 6. time per synaptic event vs activity
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.2), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        for label in LABELS:
            pts = sorted((r["events_per_neuron_per_step"], r[label]["ns_per_event_wall"]["median"]) for r in rows
                         if r["subgraph"] == name and label in r and r[label]["ns_per_event_wall"])
            if pts:
                color, ls, marker = STYLE[label]
                ax.plot(*zip(*pts), ls=ls, marker=marker, color=color, ms=4, label=label)
        ax.set(xscale="log", yscale="log", xlabel="events per neuron per step", title=name)
        _plain_log_axes(ax)
    axes[0][0].set_ylabel("wall ns per synaptic event (all step costs included)")
    axes[0][0].legend(fontsize=6)
    save(fig, "6_time_per_synaptic_event_vs_activity.png")

    # 7. full connectome activity over time
    probe = _json(f"{m25.FULL_DIR}/probe.json")
    if probe:
        fig, ax = plt.subplots(figsize=(9, 4.4))
        by_rate = defaultdict(list)
        for r in probe["rows"]:
            by_rate[r["input_rate"]].append(np.asarray(r["spikes_per_step"], dtype=float))
        n = (_json(f"{m25.FULL_DIR}/build.json") or {}).get("isolated_build", {}).get("neurons", 139_255)
        for rate, arrs in sorted(by_rate.items()):
            a = np.stack(arrs) / n * 1000  # Hz per neuron
            t = np.arange(a.shape[1])
            line, = ax.plot(t, np.median(a, axis=0), lw=1, label=f"input {rate:g} (median of {len(arrs)} seeds)")
            ax.fill_between(t, a.min(axis=0), a.max(axis=0), color=line.get_color(), alpha=0.2, lw=0)
        on = probe["rows"][0]["on_steps"]
        ax.axvline(on, color="k", ls="--", lw=1, label="external input switched off")
        ax.set(yscale="symlog", xlabel="simulated time (ms)", ylabel="population rate (Hz per neuron)",
               title=f"Full connectome ({n:,} neurons), gain {probe['gain']:.3g}: activity with input on, then off")
        ax.legend(fontsize=7)
        save(fig, "7_full_connectome_activity_over_time.png")

    # 8. phase shares before/after (supplementary)
    phases = phase_table()
    if phases:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        for ax, mode in zip(axes, ("time_step", "event_driven")):
            sel = [p for p in phases if p["mode"] == mode]
            keys = sorted({(p["subgraph"], p["input_rate"]) for p in sel}, key=lambda k: (SIZE_ORDER.index(k[0]), k[1]))
            x = np.arange(len(keys))
            for i, (tag, share_key) in enumerate((("NumPy", "share_before"), ("compiled", "share_after"))):
                bottom = np.zeros(len(keys))
                for group, color in zip(GROUPS, ("tab:green", "tab:red", "tab:gray")):
                    vals = np.array([next(p[share_key] for p in sel if (p["subgraph"], p["input_rate"]) == k and p["phase"] == group) or 0
                                     for k in keys])
                    ax.bar(x + (i - 0.5) * 0.4, vals, 0.38, bottom=bottom, color=color, alpha=1.0 if i else 0.45,
                           label=f"{group} ({tag})" if True else None)
                    bottom += vals
            ax.set_xticks(x, [f"{k[0].replace('expand_', '')}\n{k[1]:g}" for k in keys], fontsize=6)
            ax.set(title=f"{mode}: left bar NumPy, right bar compiled", ylabel="share of timed step")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.0))
        save(fig, "8_phase_shares_before_after.png")
    return made
