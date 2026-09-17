"""Milestone 2 figures, full-connectome ESTIMATE and summary — computed only from raw result files (and the store)."""

from __future__ import annotations

import inspect
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import runinfo
from . import energy, experiments

MIB = 2 ** 20
PATTERNS = ("burst", "pulse", "sparse_pattern")
ED_MODES = ("event_driven_sparse", "event_driven_auto")


def _exp(name: str) -> Path:
    return experiments.results_dir() / "experiments" / name


def _json(name: str) -> dict | None:
    path = _exp(name)
    return json.loads(path.read_text()) if path.is_file() else None


def _jsonl(name: str) -> list[dict]:
    path = experiments.results_dir() / "benchmarks" / f"{name}.jsonl"
    return experiments.load_jsonl(path) if path.is_file() else []


def _empty_rss() -> int:
    data = _json("empty_process_rss.json")
    return data["peak_rss"] if data else 0


def _valid(r: dict, gains: dict[str, float]) -> bool:
    """A benchmark record counts only if it ran at its subgraph's baseline gain under the default model."""
    g = gains.get(r["subgraph"]["name"])
    return g is not None and math.isclose(r["network"]["gain"], g, rel_tol=1e-9)


def _records(names=("main", "expand_50k")) -> list[dict]:
    from .pipeline import gains

    g = gains()
    return [r for name in names for r in _jsonl(name) if _valid(r, g)]


def _mode(r: dict) -> str:
    return "time_step" if r["mode"] == "time_step" else f"event_driven_{r['aggregation']}"


def _rate(r: dict) -> float:
    return r["config"]["inputs"]["rate"]


def _paired(records: list[dict]) -> dict:
    """(subgraph, rate) -> {mode: [records sorted by seed]}"""
    table = defaultdict(lambda: defaultdict(list))
    for r in records:
        table[(r["subgraph"]["name"], _rate(r))][_mode(r)].append(r)
    for modes in table.values():
        for runs in modes.values():
            runs.sort(key=lambda r: r["seed"])
    return table


def speedups(records: list[dict]) -> list[dict]:
    rows = []
    for (name, rate), modes in sorted(_paired(records).items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ts = modes.get("time_step", [])
        for mode in ED_MODES:
            pairs = [(a, b) for a in ts for b in modes.get(mode, []) if a["seed"] == b["seed"]]
            if not pairs:
                continue
            s = np.array([a["metrics"]["wall_s"] / b["metrics"]["wall_s"] for a, b in pairs])
            m = [b["metrics"] for _, b in pairs]
            rows.append({
                "subgraph": name, "neurons": m[0]["neurons"], "edges": m[0]["edges"], "steps": m[0]["steps"], "input_rate": rate,
                "mode": mode, "seeds": len(pairs),
                "speedup_median": float(np.median(s)), "speedup_min": float(s.min()), "speedup_max": float(s.max()),
                "time_step_wall_s": float(np.median([a["metrics"]["wall_s"] for a, _ in pairs])),
                "event_driven_wall_s": float(np.median([x["wall_s"] for x in m])),
                "spike_fraction_per_step": float(np.median([x["spike_fraction_per_step"] for x in m])),
                "events_per_neuron_per_step": float(np.median([x["synaptic_events"] / (x["neurons"] * x["steps"]) for x in m])),
                "update_fraction": float(np.median([x["update_fraction"] for x in m])),
                "skipped_update_fraction": float(np.median([x["skipped_update_fraction"] for x in m])),
                "seeds_with_identical_spike_count": sum(a["metrics"]["spikes"] == b["metrics"]["spikes"] for a, b in pairs),
            })
    return rows


def crossover(rows: list[dict], name: str, mode: str) -> dict:
    """Input rate and events/neuron/step where the median speedup crosses 1 (log-interpolated)."""
    pts = sorted((r["input_rate"], r["events_per_neuron_per_step"], r["speedup_median"]) for r in rows
                 if r["subgraph"] == name and r["mode"] == mode and r["input_rate"] > 0)
    if not pts:
        return {"status": "no data"}
    above = [p for p in pts if p[2] > 1]
    if not above:
        best = max(pts, key=lambda p: p[2])
        return {"status": "event-driven slower at every measured rate", "best_speedup": best[2], "at_rate": best[0]}
    if len(above) == len(pts):
        return {"status": "event-driven faster at every measured rate", "min_speedup": min(p[2] for p in pts)}
    crossings = [(a, b) for a, b in zip(pts, pts[1:]) if (a[2] > 1) != (b[2] > 1)]
    (r0, e0, s0), (r1, e1, s1) = crossings[0]
    f = np.log(s0) / (np.log(s0) - np.log(s1))
    lerp = lambda x0, x1: float(np.exp(np.log(max(x0, 1e-12)) + f * (np.log(max(x1, 1e-12)) - np.log(max(x0, 1e-12)))))
    return {"status": "crossover" if len(crossings) == 1 else "multiple crossings (first reported)", "crossings": len(crossings),
            "input_rate": lerp(r0, r1), "events_per_neuron_per_step": lerp(e0, e1), "between_rates": [r0, r1], "faster_below": bool(s0 > 1)}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figures() -> list[str]:
    plt = _plt()
    fig_dir = experiments.results_dir() / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    records = _records()
    rows = speedups(records)
    size = {r["subgraph"]: r["neurons"] for r in rows}
    names = sorted(size, key=size.get)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=130)
        plt.close(fig)
        made.append(name)

    # 1. activity vs event-driven speedup
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, xkey, xlabel in ((axes[0], "input_rate", "external input probability per neuron per step"),
                             (axes[1], "events_per_neuron_per_step", "synaptic events arriving per neuron per step (measured)")):
        for name in names:
            for mode, style in zip(ED_MODES, ("o-", "s--")):
                pts = sorted((r[xkey], r["speedup_median"], r["speedup_min"], r["speedup_max"]) for r in rows
                             if r["subgraph"] == name and r["mode"] == mode)
                if pts:
                    x, y, lo, hi = map(np.array, zip(*pts))
                    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=style, ms=4, capsize=2,
                                label=f"{name}, {mode.replace('event_driven_', 'ED ')}")
        ax.axhline(1, color="k", lw=0.8)
        ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel="speedup = time-step wall / event-driven wall")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Event-driven speedup vs activity (median over seeds; bars = min–max)")
    save(fig, "1_speedup_vs_activity.png")

    # 2. RAM vs neurons, 3. wall vs neurons
    empty = _empty_rss()
    by = defaultdict(list)
    for r in records:
        by[(r["subgraph"]["name"], _rate(r), _mode(r))].append(r)
    for metric, label, fname in (("ram", "peak RSS above an empty worker (MiB)", "2_ram_vs_neurons.png"),
                                 ("wall", "wall seconds per simulated second", "3_wall_vs_neurons.png")):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for rate in (0.0001, 0.01, 0.5):
            for mode, style in zip(("time_step", *ED_MODES), ("-", "--", ":")):
                pts = []
                for name in names:
                    runs = by.get((name, rate, mode))
                    if runs:
                        value = [(x["process"]["peak_rss"] - empty) / MIB if metric == "ram" else x["metrics"]["wall_s"] / x["metrics"]["sim_time_s"]
                                 for x in runs]
                        pts.append((runs[0]["metrics"]["neurons"], np.median(value)))
                if pts:
                    ax.plot(*zip(*sorted(pts)), style, marker="o", ms=4, label=f"input {rate:g}, {mode.replace('event_driven_', 'ED ')}")
        if metric == "ram":
            topo = sorted({(r["metrics"]["neurons"], (r["memory"]["topology"] + r["memory"]["synapse_state"]) / MIB) for r in records})
            ax.plot(*zip(*topo), "k-", lw=2, alpha=0.4, label="topology + weights (accounted)")
        ax.set(xscale="log", yscale="log", xlabel="neurons in subgraph", ylabel=label)
        ax.legend(fontsize=6)
        save(fig, fname)

    # 4. activity vs synaptic operations
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name in names:
        sel = [r for r in records if r["subgraph"]["name"] == name and r["mode"] == "time_step"]
        per = defaultdict(list)
        for r in sel:
            per[_rate(r)].append(r["metrics"]["synaptic_events"] / r["metrics"]["sim_time_s"] / r["metrics"]["neurons"])
        ax.plot(sorted(per), [np.median(per[k]) for k in sorted(per)], "o-", ms=4, label=name)
    ax.set(xscale="log", yscale="log", xlabel="external input probability per neuron per step",
           ylabel="synaptic operations per neuron per simulated second\n(median over seeds, time-step runs)")
    ax.legend(fontsize=7)
    save(fig, "4_synops_vs_activity.png")

    # 5. stability (calibration phase diagram)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = {"DEAD": "tab:blue", "STABLE": "tab:green", "SATURATED": "tab:red"}
    for path in sorted((experiments.results_dir() / "experiments").glob("calibration_*.json")):
        cal = json.loads(path.read_text())
        by_gain = defaultdict(list)
        for row in cal["rows"]:
            by_gain[row["gain"]].append(row)
        g = sorted(by_gain)
        rate = [np.median([x["population_rate_hz"] for x in by_gain[k]]) for k in g]
        label = cal["subgraph"] + (f" ({cal['variant']})" if "variant" in cal else "")
        ax.plot(g, np.maximum(rate, 1e-3), "--" if "variant" in cal else "-", lw=1, label=label)
        for k, r in zip(g, rate):
            regimes = {x["regime"] for x in by_gain[k]}
            ax.scatter([k], [max(r, 1e-3)], s=18, color=colors[regimes.pop()] if len(regimes) == 1 else "gray", zorder=3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="gray", label="seeds disagree"))
    ax.set(xscale="log", yscale="log", xlabel="gain (mean |weight| of a connectome edge, threshold units)",
           ylabel="population rate while input on (Hz, median of 3 seeds)", title="Calibration: 1 % input for 300 ms, then 300 ms off")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + handles, fontsize=6, ncol=2)
    save(fig, "5_stability_calibration.png")

    # 6. time-step vs event-driven comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    pairs = [(a["metrics"]["spikes"], b["metrics"]["spikes"]) for modes in _paired(records).values()
             for a in modes.get("time_step", []) for b in modes.get("event_driven_sparse", []) if a["seed"] == b["seed"]]
    x, y = map(np.array, zip(*pairs))
    axes[0].loglog(np.maximum(x, 1), np.maximum(y, 1), "o", ms=3)
    axes[0].plot([1, x.max()], [1, x.max()], "k-", lw=0.8)
    axes[0].set(xlabel="spikes, time-step", ylabel="spikes, event-driven (sparse)",
                title=f"{len(pairs)} paired benchmark runs (float32); identical counts: {int(np.sum(x == y))}")
    eq = _json("equivalence.json")
    if eq:
        for dtype, marker in (("float64", "o"), ("float32", "s")):
            pts = [(max(r["spikes"], 1), max(r["max_abs_final_v_difference"], 1e-18)) for r in eq["rows"] if r["dtype"] == dtype]
            axes[1].loglog(*zip(*pts), marker, ls="", ms=4, label=f"{dtype} (tolerance {eq['tolerance'][dtype]:g})")
            axes[1].axhline(eq["tolerance"][dtype], color="k", lw=0.6, ls=":")
        axes[1].set(xlabel="spikes in the compared run", ylabel="max |v_time-step − v_event-driven| at the end",
                    title="equivalence runs: membrane-potential agreement")
        axes[1].legend(fontsize=7)
    save(fig, "6_time_step_vs_event_driven.png")

    # 7. profile
    prof = _json("profile.json")
    if prof:
        default_steps = inspect.signature(experiments.profile_phases).parameters["steps"].default
        rows_p = prof["rows"]
        fig, ax = plt.subplots(figsize=(11, 4.8))
        phases = sorted({p for r in rows_p for p in r["phases_s"]})
        labels = [f"{r['subgraph'].replace('expand_', '')}\n{r['input_rate']:g}\n"
                  f"{'TS' if r['mode'] == 'time_step' else 'ED/' + r['aggregation']}" for r in rows_p]
        bottom = np.zeros(len(rows_p))
        for phase in phases:
            vals = np.array([r["phases_s"].get(phase, 0.0) * 1e6 / r.get("steps", default_steps) for r in rows_p])
            ax.bar(range(len(rows_p)), vals, bottom=bottom, label=phase)
            bottom += vals
        ax.set_xticks(range(len(rows_p)), labels, fontsize=5)
        ax.set(ylabel="µs per step (stacked phase timers)", title="Where the time goes: subgraph / input rate / mode")
        ax.legend(fontsize=6, ncol=3)
        save(fig, "7_profile_phases.png")
    return made


def _full_connectome() -> dict:
    """Size of the whole store and the fraction of its edges that the default sign table sets to 0 (source = neuron)."""
    from ..connectome.store import Connectome
    from .config import SignSpec
    from .network import SIGN_KEYS

    conn = Connectome.load("flywire_fafb_v783")
    degree = np.diff(np.asarray(conn["out_indptr"]))
    spec = SignSpec()
    table = {k: spec.value(k) for k in SIGN_KEYS}
    zero = np.array([table.get(nt, spec.missing) == 0 for nt in conn.labels("top_nt", np.arange(degree.size))])
    return {"neurons": int(degree.size), "edges": int(degree.sum()), "zero_sign_edge_fraction": float(degree[zero].sum() / degree.sum())}


def estimate() -> Path:
    """Least-squares cost model from measured runs, extrapolated to the full connectome. Output is labelled ESTIMATE."""
    from scipy.optimize import nnls

    records = _records()
    full = _full_connectome()
    n_full = full["neurons"]
    edges_full = full["edges"] * (1 - full["zero_sign_edge_fraction"])
    out_degree = edges_full / n_full
    models = {}
    for mode in ("time_step", *ED_MODES):
        m = [r["metrics"] for r in records if _mode(r) == mode]
        steps = np.array([x["steps"] for x in m], dtype=float)
        y = np.array([x["wall_s"] for x in m]) / steps
        n = np.array([x["neurons"] for x in m], dtype=float)
        ev = np.array([x["synaptic_events"] for x in m]) / steps
        up = np.array([x["neuron_updates"] for x in m]) / steps
        sp = np.array([x["spikes"] for x in m]) / steps
        X = np.stack([np.ones_like(n), n if mode == "time_step" else up, ev, sp], axis=1)
        coef, _ = nnls(X / y[:, None], np.ones_like(y))  # relative least squares
        rel = np.abs(X @ coef - y) / y
        models[mode] = {"terms": ["per step", "per neuron (time-step) / per updated neuron (event-driven)", "per synaptic event", "per spike"],
                        "seconds": coef.tolist(), "median_abs_relative_error": float(np.median(rel)),
                        "max_abs_relative_error": float(rel.max()), "runs": len(m)}
    ed = [r["metrics"] for r in records if _mode(r) == "event_driven_sparse"]
    load = np.array([x["synaptic_events"] / (x["neurons"] * x["steps"]) for x in ed])
    upd = np.array([x["update_fraction"] for x in ed])
    order = np.argsort(load)
    scenarios = []
    for spike_fraction in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
        events = spike_fraction * n_full * out_degree
        load_full = events / n_full
        update_fraction = float(np.interp(np.log10(load_full), np.log10(np.maximum(load[order], 1e-12)), upd[order]))
        row = {"spike_fraction_per_step": spike_fraction, "synaptic_events_per_step": events, "events_per_neuron_per_step": load_full,
               "event_driven_update_fraction_interpolated": update_fraction}
        for mode, model in models.items():
            c = model["seconds"]
            per = n_full if mode == "time_step" else update_fraction * n_full
            row[f"{mode}_seconds_per_simulated_second"] = 1000 * (c[0] + c[1] * per + c[2] * events + c[3] * spike_fraction * n_full)
        scenarios.append(row)
    ram = {"topology_and_float32_weights_mib": ((n_full + 1) * 8 + edges_full * (4 + 4)) / MIB,
           "time_step_state_and_ring_mib": n_full * (4 + 4 + 4) / MIB,
           "event_driven_state_mib": n_full * 12 / MIB,
           "empty_worker_process_mib": _empty_rss() / MIB,
           "per_step_temporaries_mib_at_1pct_spikes": (1e-2 * n_full * out_degree * 24 + n_full * 8) / MIB,
           "note": "per event: gather index (8 B) + targets (4 B) + weights (4 B) + aggregation (~8 B); plus one float64 N-vector"}
    ram["time_step_total_mib_at_1pct_spikes"] = (ram["topology_and_float32_weights_mib"] + ram["time_step_state_and_ring_mib"]
                                                 + ram["empty_worker_process_mib"] + ram["per_step_temporaries_mib_at_1pct_spikes"])
    sub = {r["subgraph"]["name"]: r["subgraph"]["edges"] / r["subgraph"]["neurons"] for r in records}
    result = {"label": "ESTIMATE — extrapolated from measured subgraph runs, not measured on the full connectome",
              "full_connectome": {**full, "simulated_edges_after_zero_sign_removal": edges_full, "mean_out_degree_simulated": out_degree},
              "cost_models": models, "scenarios": scenarios, "ram": ram,
              "caveats": [f"subgraphs are denser than the whole brain: mean out-degree {', '.join(f'{k} {v:.0f}' for k, v in sorted(sub.items(), key=lambda kv: kv[1]))}"
                          f" vs {full['edges'] / full['neurons']:.0f} in the full store",
                          "activity of the full brain under the same gain is unknown; spike fractions are scenarios, not predictions",
                          "the linear cost model ignores cache effects that grow with array size",
                          "event-driven update fraction is interpolated from subgraph runs (arriving events per neuron per step)"],
              "run": runinfo.collect()}
    return experiments._write(experiments.results_dir() / "estimate_full_connectome.json", result)


def _per_rate(runs: list[dict], fn) -> dict[str, float]:
    by = defaultdict(list)
    for r in runs:
        by[_rate(r)].append(fn(r))
    return {f"{rate:g}": float(np.median(v)) for rate, v in sorted(by.items())}


def _stability(name: str) -> dict | None:
    c = _json(f"calibration_{name}.json")
    if c is None:
        return None
    measured = (_json("baseline_regime.json") or {"subgraphs": {}})["subgraphs"].get(name)
    at = measured["rows"] if measured else [row for row in c["rows"] if c["baseline_gain"] is not None
                                            and math.isclose(row["gain"], c["baseline_gain"], rel_tol=1e-9)]
    return {"baseline_gain": c["baseline_gain"], "widest_stable_range": c["widest_stable_range"],
            "gains_all_seeds_stable": len(c["stable_gains"]), "gains_tested": len(c["gains"]),
            "regime_source": "baseline_regime.json" if measured else "calibration grid point",
            "regimes_at_baseline": sorted({row["regime"] for row in at}), "seeds_at_baseline": len(at),
            "population_rate_hz_at_baseline": [row["population_rate_hz"] for row in at],
            "self_sustained_seeds_at_baseline": int(sum(row["self_sustained"] for row in at))}


def _subgraph_summary(records: list[dict], rows: list[dict], name: str) -> dict:
    runs = [r for r in records if r["subgraph"]["name"] == name]
    empty = _empty_rss()
    m0 = runs[0]["metrics"]
    entry = {"neurons": m0["neurons"], "edges": m0["edges"], "steps": m0["steps"], "seeds": sorted({r["seed"] for r in runs}),
             "stability": _stability(name)}
    ts = [r for r in runs if r["mode"] == "time_step"]
    entry["spikes_per_sim_s"] = _per_rate(ts, lambda r: r["metrics"]["spikes"] / r["metrics"]["sim_time_s"])
    entry["synaptic_ops_per_sim_s"] = _per_rate(ts, lambda r: r["metrics"]["synaptic_events"] / r["metrics"]["sim_time_s"])
    for mode in ("time_step", *ED_MODES):
        sel = [r for r in runs if _mode(r) == mode]
        if sel:
            entry[mode] = {
                "wall_s_per_sim_s": _per_rate(sel, lambda r: r["metrics"]["wall_s"] / r["metrics"]["sim_time_s"]),
                "peak_rss_above_empty_mib": _per_rate(sel, lambda r: (r["process"]["peak_rss"] - empty) / MIB),
                "skipped_update_fraction": _per_rate(sel, lambda r: r["metrics"]["skipped_update_fraction"]),
                "accounted_memory_mib": float(max(sum(r["memory"].values()) for r in sel) / MIB),
            }
    for mode in ED_MODES:
        mine = [r for r in rows if r["subgraph"] == name and r["mode"] == mode]
        if mine:
            best = max(mine, key=lambda r: r["speedup_median"])
            entry[mode]["speedup_median_by_rate"] = {f"{r['input_rate']:g}": r["speedup_median"] for r in mine}
            entry[mode]["best_speedup"] = {"median": best["speedup_median"], "min": best["speedup_min"], "max": best["speedup_max"],
                                           "input_rate": best["input_rate"]}
            entry[mode]["crossover"] = crossover(rows, name, mode)
    return entry


def _energy() -> dict | None:
    recs = _jsonl("energy_estimate")
    if not recs:
        return None
    runs = defaultdict(dict)
    for r in recs:
        runs[(_rate(r), r["experiment_id"].rsplit("-r", 1)[1])][r["mode"]] = r
    joules = lambda r: r["os_cpu_energy_estimate"]["energy_nj"] / 1e9
    rows = []
    for rate in sorted({k[0] for k in runs}):
        pairs = [v for (rt, _), v in runs.items() if rt == rate and len(v) == 2]
        ratio = [joules(p["event_driven"]) / joules(p["time_step"]) for p in pairs]
        wall = [p["event_driven"]["metrics"]["wall_s"] / p["time_step"]["metrics"]["wall_s"] for p in pairs]
        rows.append({"input_rate": rate, "steps": pairs[0]["time_step"]["metrics"]["steps"], "repeats": len(pairs),
                     "time_step_estimate_j": float(np.median([joules(p["time_step"]) for p in pairs])),
                     "event_driven_estimate_j": float(np.median([joules(p["event_driven"]) for p in pairs])),
                     "event_driven_over_time_step_estimate": [float(min(ratio)), float(np.median(ratio)), float(max(ratio))],
                     "event_driven_over_time_step_wall": [float(min(wall)), float(np.median(wall)), float(max(wall))],
                     "spike_counts_equal_in_all_repeats": all(p["time_step"]["metrics"]["spikes"] == p["event_driven"]["metrics"]["spikes"] for p in pairs)})
    per_cpu_s = [joules(r) / r["os_cpu_energy_estimate"]["cpu_time_s"] for r in recs]
    return {"label": energy.LABEL, "subgraph": recs[0]["subgraph"]["name"], "rows": rows,
            "estimate_joules_per_cpu_second": [float(min(per_cpu_s)), float(np.median(per_cpu_s)), float(max(per_cpu_s))],
            "all_windows_reliable": all(r["os_cpu_energy_estimate"]["reliable_window"] for r in recs),
            "power_sources": sorted({str(r["process"].get("power_source")) for r in recs}),
            "low_power_mode": sorted({str(r["process"].get("low_power_mode")) for r in recs})}


def _microbench() -> list[dict] | None:
    path = experiments.results_dir() / "benchmarks" / "aggregation_microbench.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())["rows"]
    return [{"neurons": n, "fastest_by_events": {str(r["events"]): r["fastest"] for r in rows if r["neurons"] == n}}
            for n in sorted({r["neurons"] for r in rows})]


def summary() -> Path:
    from .pipeline import gains

    records = _records()
    rows = speedups(records)
    names = sorted({r["subgraph"]["name"] for r in records}, key=lambda s: next(r["metrics"]["neurons"] for r in records if r["subgraph"]["name"] == s))
    neuropil_records = _records(("neuropils",))
    neuropil_rows = speedups(neuropil_records)
    excluded = sorted({r["subgraph"]["name"] for r in _jsonl("neuropils")} - {r["subgraph"]["name"] for r in neuropil_records})
    eq = _json("equivalence.json") or {"rows": []}
    long_eq = _json("long_equivalence.json")
    patterns = {}
    for p in PATTERNS:
        prow = speedups(_records((f"pattern_{p}",)))
        patterns[p] = [{k: r[k] for k in ("subgraph", "mode", "input_rate", "seeds", "speedup_median", "speedup_min", "speedup_max",
                                          "events_per_neuron_per_step", "skipped_update_fraction")} for r in prow]
    out = {
        "per_subgraph": {name: _subgraph_summary(records, rows, name) for name in names},
        "neuropils": {"included": {name: _subgraph_summary(neuropil_records, neuropil_rows, name)
                                   for name in sorted({r["subgraph"]["name"] for r in neuropil_records})},
                      "excluded": {name: "no gain is STABLE for all seeds under the default model; its rows ran at a variant's gain by mistake"
                                   for name in excluded if name not in gains()}},
        "patterns": patterns,
        "equivalence": {
            "comparisons": len(eq["rows"]), "equivalent": sum(r["equivalent"] for r in eq["rows"]),
            "identical_spikes": sum(r["spikes_equal"] for r in eq["rows"]),
            "not_equivalent": [{k: r[k] for k in ("subgraph", "dtype", "input_rate", "aggregation", "spikes_equal", "max_abs_final_v_difference", "v_tolerance")}
                               for r in eq["rows"] if not r["equivalent"]],
            "benchmark_seed_pairs": sum(r["seeds"] for r in rows),
            "benchmark_seed_pairs_with_identical_spike_count": sum(r["seeds_with_identical_spike_count"] for r in rows),
            "long_run": long_eq and {k: long_eq[k] for k in ("subgraph", "steps", "input_rate", "rows")},
        },
        "energy_estimate": _energy(),
        "aggregation_microbench": _microbench(),
        "empty_worker_peak_rss_mib": _empty_rss() / MIB,
        "speedup_rows": rows,
        "run": runinfo.collect(),
    }
    return experiments._write(experiments.results_dir() / "summary.json", out)
