"""Milestone 2 figures, full-connectome ESTIMATE and summary — computed only from raw result files."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import runinfo
from . import experiments

FULL_NEURONS, FULL_EDGES = 139_255, 15_091_983
MIB = 2 ** 20


def _exp(name: str) -> Path:
    return experiments.results_dir() / "experiments" / name


def _records(names=("main", "expand_50k")) -> list[dict]:
    out = []
    for name in names:
        path = experiments.results_dir() / "benchmarks" / f"{name}.jsonl"
        if path.is_file():
            out += experiments.load_jsonl(path)
    return out


def _mode(r: dict) -> str:
    return "time_step" if r["mode"] == "time_step" else f"event_driven_{r['aggregation']}"


def _paired(records: list[dict]) -> dict:
    """(subgraph, rate) -> {mode: [records sorted by seed]}"""
    table = defaultdict(lambda: defaultdict(list))
    for r in records:
        table[(r["subgraph"]["name"], r["config"]["inputs"]["rate"])][_mode(r)].append(r)
    for modes in table.values():
        for runs in modes.values():
            runs.sort(key=lambda r: r["seed"])
    return table


def speedups(records: list[dict]) -> list[dict]:
    rows = []
    for (name, rate), modes in sorted(_paired(records).items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ts = modes.get("time_step", [])
        for mode in ("event_driven_sparse", "event_driven_auto"):
            ed = modes.get(mode, [])
            pairs = [(a, b) for a in ts for b in ed if a["seed"] == b["seed"]]
            if not pairs:
                continue
            s = np.array([a["metrics"]["wall_s"] / b["metrics"]["wall_s"] for a, b in pairs])
            m = [b["metrics"] for _, b in pairs]
            tm = [a["metrics"] for a, _ in pairs]
            rows.append({
                "subgraph": name, "neurons": m[0]["neurons"], "edges": m[0]["edges"], "input_rate": rate, "mode": mode,
                "seeds": len(pairs), "speedup_median": float(np.median(s)), "speedup_min": float(s.min()), "speedup_max": float(s.max()),
                "time_step_wall_s": float(np.median([x["wall_s"] for x in tm])), "event_driven_wall_s": float(np.median([x["wall_s"] for x in m])),
                "spike_fraction_per_step": float(np.median([x["spike_fraction_per_step"] for x in m])),
                "events_per_neuron_per_step": float(np.median([x["synaptic_events"] / (x["neurons"] * x["steps"]) for x in m])),
                "update_fraction": float(np.median([x["update_fraction"] for x in m])),
                "skipped_update_fraction": float(np.median([x["skipped_update_fraction"] for x in m])),
                "spikes_equal_to_time_step": all(a["metrics"]["spikes"] == b["metrics"]["spikes"] for a, b in pairs),
            })
    return rows


def crossover(rows: list[dict], name: str, mode: str) -> dict:
    """Input rate and events/neuron/step where the median speedup crosses 1 (log-interpolated)."""
    pts = sorted((r["input_rate"], r["events_per_neuron_per_step"], r["speedup_median"]) for r in rows
                 if r["subgraph"] == name and r["mode"] == mode)
    if not pts:
        return {"status": "no data"}
    above = [p for p in pts if p[2] > 1]
    if not above:
        return {"status": "event-driven slower at every measured rate", "best_speedup": max(p[2] for p in pts),
                "at_rate": max(pts, key=lambda p: p[2])[0]}
    if len(above) == len(pts):
        return {"status": "event-driven faster at every measured rate", "min_speedup": min(p[2] for p in pts)}
    for (r0, e0, s0), (r1, e1, s1) in zip(pts, pts[1:]):
        if s0 > 1 >= s1 or s0 <= 1 < s1:
            f = (np.log(s0) - 0) / (np.log(s0) - np.log(s1))
            return {"status": "crossover", "input_rate": float(np.exp(np.log(r0) + f * (np.log(r1) - np.log(r0)))),
                    "events_per_neuron_per_step": float(np.exp(np.log(max(e0, 1e-12)) + f * (np.log(max(e1, 1e-12)) - np.log(max(e0, 1e-12))))),
                    "between_rates": [r0, r1], "faster_below": s0 > 1}
    return {"status": "non-monotonic", "points": pts}


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
    names = sorted({r["subgraph"] for r in rows}, key=lambda s: [x["neurons"] for x in rows if x["subgraph"] == s][0])

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
            for mode, style in (("event_driven_sparse", "o-"), ("event_driven_auto", "s--")):
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
    empty = json.loads(_exp("empty_process_rss.json").read_text())["peak_rss"] if _exp("empty_process_rss.json").is_file() else 0
    by = defaultdict(list)
    for r in records:
        by[(r["subgraph"]["name"], r["config"]["inputs"]["rate"], _mode(r))].append(r)
    for idx, (metric, label, fname) in enumerate((("ram", "peak RSS above an empty worker (MiB)", "2_ram_vs_neurons.png"),
                                                   ("wall", "wall seconds per simulated second", "3_wall_vs_neurons.png"))):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for rate in (0.0001, 0.01, 0.5):
            for mode, style in (("time_step", "-"), ("event_driven_sparse", "--"), ("event_driven_auto", ":")):
                pts = []
                for name in names:
                    runs = by.get((name, rate, mode))
                    if not runs:
                        continue
                    n = runs[0]["metrics"]["neurons"]
                    if metric == "ram":
                        pts.append((n, np.median([(x["process"]["peak_rss"] - empty) / MIB for x in runs])))
                    else:
                        pts.append((n, np.median([x["metrics"]["wall_s"] / x["metrics"]["sim_time_s"] for x in runs])))
                if pts:
                    x, y = zip(*sorted(pts))
                    ax.plot(x, y, style, marker="o", ms=4, label=f"input {rate:g}, {mode.replace('event_driven_', 'ED ')}")
        if metric == "ram":
            topo = sorted({(r["metrics"]["neurons"], (r["memory"]["topology"] + r["memory"]["synapse_state"]) / MIB) for r in records})
            if topo:
                ax.plot(*zip(*topo), "k-", lw=2, alpha=0.4, label="topology + weights (accounted)")
        ax.set(xscale="log", yscale="log", xlabel="neurons in subgraph", ylabel=label)
        ax.legend(fontsize=6)
        save(fig, fname)

    # 4. activity vs synaptic operations
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name in names:
        pts = sorted({(r["config"]["inputs"]["rate"], r["metrics"]["synaptic_events"] / r["metrics"]["sim_time_s"] / r["metrics"]["neurons"])
                      for r in records if r["subgraph"]["name"] == name and r["mode"] == "time_step" and r["seed"] == 1})
        if pts:
            ax.plot(*zip(*pts), "o-", ms=4, label=name)
    ax.set(xscale="log", yscale="log", xlabel="external input probability per neuron per step",
           ylabel="synaptic operations per neuron per simulated second")
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
        line, = ax.plot(g, np.maximum(rate, 1e-3), "-", lw=1, label=cal["subgraph"])
        for k, r in zip(g, rate):
            labels = {x["regime"] for x in by_gain[k]}
            ax.scatter([k], [max(r, 1e-3)], s=18, color=colors[labels.pop()] if len(labels) == 1 else "gray", zorder=3)
    ax.set(xscale="log", yscale="log", xlabel="gain (mean |weight| of a connectome edge, threshold units)",
           ylabel="population rate (Hz), 1 % input for 300 ms then off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items()]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + handles, fontsize=6, ncol=2)
    save(fig, "5_stability_calibration.png")

    # 6. time-step vs event-driven comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    pairs = [(a["metrics"]["spikes"], b["metrics"]["spikes"]) for (name, rate), modes in _paired(records).items()
             for a in modes.get("time_step", []) for b in modes.get("event_driven_sparse", []) if a["seed"] == b["seed"]]
    if pairs:
        x, y = map(np.array, zip(*pairs))
        axes[0].loglog(np.maximum(x, 1), np.maximum(y, 1), "o", ms=3)
        axes[0].plot([1, x.max()], [1, x.max()], "k-", lw=0.8)
        axes[0].set(xlabel="spikes, time-step", ylabel="spikes, event-driven",
                    title=f"{len(pairs)} paired benchmark runs; identical: {int(np.sum(x == y))}")
    if _exp("equivalence.json").is_file():
        eq = json.loads(_exp("equivalence.json").read_text())["rows"]
        for dtype, marker in (("float64", "o"), ("float32", "s")):
            pts = [(r["spikes"], max(r["max_abs_final_v_difference"], 1e-18)) for r in eq if r["dtype"] == dtype]
            if pts:
                axes[1].loglog(*zip(*pts), marker, ls="", ms=4, label=f"{dtype} (tolerance {1e-9 if dtype == 'float64' else 1e-4:g})")
        axes[1].set(xlabel="spikes in the compared run", ylabel="max |v_time-step − v_event-driven| at the end",
                    title="membrane-potential agreement")
        axes[1].legend(fontsize=7)
    save(fig, "6_time_step_vs_event_driven.png")

    if _exp("profile.json").is_file():
        prof = json.loads(_exp("profile.json").read_text())["rows"]
        fig, ax = plt.subplots(figsize=(11, 4.8))
        phases = sorted({p for r in prof for p in r["phases_s"]})
        labels = [f"{r['subgraph'].replace('expand_', '')}\n{r['input_rate']:g}\n{r['mode'][:4]}{'/' + r['aggregation'][0] if r['mode'] != 'time_step' else ''}" for r in prof]
        bottom = np.zeros(len(prof))
        for phase in phases:
            vals = np.array([r["phases_s"].get(phase, 0.0) / (r["wall_s"]) * r["wall_s"] * 1e3 / 500 for r in prof])
            ax.bar(range(len(prof)), vals, bottom=bottom, label=phase)
            bottom += vals
        ax.set_xticks(range(len(prof)), labels, fontsize=5)
        ax.set(yscale="log", ylabel="ms per step (stacked phases)", title="Where the time goes (profile)")
        ax.legend(fontsize=6, ncol=3)
        save(fig, "7_profile_phases.png")
    return made


def estimate() -> Path:
    """Least-squares cost model from measured runs, extrapolated to the full connectome. Output is labelled ESTIMATE."""
    from scipy.optimize import nnls

    records = _records()
    empty = json.loads(_exp("empty_process_rss.json").read_text())["peak_rss"]
    models = {}
    for mode in ("time_step", "event_driven_sparse", "event_driven_auto"):
        runs = [r for r in records if _mode(r) == mode]
        if len(runs) < 4:
            continue
        m = [r["metrics"] for r in runs]
        steps = np.array([x["steps"] for x in m], dtype=float)
        y = np.array([x["wall_s"] for x in m]) / steps
        n = np.array([x["neurons"] for x in m], dtype=float)
        ev = np.array([x["synaptic_events"] for x in m]) / steps
        up = np.array([x["neuron_updates"] for x in m]) / steps
        sp = np.array([x["spikes"] for x in m]) / steps
        X = np.stack([np.ones_like(n), n if mode == "time_step" else up, ev, sp], axis=1)
        coef, _ = nnls(X / y[:, None], np.ones_like(y))  # relative least squares
        pred = X @ coef
        models[mode] = {"terms": ["per step", "per neuron (time-step) / per updated neuron (event-driven)", "per synaptic event", "per spike"],
                        "seconds": coef.tolist(), "median_abs_relative_error": float(np.median(np.abs(pred - y) / y)), "runs": len(runs)}
    # event-driven update fraction as a function of arriving events per neuron, from measurements
    ed = [r["metrics"] for r in records if _mode(r) == "event_driven_sparse"]
    load = np.array([x["synaptic_events"] / (x["neurons"] * x["steps"]) for x in ed])
    upd = np.array([x["update_fraction"] for x in ed])
    zero_sign = np.median([r["network"]["removed_zero_sign"] / max(r["network"]["subgraph_edges"], 1) for r in records])
    full_edges = FULL_EDGES * (1 - zero_sign)
    out_degree = full_edges / FULL_NEURONS
    scenarios = []
    for spike_fraction in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
        events = spike_fraction * FULL_NEURONS * out_degree
        load_full = events / FULL_NEURONS
        update_fraction = float(np.interp(np.log10(max(load_full, 1e-9)), np.log10(np.maximum(load, 1e-9))[np.argsort(load)],
                                          upd[np.argsort(load)])) if load.size else None
        row = {"spike_fraction_per_step": spike_fraction, "synaptic_events_per_step": events, "events_per_neuron_per_step": load_full,
               "update_fraction_event_driven (interpolated)": update_fraction}
        for mode, model in models.items():
            c = model["seconds"]
            per = (FULL_NEURONS if mode == "time_step" else update_fraction * FULL_NEURONS) if update_fraction is not None else FULL_NEURONS
            row[f"{mode}_seconds_per_simulated_second"] = 1000 * (c[0] + c[1] * per + c[2] * events + c[3] * spike_fraction * FULL_NEURONS)
        scenarios.append(row)
    bytes_net = (FULL_NEURONS + 1) * 8 + full_edges * 4 + full_edges * 4
    ram = {"topology_and_float32_weights_mib": bytes_net / MIB,
           "time_step_state_and_ring_mib": FULL_NEURONS * (8 + 4) / MIB,
           "event_driven_state_mib": FULL_NEURONS * 12 / MIB,
           "empty_worker_process_mib": empty / MIB,
           "per_step_temporaries_mib_at_1pct_spikes": (1e-2 * FULL_NEURONS * out_degree * 24 + FULL_NEURONS * 8) / MIB,
           "note": "temporaries: gather index (8 B) + targets (4 B) + weights (4 B) + aggregation (~8 B) per event, plus one float64 N-vector"}
    ram["total_estimate_mib_at_1pct_spikes"] = sum(v for k, v in ram.items() if k.endswith("_mib") and "event_driven" not in k) + ram["per_step_temporaries_mib_at_1pct_spikes"]
    result = {"label": "ESTIMATE — extrapolated from measured subgraph runs, not measured on the full connectome",
              "full_connectome": {"neurons": FULL_NEURONS, "edges": FULL_EDGES, "simulated_edges_after_zero_sign_removal": full_edges,
                                  "zero_sign_fraction_used": float(zero_sign), "mean_out_degree": out_degree},
              "cost_models": models, "scenarios": scenarios, "ram": ram,
              "caveats": ["subgraphs are denser than the whole brain (expand_10k mean out-degree 177 vs 108 overall)",
                          "activity of the full brain under the same gain is unknown; spike fractions are scenarios, not predictions",
                          "the linear cost model ignores cache effects that grow with array size"],
              "run": runinfo.collect()}
    return experiments._write(experiments.results_dir() / "estimate_full_connectome.json", result)


def summary() -> Path:
    records = _records()
    rows = speedups(records)
    empty = json.loads(_exp("empty_process_rss.json").read_text())["peak_rss"] if _exp("empty_process_rss.json").is_file() else 0
    per_size = {}
    for name in sorted({r["subgraph"]["name"] for r in records}):
        runs = [r for r in records if r["subgraph"]["name"] == name]
        m0 = runs[0]["metrics"]
        entry = {"neurons": m0["neurons"], "edges": m0["edges"]}
        for mode in ("time_step", "event_driven_sparse", "event_driven_auto"):
            sel = [r for r in runs if _mode(r) == mode]
            if not sel:
                continue
            at = lambda rate: [r for r in sel if r["config"]["inputs"]["rate"] == rate]
            entry[mode] = {
                "peak_rss_above_empty_mib_at_1pct": float(np.median([(r["process"]["peak_rss"] - empty) / MIB for r in at(0.01)])) if at(0.01) else None,
                "peak_rss_above_empty_mib_max": float(max((r["process"]["peak_rss"] - empty) / MIB for r in sel)),
                "wall_s_per_sim_s_at_1pct": float(np.median([r["metrics"]["wall_s"] / r["metrics"]["sim_time_s"] for r in at(0.01)])) if at(0.01) else None,
                "skipped_update_fraction_range": [float(min(r["metrics"]["skipped_update_fraction"] for r in sel)),
                                                  float(max(r["metrics"]["skipped_update_fraction"] for r in sel))],
            }
        entry["spikes_per_sim_s_at_1pct"] = float(np.median([r["metrics"]["spikes"] / r["metrics"]["sim_time_s"] for r in runs if r["config"]["inputs"]["rate"] == 0.01] or [np.nan]))
        entry["synaptic_ops_per_sim_s_at_1pct"] = float(np.median([r["metrics"]["synaptic_events"] / r["metrics"]["sim_time_s"] for r in runs if r["config"]["inputs"]["rate"] == 0.01] or [np.nan]))
        entry["crossover"] = {mode: crossover(rows, name, mode) for mode in ("event_driven_sparse", "event_driven_auto")}
        entry["speedup_range"] = {mode: [min((r["speedup_median"] for r in rows if r["subgraph"] == name and r["mode"] == mode), default=None),
                                         max((r["speedup_median"] for r in rows if r["subgraph"] == name and r["mode"] == mode), default=None)]
                                  for mode in ("event_driven_sparse", "event_driven_auto")}
        cal = _exp(f"calibration_{name}.json")
        if cal.is_file():
            c = json.loads(cal.read_text())
            entry["calibration"] = {"baseline_gain": c["baseline_gain"], "widest_stable_range": c["widest_stable_range"]}
        per_size[name] = entry
    eq = json.loads(_exp("equivalence.json").read_text())["rows"] if _exp("equivalence.json").is_file() else []
    out = {"per_subgraph": per_size, "speedup_rows": rows,
           "equivalence": {"comparisons": len(eq), "equivalent": sum(r["equivalent"] for r in eq),
                           "benchmark_pairs_with_identical_spikes": sum(r["spikes_equal_to_time_step"] for r in rows), "benchmark_pairs": len(rows)},
           "empty_worker_peak_rss_mib": empty / MIB, "run": runinfo.collect()}
    return experiments._write(experiments.results_dir() / "summary.json", out)
