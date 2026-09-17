"""`biobrain m2 <step>`: the Milestone 2 experiment pipeline, step by step. Every step writes raw results under
results/milestone2/ and can be re-run; figures and the summary are built only from those files."""

from __future__ import annotations

import json
import time

import numpy as np

from .. import paths
from ..connectome.store import Connectome
from . import experiments, subgraph
from .config import SimConfig

SIZES = ("expand_100", "expand_1k", "expand_10k")
NEUROPILS = ("neuropil_PB", "neuropil_SPS_R", "neuropil_LO_R")
# per-step input probability; the task's levels 0.1 %–50 % plus three sparser ones to locate a crossover
RATES = (0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50)
SEEDS = (1, 2, 3)
STEPS = 1000
BASE = SimConfig()


def _exp_dir():
    return experiments.results_dir() / "experiments"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gains() -> dict[str, float]:
    """Baseline gains calibrated under the DEFAULT model only (variant calibrations are not used)."""
    out = {}
    for path in sorted(_exp_dir().glob("calibration_*.json")):
        data = json.loads(path.read_text())
        if path.name == f"calibration_{data['subgraph']}.json" and "variant" not in data and data["baseline_gain"] is not None:
            out[data["subgraph"]] = data["baseline_gain"]
    return out


def step_subgraphs(conn: Connectome, names=(*SIZES, "expand_50k", *NEUROPILS)) -> None:
    for name in names:
        sub = subgraph.build(conn, name, subgraph.CANONICAL[name])
        subgraph.save(sub)
        m = sub.manifest
        _log(f"{name}: {m['neurons']:,} neurons, {m['edges']:,} edges, {m['synapses']:,} synapses, SCC {m['largest_strong_component']:,}")


def step_calibrate(conn: Connectome, names=(*SIZES, "expand_50k", *NEUROPILS)) -> None:
    for name in names:
        _log(f"calibrating {name}")
        result = experiments.calibrate(conn, name, BASE, log=_log)
        experiments._write(_exp_dir() / f"calibration_{name}.json", result)
        _log(f"{name}: stable gains {result['widest_stable_range']} -> baseline {result['baseline_gain']}")


def step_calibrate_pb_glutamate(conn: Connectome) -> None:
    """neuropil_PB is DEAD at every gain under glutamate = -1; does the sign assumption alone explain it?"""
    result = experiments.calibrate(conn, "neuropil_PB", BASE.replace(signs={"glutamate": 1.0}), log=_log)
    result["variant"] = "glutamate = +1"
    experiments._write(_exp_dir() / "calibration_variant_neuropil_PB_glutamate_plus.json", result)
    _log(f"neuropil_PB with glutamate=+1: stable {result['widest_stable_range']} -> baseline {result['baseline_gain']}")


def step_equivalence(conn: Connectome, names=(*SIZES, "expand_50k", "neuropil_LO_R")) -> None:
    g = gains()
    rows = []
    for name in names:
        if name not in g:
            _log(f"skip {name}: no calibrated gain")
            continue
        steps = 300 if name == "expand_50k" else 500
        rows += experiments.equivalence(conn, name, BASE.replace(weights={"gain": g[name]}), rates=(0.0001, 0.001, 0.01, 0.1),
                                        steps=steps)
        bad = [r for r in rows if r["subgraph"] == name and not r["equivalent"]]
        _log(f"{name}: {sum(r['subgraph'] == name for r in rows)} comparisons, not equivalent: {len(bad)}")
    experiments._write(_exp_dir() / "equivalence.json", {"rows": rows, "tolerance": {"float64": 1e-9, "float32": 1e-4}})


def step_long_equivalence(conn: Connectome, name: str = "expand_10k", steps: int = 40_000, rate: float = 0.001) -> None:
    """The energy experiment's longest run showed different spike counts between modes in float32. Where does the
    divergence start, and does it also happen in float64?"""
    from . import engine, metrics, network
    from .inputs import generate

    g = gains()
    sub = subgraph.load(conn, name)
    rows = []
    for dtype in ("float32", "float64"):
        cfg = BASE.replace(weights={"gain": g[name]}, neuron={"dtype": dtype}, inputs={"rate": rate}, run={"steps": steps, "seed": 1})
        net = network.build(conn, sub, cfg)
        sched = generate(cfg.inputs, net.n, steps, 1)
        ts = engine.run(net, sched, cfg.neuron, steps, "time_step")
        ed = engine.run(net, sched, cfg.neuron, steps, "event_driven", aggregation="auto")
        cmp = metrics.compare(ts, ed, metrics.V_TOLERANCE[dtype])
        gap = np.abs(np.cumsum(ts.spikes_per_step.astype(np.int64) - ed.spikes_per_step.astype(np.int64)))
        checkpoints = [c for c in (1_000, 2_000, 5_000, 10_000, 20_000, steps) if c <= steps]
        rows.append({"subgraph": name, "dtype": dtype, "steps": steps, "input_rate": rate, "gain": g[name], **cmp,
                     "spikes_time_step": ts.counters["spikes"], "spikes_event_driven": ed.counters["spikes"],
                     "relative_spike_count_difference": abs(ts.counters["spikes"] - ed.counters["spikes"]) / max(ts.counters["spikes"], 1),
                     "abs_cumulative_spike_count_difference": {str(c): int(gap[c - 1]) for c in checkpoints}})
        _log(f"{dtype}: first divergent step {cmp['first_divergent_step']}, spikes {ts.counters['spikes']:,} vs {ed.counters['spikes']:,}")
    experiments._write(_exp_dir() / "long_equivalence.json", {"rows": rows})


def step_bench(conn: Connectome) -> None:
    g = gains()
    experiments.benchmark(conn, "main", {n: g[n] for n in SIZES}, RATES, SEEDS, STEPS, BASE, log=_log)


def step_bench_50k(conn: Connectome) -> None:
    g = gains()
    experiments.benchmark(conn, "expand_50k", {"expand_50k": g["expand_50k"]}, RATES, (1, 2), 300, BASE, log=_log)


def step_bench_neuropils(conn: Connectome) -> None:
    g = gains()
    experiments.benchmark(conn, "neuropils", {n: g[n] for n in NEUROPILS if n in g}, RATES, (1,), STEPS, BASE, log=_log)


PATTERNS = {
    "burst": {"pattern": "burst", "rate": 0.05, "burst_period": 200, "burst_on": 50},
    "pulse": {"pattern": "pulse", "pulse_every": 100, "pulse_fraction": 0.1},
    "sparse_pattern": {"pattern": "sparse_pattern", "pattern_neurons": 20, "pattern_length": 50, "pattern_every": 100},
}


def step_patterns(conn: Connectome) -> None:
    g = gains()
    for label, change in PATTERNS.items():
        change = dict(change)
        rate = change.pop("rate", 0.0)
        experiments.benchmark(conn, f"pattern_{label}", {n: g[n] for n in ("expand_1k", "expand_10k")}, (rate,), SEEDS, STEPS,
                              BASE, input_changes=change, log=_log)


def step_sensitivity(conn: Connectome, names=("expand_1k", "neuropil_SPS_R")) -> None:
    g = gains()
    for name in names:
        _log(f"sensitivity {name} at gain {g[name]:.4g}")
        result = experiments.sensitivity(conn, name, BASE.replace(weights={"gain": g[name]}), log=_log)
        experiments._write(_exp_dir() / f"sensitivity_{name}.json", result)


def step_nulls(conn: Connectome, name: str = "expand_1k") -> None:
    g = gains()
    result = experiments.null_controls(conn, name, BASE.replace(weights={"gain": g[name]}), log=_log)
    experiments._write(_exp_dir() / f"nulls_{name}.json", result)


def step_profile(conn: Connectome) -> None:
    g = gains()
    rows = []
    for name in ("expand_1k", "expand_10k"):
        rows += experiments.profile_phases(conn, name, BASE.replace(weights={"gain": g[name]}), rates=(0.0001, 0.001, 0.01, 0.1))
    experiments._write(_exp_dir() / "profile.json", {"rows": rows, "note": "phase timers add a small constant cost per phase call"})
    _log(f"profile rows: {len(rows)}")


ENERGY_LEVELS = {0.00001: 300_000, 0.001: 40_000, 0.1: 6_000}  # input rate -> steps (>= ~2 s for the faster mode)


def step_energy(conn: Connectome, name: str = "expand_10k", repeats: int = 5) -> None:
    """Interleaved repeats of time-step vs event-driven(auto); records the OS CPU energy ESTIMATE next to wall and CPU time."""
    g = gains()
    out = experiments.results_dir() / "benchmarks" / "energy_estimate.jsonl"
    done = experiments._done(out)
    for repeat in range(1, repeats + 1):
        for rate, steps in ENERGY_LEVELS.items():
            cfg = BASE.replace(weights={"gain": g[name]}, inputs={"rate": rate}, run={"steps": steps, "seed": 1})
            path = experiments.prepare(conn, name, cfg)
            for mode, aggregation in (("time_step", "sparse"), ("event_driven", "auto")):
                run_cfg = cfg.replace(run={"mode": mode, "aggregation": aggregation})
                tag = f"r{repeat}"
                if experiments.experiment_id("energy", name, "real", run_cfg) + f"-{tag}" in done:
                    continue
                record = experiments.run_isolated(path, run_cfg, kind="energy", tag=tag)
                experiments._append(out, record)
                e = record["os_cpu_energy_estimate"] or {}
                _log(f"  repeat {repeat} rate {rate:g} {mode}: wall {record['metrics']['wall_s']:.2f}s cpu {e.get('cpu_time_s', 0):.2f}s "
                     f"estimate {e.get('energy_nj', 0) / 1e9:.2f} J ({record['process'].get('power_source')})")


def step_baseline_rss() -> None:
    experiments._write(_exp_dir() / "empty_process_rss.json", {"peak_rss": experiments.empty_process_rss(),
                                                                  "what": "peak RSS of a worker process that imports the simulator and exits"})


STEPS_ORDER = {
    "subgraphs": step_subgraphs, "calibrate": step_calibrate, "calibrate-pb-glutamate": step_calibrate_pb_glutamate, "long-equivalence": step_long_equivalence,
    "equivalence": step_equivalence, "baseline-rss": step_baseline_rss,
    "bench": step_bench, "bench-50k": step_bench_50k, "bench-neuropils": step_bench_neuropils, "patterns": step_patterns,
    "sensitivity": step_sensitivity, "nulls": step_nulls, "profile": step_profile, "energy": step_energy,
}


def run_step(name: str) -> int:
    if name in ("figures", "estimate", "summary"):
        from . import report

        getattr(report, name)()
        return 0
    fn = STEPS_ORDER[name]
    if name == "baseline-rss":
        fn()
    else:
        fn(Connectome.load("flywire_fafb_v783"))
    return 0
