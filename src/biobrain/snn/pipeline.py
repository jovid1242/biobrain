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
    out = {}
    for path in sorted(_exp_dir().glob("calibration_*.json")):
        data = json.loads(path.read_text())
        if data["baseline_gain"] is not None:
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


def step_baseline_rss() -> None:
    experiments._write(_exp_dir() / "empty_process_rss.json", {"peak_rss": experiments.empty_process_rss(),
                                                                  "what": "peak RSS of a worker process that imports the simulator and exits"})


STEPS_ORDER = {
    "subgraphs": step_subgraphs, "calibrate": step_calibrate, "equivalence": step_equivalence, "baseline-rss": step_baseline_rss,
    "bench": step_bench, "bench-50k": step_bench_50k, "bench-neuropils": step_bench_neuropils, "patterns": step_patterns,
    "sensitivity": step_sensitivity, "nulls": step_nulls, "profile": step_profile,
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
