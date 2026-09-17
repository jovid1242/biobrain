"""`biobrain m25 <step>`: Milestone 2.5 — compiled backend and full-connectome validation.

Raw results go to results/milestone25/. Milestone 2 results are read (and their checksums verified), never written.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .. import paths, runinfo
from ..connectome.store import sha256_file
from . import experiments


def results_dir() -> Path:
    return paths.results_dir() / "milestone25"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write(relative: str, data: dict) -> Path:
    return experiments._write(results_dir() / relative, data)


# ---- baseline ----------------------------------------------------------------------------------------------------------
def _m2_checksums() -> dict[str, str]:
    m2 = experiments.results_dir()
    return {str(p.relative_to(m2)): sha256_file(p) for p in sorted(m2.rglob("*")) if p.is_file()}


def step_freeze_baseline() -> None:
    """Checksums of every Milestone 2 result file plus the per-run numbers M2.5 is compared against."""
    m2 = experiments.results_dir()
    rows = []
    for name in ("main", "expand_50k", "neuropils"):
        for r in experiments.load_jsonl(m2 / "benchmarks" / f"{name}.jsonl"):
            m = r["metrics"]
            rows.append({"file": f"benchmarks/{name}.jsonl", "experiment_id": r["experiment_id"], "subgraph": r["subgraph"]["name"],
                         "mode": r["mode"], "aggregation": r["aggregation"], "input_rate": r["config"]["inputs"]["rate"],
                         "seed": r["seed"], "steps": m["steps"], "gain": r["network"]["gain"], "config_hash": r["config_hash"],
                         "model_hash": r["model_hash"], "stimulus_hash": r["stimulus_hash"], "wall_s": m["wall_s"],
                         "spikes": m["spikes"], "synaptic_events": m["synaptic_events"], "neuron_updates": m["neuron_updates"],
                         "peak_rss": r["process"]["peak_rss"], "power_source": r["process"].get("power_source")})
    commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(m2)], capture_output=True, text=True,
                            cwd=paths.project_root()).stdout.strip()
    _write("baseline_m2.json", {
        "what": "Frozen Milestone 2 baseline: sha256 of every results/milestone2 file and the per-run numbers of its benchmarks",
        "m2_results_commit": commit, "files_sha256": _m2_checksums(), "benchmark_rows": rows,
        "empty_worker_peak_rss": json.loads((m2 / "experiments" / "empty_process_rss.json").read_text())["peak_rss"],
        "m2_full_connectome_estimate": json.loads((m2 / "estimate_full_connectome.json").read_text()),
        "run": runinfo.collect()})
    _log(f"baseline frozen: {len(rows)} benchmark runs, M2 results commit {commit[:7]}")


def m2_unchanged() -> dict:
    frozen = json.loads((results_dir() / "baseline_m2.json").read_text())["files_sha256"]
    now = _m2_checksums()
    changed = sorted(k for k in frozen if now.get(k) != frozen[k])
    return {"m2_files": len(frozen), "changed_or_missing": changed, "new_files": sorted(set(now) - set(frozen)),
            "unchanged": not changed and set(now) == set(frozen)}


STEPS = {"freeze-baseline": step_freeze_baseline}


def run_step(name: str) -> int:
    if name not in STEPS:
        raise SystemExit(f"unknown step {name!r}; steps: {', '.join(STEPS)}")
    STEPS[name]()
    return 0
