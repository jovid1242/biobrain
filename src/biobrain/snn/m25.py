"""`biobrain m25 <step>`: Milestone 2.5 — compiled backend and full-connectome validation.

Raw results go to results/milestone25/. Milestone 2 results are read (and their checksums verified), never written.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

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


# ---- profile before optimisation (NumPy engine, unchanged) ------------------------------------------------------------
PROFILE_SIZES = ("expand_1k", "expand_10k", "expand_50k")
PROFILE_RATES = (0.00001, 0.0001, 0.001, 0.01, 0.1)
NUMPY_MODES = (("time_step", "sparse"), ("event_driven", "sparse"), ("event_driven", "auto"))


def _bytecode_ops(fn, steps: int) -> dict:
    """Executed bytecode instructions per step inside the engine's own functions (sys.monitoring). Nearly every
    operator, subscript and call there acts on a NumPy array, so this counts NumPy dispatches from Python."""
    import dis
    import sys

    from . import engine

    mon = sys.monitoring
    tool = mon.PROFILER_ID
    codes = [engine.run_time_step.__code__, engine.run_event_driven.__code__, engine.gather.__code__, engine._lazy.__code__]
    # Python 3.14 folds subscripts into BINARY_OP ("[]"), so the operator's argrepr separates them
    names = {c: {i.offset: (f"{i.opname}[]" if i.opname == "BINARY_OP" and i.argrepr == "[]" else i.opname)
                 for i in dis.get_instructions(c)} for c in codes}
    counts: dict[str, int] = {}

    def on_instruction(code, offset):
        op = names[code].get(offset, "?")
        counts[op] = counts.get(op, 0) + 1

    mon.use_tool_id(tool, "biobrain-m25")
    try:
        mon.register_callback(tool, mon.events.INSTRUCTION, on_instruction)
        for c in codes:
            mon.set_local_events(tool, c, mon.events.INSTRUCTION)
        fn()
    finally:
        for c in codes:
            mon.set_local_events(tool, c, 0)
        mon.register_callback(tool, mon.events.INSTRUCTION, None)
        mon.free_tool_id(tool)
    groups = {"call": ("CALL", "CALL_KW", "CALL_FUNCTION_EX"), "operator": ("BINARY_OP",),
              "subscript_read": ("BINARY_OP[]", "BINARY_SUBSCR", "BINARY_SLICE"), "subscript_write": ("STORE_SUBSCR", "STORE_SLICE"),
              "compare": ("COMPARE_OP",)}
    out = {g: sum(counts.get(op, 0) for op in ops) / steps for g, ops in groups.items()}
    out["all_instructions"] = sum(counts.values()) / steps
    return out


def _cprofile(fn, steps: int) -> dict:
    import cProfile
    import pstats

    prof = cProfile.Profile()
    prof.enable()
    fn()
    prof.disable()
    stats = pstats.Stats(prof).stats
    total_calls = sum(nc for (_, _, _), (_, nc, _, _, _) in stats.items())
    numpy_calls, rows = 0, []
    for (file, _, func), (_, nc, tt, _, _) in stats.items():
        is_numpy = "numpy" in func or "/numpy/" in file
        numpy_calls += nc if is_numpy else 0
        rows.append({"function": func if file == "~" else f"{Path(file).name}:{func}", "calls_per_step": nc / steps,
                     "seconds_per_step": tt / steps, "numpy": is_numpy})
    rows.sort(key=lambda r: -r["seconds_per_step"])
    return {"python_visible_calls_per_step": total_calls / steps, "numpy_function_calls_per_step": numpy_calls / steps,
            "top_by_own_time": rows[:12],
            "note": "cProfile sees function and method calls only; in-place operators and fancy indexing are C slots (see bytecode_ops)"}


def _temporary_memory(fn) -> dict:
    import tracemalloc

    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        result = fn()
        after, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    persistent = sum(v for k, v in result.memory.items() if k in ("neuron_state", "event_buffers", "instrumentation", "decay_table"))
    return {"peak_traced_bytes_above_start": peak - before, "retained_bytes_after_run": after - before,
            "engine_persistent_arrays_bytes": persistent,
            "transient_peak_bytes": max(peak - after, 0),
            "note": "tracemalloc sees NumPy data buffers; transient = peak during the run minus what the run retained"}


def step_profile_numpy(sizes=PROFILE_SIZES, rates=PROFILE_RATES, steps: int = 500, count_steps: int = 200) -> None:
    from ..connectome.store import Connectome
    from . import engine
    from .pipeline import BASE, gains

    conn = Connectome.load("flywire_fafb_v783")
    g = gains()
    rows = []
    for name in sizes:
        for rate in rates:
            cfg = BASE.replace(weights={"gain": g[name]}, inputs={"rate": rate}, run={"steps": steps, "seed": 1})
            net, sched, meta = experiments.load_prepared(experiments.prepare(conn, name, cfg))
            p = cfg.neuron
            for mode, agg in NUMPY_MODES:
                run = lambda k, **kw: engine.run(net, sched, p, k, mode, aggregation=agg, **kw)
                run(50)  # warm-up
                timed = run(steps, profile=True)
                total = sum(timed.phases.values())
                row = {"subgraph": name, "neurons": net.n, "input_rate": rate, "mode": mode, "aggregation": agg, "steps": steps,
                       "wall_s": timed.wall_s, "phases_s": timed.phases, "phase_share": {k: v / total for k, v in timed.phases.items()},
                       "spikes": timed.counters["spikes"], "synaptic_events": timed.counters["synaptic_events"],
                       "neuron_updates": timed.counters["neuron_updates"],
                       "cprofile": _cprofile(lambda: run(count_steps), count_steps),
                       "bytecode_ops": _bytecode_ops(lambda: run(count_steps), count_steps),
                       "memory": _temporary_memory(lambda: run(count_steps))}
                rows.append(row)
                _log(f"{name} {rate:g} {mode}/{agg}: {1e6 * timed.wall_s / steps:.1f} us/step, "
                     f"{row['cprofile']['python_visible_calls_per_step']:.0f} calls + "
                     f"{row['bytecode_ops']['operator'] + row['bytecode_ops']['subscript_read'] + row['bytecode_ops']['subscript_write']:.0f} "
                     f"operator/subscript ops per step, transient {row['memory']['transient_peak_bytes'] / 2**20:.1f} MiB")
    _write("profile/profile_numpy.json", {"rows": rows, "gains": g, "run": runinfo.collect(),
                                          "note": "NumPy engine of Milestone 2, unchanged; phase timers add a small cost per phase"})



# ---- compiled time-step first: control baseline on real subgraphs --------------------------------------------------------
def _identical(a, b) -> dict:
    """Bit-level comparison of two runs of the same mode (criteria of MILESTONE25_PLAN §3)."""
    raster = a.raster is not None and b.raster is not None and np.array_equal(a.raster[0], b.raster[0]) \
        and np.array_equal(a.raster[1], b.raster[1])
    arrays = all(np.array_equal(getattr(a, f), getattr(b, f)) for f in ("spikes_per_step", "updates_per_step", "events_per_step", "spike_counts"))
    v_bits = a.final_v.dtype == b.final_v.dtype and a.final_v.tobytes() == b.final_v.tobytes()
    dv = float(np.max(np.abs(a.final_v.astype(np.float64) - b.final_v.astype(np.float64)))) if a.n else 0.0
    return {"raster_identical": bool(raster), "per_step_and_per_neuron_counters_identical": bool(arrays),
            "counters_identical": a.counters == b.counters, "final_v_bit_identical": bool(v_bits), "max_abs_final_v_difference": dv,
            "identical": bool(raster and arrays and a.counters == b.counters and v_bits)}


def step_check_ts(sizes=("expand_100", "expand_1k", "expand_10k", "expand_50k"), rates=(0.0001, 0.001, 0.01, 0.1),
                  steps: int = 1000) -> None:
    """Compiled time-step against NumPy time-step before any event-driven work: same arrays, same seed, in-process."""
    from ..connectome.store import Connectome
    from . import compiled, engine
    from .pipeline import BASE, gains

    conn = Connectome.load("flywire_fafb_v783")
    g = gains()
    compile_s = compiled.warmup("float32", modes=("time_step",))
    rows = []
    for name in sizes:
        for rate in rates:
            cfg = BASE.replace(weights={"gain": g[name]}, inputs={"rate": rate}, run={"steps": steps, "seed": 1})
            net, sched, _ = experiments.load_prepared(experiments.prepare(conn, name, cfg))
            p = cfg.neuron
            numpy_run = engine.run(net, sched, p, steps, "time_step", record_spikes=True)
            numba_run = engine.run(net, sched, p, steps, "time_step", backend="numba", record_spikes=True)
            numpy_wall = min(numpy_run.wall_s, engine.run(net, sched, p, steps, "time_step").wall_s)
            numba_wall = min(numba_run.wall_s, engine.run(net, sched, p, steps, "time_step", backend="numba").wall_s)
            row = {"subgraph": name, "neurons": net.n, "edges": net.m, "input_rate": rate, "steps": steps, "dtype": p.dtype,
                   "spikes": numpy_run.counters["spikes"], "synaptic_events": numpy_run.counters["synaptic_events"],
                   "numpy_wall_s": numpy_wall, "compiled_wall_s": numba_wall, "compiled_speedup": numpy_wall / numba_wall,
                   **_identical(numpy_run, numba_run)}
            rows.append(row)
            _log(f"{name} {rate:g}: NumPy {1e6 * numpy_wall / steps:.1f} us/step, compiled {1e6 * numba_wall / steps:.1f} us/step "
                 f"({row['compiled_speedup']:.1f}x), identical {row['identical']}")
    _write("experiments/check_compiled_time_step.json",
           {"rows": rows, "compile_or_cache_load_s": compile_s, "run": runinfo.collect(),
            "note": "in-process check (min of 2 runs each); the isolated benchmark matrix is separate"})



def step_check_ed(sizes=("expand_1k", "expand_10k", "expand_50k"), rates=(0.00001, 0.0001, 0.001, 0.01, 0.1), seeds=(1, 2, 3),
                  steps: int = 1000) -> None:
    """Compiled event-driven variants against NumPy event-driven (auto), in-process.

    Seed 1 of each cell is recorded and compared bit by bit with NumPy. Timing runs (no recording) go in a shuffled
    order. Selection rule for the default variant, fixed before this step runs: the lowest geometric mean over all
    (size, input) cells of the per-cell median wall time; a tie keeps `touched`."""
    import random

    from ..connectome.store import Connectome
    from . import compiled, engine
    from .pipeline import BASE, gains

    conn = Connectome.load("flywire_fafb_v783")
    g = gains()
    compile_s = compiled.warmup("float32", variants=compiled.VARIANTS)
    rows = []
    for name in sizes:
        for rate in rates:
            for seed in seeds:
                cfg = BASE.replace(weights={"gain": g[name]}, inputs={"rate": rate}, run={"steps": steps, "seed": seed})
                net, sched, _ = experiments.load_prepared(experiments.prepare(conn, name, cfg))
                p = cfg.neuron
                runners = {"numpy_event_driven_auto": lambda **kw: engine.run(net, sched, p, steps, "event_driven", aggregation="auto", **kw),
                           "compiled_time_step": lambda **kw: engine.run(net, sched, p, steps, "time_step", backend="numba", **kw)}
                for variant in compiled.VARIANTS:
                    runners[f"compiled_event_driven_{variant}"] = (
                        lambda variant=variant, **kw: engine.run(net, sched, p, steps, "event_driven", backend="numba", variant=variant, **kw))
                row = {"subgraph": name, "neurons": net.n, "input_rate": rate, "seed": seed, "steps": steps}
                if seed == seeds[0]:
                    ref = runners["numpy_event_driven_auto"](record_spikes=True)
                    row["identical_to_numpy_event_driven"] = {
                        v: _identical(ref, runners[f"compiled_event_driven_{v}"](record_spikes=True)) for v in compiled.VARIANTS}
                order = list(runners)
                random.Random(f"{name}-{rate}-{seed}").shuffle(order)
                walls = {}
                for key in order:
                    res = runners[key]()
                    walls[key] = res.wall_s
                    if key == "compiled_event_driven_touched":
                        row.update(spikes=res.counters["spikes"], synaptic_events=res.counters["synaptic_events"],
                                   neuron_updates=res.counters["neuron_updates"], unique_targets=res.extra["unique_targets"],
                                   peak_pending_events=res.extra["peak_pending_events"])
                row["wall_s"] = walls
                row["order"] = order
                rows.append(row)
                ident = row.get("identical_to_numpy_event_driven")
                _log(f"{name} {rate:g} s{seed}: " + ", ".join(f"{k.replace('compiled_', 'c_').replace('numpy_', 'np_')} "
                                                              f"{1e6 * w / steps:.1f}" for k, w in walls.items())
                     + " us/step" + (f"; identical: {[v['identical'] for v in ident.values()]}" if ident else ""))
    cells = {}
    for r in rows:
        for key, wall in r["wall_s"].items():
            cells.setdefault(key, {}).setdefault((r["subgraph"], r["input_rate"]), []).append(wall)
    geo = {key: float(np.exp(np.mean([np.log(np.median(w)) for w in per.values()]))) for key, per in cells.items()}
    ed = {v: geo[f"compiled_event_driven_{v}"] for v in compiled.VARIANTS}
    best = min(ed, key=lambda v: (ed[v], v != "touched"))
    _write("experiments/check_compiled_event_driven.json",
           {"rows": rows, "geometric_mean_wall_s": geo, "selected_variant": best,
            "selection_rule": "lowest geometric mean over (size, input) cells of the per-cell median wall; tie keeps touched",
            "compile_or_cache_load_s": compile_s, "run": runinfo.collect()})
    _log(f"selected compiled event-driven variant: {best} ({ {k: round(v * 1e3, 2) for k, v in ed.items()} } ms geo-mean)")


STEPS = {"freeze-baseline": step_freeze_baseline, "profile-numpy": step_profile_numpy, "check-ts": step_check_ts,
         "check-ed": step_check_ed}


def run_step(name: str) -> int:
    if name not in STEPS:
        raise SystemExit(f"unknown step {name!r}; steps: {', '.join(STEPS)}")
    STEPS[name]()
    return 0
