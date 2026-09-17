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



# ---- isolated benchmark matrix ---------------------------------------------------------------------------------------------
BENCH_SIZES = ("expand_1k", "expand_10k", "expand_50k")
BENCH_RATES = (0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5)
BENCH_SEEDS = (1, 2, 3)
BENCH_STEPS = 1000
ORDER_SEED = 20260917
BACKENDS = {  # label -> (backend, mode, NumPy aggregation)
    "numpy_time_step": ("numpy", "time_step", "sparse"),
    "numpy_event_driven": ("numpy", "event_driven", "auto"),
    "compiled_time_step": ("numba", "time_step", "sparse"),
    "compiled_event_driven": ("numba", "event_driven", "auto"),  # aggregation is NumPy-only; kept equal so config_hash matches
}


def selected_variant() -> str:
    return json.loads((results_dir() / "experiments" / "check_compiled_event_driven.json").read_text())["selected_variant"]


def step_baseline_rss(repeats: int = 3) -> None:
    out = {backend: sorted(experiments.empty_process_rss(backend) for _ in range(repeats)) for backend in ("numpy", "numba")}
    _write("experiments/empty_process_rss.json", {
        "peak_rss_bytes": out, "median_bytes": {k: v[len(v) // 2] for k, v in out.items()},
        "what": "peak RSS of a worker that imports the simulator (numba: and compiles/loads every kernel) and exits",
        "run": runinfo.collect()})
    _log({k: f"{v[len(v) // 2] / 2**20:.1f} MiB" for k, v in out.items()})


def bench_matrix(conn, out_name: str, subgraphs: dict[str, float], rates, seeds, steps: int, backends=tuple(BACKENDS),
                 order_seed: int = ORDER_SEED, input_changes: dict | None = None) -> Path:
    """Every run in its own process, in one shuffled order across sizes, inputs, seeds and backends; resumable."""
    import random

    from .pipeline import BASE

    variant = selected_variant()
    out = results_dir() / "benchmarks" / f"{out_name}.jsonl"
    done = experiments._done(out)
    runs = [(name, rate, seed, label) for name in subgraphs for rate in rates for seed in seeds for label in backends]
    random.Random(order_seed).shuffle(runs)
    for index, (name, rate, seed, label) in enumerate(runs):
        backend, mode, aggregation = BACKENDS[label]
        cfg = BASE.replace(weights={"gain": subgraphs[name]}, inputs={"rate": rate, **(input_changes or {})},
                           run={"steps": steps, "seed": seed, "mode": mode, "aggregation": aggregation})
        eid = experiments.experiment_id("benchmark", name, "real", cfg, backend, variant, prefix="m25")
        if eid in done:
            continue
        path = experiments.prepare(conn, name, cfg)
        t = time.monotonic()
        record = experiments.run_isolated(path, cfg, "benchmark", backend=backend, variant=variant, prefix="m25")
        record.update(backend_label=label, order_index=index, order_seed=order_seed, runs_in_matrix=len(runs))
        experiments._append(out, record)
        m = record["metrics"]
        _log(f"[{index + 1}/{len(runs)}] {name} {rate:g} s{seed} {label}: {1e6 * m['wall_s'] / m['steps']:.1f} us/step, "
             f"peak RSS {record['process']['peak_rss'] / 2**20:.0f} MiB, {record['process'].get('power_source')} "
             f"{record['process'].get('battery_percent')}% ({time.monotonic() - t:.1f}s)")
    return out


def step_bench() -> None:
    from ..connectome.store import Connectome
    from .pipeline import gains

    g = gains()
    bench_matrix(Connectome.load("flywire_fafb_v783"), "matrix", {n: g[n] for n in BENCH_SIZES}, BENCH_RATES, BENCH_SEEDS, BENCH_STEPS)



# ---- profile after compilation ----------------------------------------------------------------------------------------------
def step_profile_compiled(sizes=PROFILE_SIZES, rates=PROFILE_RATES, steps: int = 500) -> None:
    """Phase timers inside the compiled kernels (clock_gettime_nsec_np), same grid and step count as profile-numpy."""
    from ..connectome.store import Connectome
    from . import compiled, engine
    from .pipeline import BASE, gains

    conn = Connectome.load("flywire_fafb_v783")
    g = gains()
    variant = selected_variant()
    compiled.warmup("float32", variants=(variant,))
    clock_ns = float(compiled.clock_cost_ns(1_000_000))
    rows = []
    for name in sizes:
        for rate in rates:
            cfg = BASE.replace(weights={"gain": g[name]}, inputs={"rate": rate}, run={"steps": steps, "seed": 1})
            net, sched, _ = experiments.load_prepared(experiments.prepare(conn, name, cfg))
            for mode in ("time_step", "event_driven"):
                run = lambda k, **kw: engine.run(net, sched, cfg.neuron, k, mode, backend="numba", variant=variant, **kw)
                run(50)
                plain = min(run(steps).wall_s for _ in range(3))
                timed = run(steps, profile=True)
                total = sum(timed.phases.values())
                timer_calls = steps * (6 if mode == "time_step" else 7)
                rows.append({"subgraph": name, "neurons": net.n, "input_rate": rate, "mode": mode, "backend": "numba",
                             "variant": variant if mode == "event_driven" else None, "steps": steps, "wall_s": timed.wall_s,
                             "wall_s_without_timers": plain, "phases_s": timed.phases,
                             "phase_share": {k: v / total for k, v in timed.phases.items()},
                             "timer_overhead_s_estimate": timer_calls * clock_ns * 1e-9,
                             "spikes": timed.counters["spikes"], "synaptic_events": timed.counters["synaptic_events"],
                             "neuron_updates": timed.counters["neuron_updates"],
                             "unique_targets": timed.extra.get("unique_targets")})
                _log(f"{name} {rate:g} {mode}: {1e6 * plain / steps:.1f} us/step; " +
                     ", ".join(f"{k} {100 * v:.0f}%" for k, v in rows[-1]["phase_share"].items()))
    _write("profile/profile_compiled.json", {"rows": rows, "clock_ns_per_call": clock_ns, "variant": variant,
                                             "run": runinfo.collect()})


# ---- equivalence on real subgraphs ---------------------------------------------------------------------------------------------
def _cross_mode(a, b, dtype: str) -> dict:
    """Different modes: identical rasters and counters, potentials within the M2 absolute and the M2.5 relative tolerance."""
    from . import metrics

    cmp = metrics.compare(a, b, metrics.V_TOLERANCE[dtype])
    fa, fb = a.final_v.astype(np.float64), b.final_v.astype(np.float64)
    scale = np.maximum(1.0, np.maximum(np.abs(fa), np.abs(fb)))
    rel_ok = bool(np.all(np.abs(fa - fb) <= metrics.V_TOLERANCE[dtype] * scale)) if a.n else True
    return {**cmp, "within_m2_absolute_tolerance": cmp["equivalent"],
            "within_m25_relative_tolerance": bool(cmp["spikes_equal"] and cmp["raster_equal"] is not False and
                                                  (rel_ok if dtype == "float32" else cmp["max_abs_final_v_difference"] <= 1e-9))}


def step_equivalence(sizes=("expand_1k", "expand_10k", "expand_50k", "full"), rates=(0.0001, 0.001, 0.01, 0.1), steps: int = 1000) -> None:
    from ..connectome.store import Connectome
    from . import engine
    from .pipeline import BASE, gains

    conn = Connectome.load("flywire_fafb_v783")
    g = {**gains(), **full_gain()}
    variant = selected_variant()
    rows = []
    for name in sizes:
        if name not in g:
            _log(f"skip {name}: no calibrated gain yet")
            continue
        for rate in rates:
            for dtype in ("float32", "float64"):
                cfg = BASE.replace(weights={"gain": g[name]}, neuron={"dtype": dtype}, inputs={"rate": rate},
                                   run={"steps": steps, "seed": 1})
                net, sched, _ = experiments.load_prepared(experiments.prepare(conn, name, cfg))
                kw = dict(record_spikes=True, record_voltage_every=100)
                runs = {"numpy_time_step": engine.run(net, sched, cfg.neuron, steps, "time_step", **kw),
                        "numpy_event_driven": engine.run(net, sched, cfg.neuron, steps, "event_driven", aggregation="auto", **kw),
                        "compiled_time_step": engine.run(net, sched, cfg.neuron, steps, "time_step", backend="numba", **kw),
                        "compiled_event_driven": engine.run(net, sched, cfg.neuron, steps, "event_driven", backend="numba",
                                                            variant=variant, **kw)}
                row = {"subgraph": name, "neurons": net.n, "input_rate": rate, "dtype": dtype, "steps": steps,
                       "spikes": runs["numpy_time_step"].counters["spikes"],
                       "same_mode_numpy_vs_compiled": {
                           "time_step": _identical(runs["numpy_time_step"], runs["compiled_time_step"]),
                           "event_driven": _identical(runs["numpy_event_driven"], runs["compiled_event_driven"])},
                       "cross_mode": {
                           "compiled_time_step_vs_compiled_event_driven": _cross_mode(runs["compiled_time_step"], runs["compiled_event_driven"], dtype),
                           "numpy_time_step_vs_compiled_event_driven": _cross_mode(runs["numpy_time_step"], runs["compiled_event_driven"], dtype)}}
                row["voltage_records_identical"] = (runs["numpy_time_step"].voltage == runs["compiled_time_step"].voltage and
                                                    runs["numpy_event_driven"].voltage == runs["compiled_event_driven"].voltage)
                rows.append(row)
                same = [v["identical"] for v in row["same_mode_numpy_vs_compiled"].values()]
                cross = row["cross_mode"]["compiled_time_step_vs_compiled_event_driven"]
                _log(f"{name} {rate:g} {dtype}: same-mode bit-identical {same}, voltage {row['voltage_records_identical']}; "
                     f"compiled TS vs ED: spikes {cross['spikes_equal']}, |dv| {cross['max_abs_final_v_difference']:.2g}, "
                     f"M2 tol {cross['within_m2_absolute_tolerance']}, M2.5 tol {cross['within_m25_relative_tolerance']}")
    _write("experiments/equivalence.json", {"rows": rows, "variant": variant, "criteria": "docs/MILESTONE25_PLAN.md §3",
                                            "run": runinfo.collect()})


def step_long_equivalence(name: str = "expand_10k", steps: int = 40_000, rate: float = 0.001) -> None:
    from ..connectome.store import Connectome
    from . import engine, metrics
    from .pipeline import BASE, gains

    conn = Connectome.load("flywire_fafb_v783")
    g = gains()
    variant = selected_variant()
    rows = []
    for dtype in ("float64", "float32"):
        cfg = BASE.replace(weights={"gain": g[name]}, neuron={"dtype": dtype}, inputs={"rate": rate}, run={"steps": steps, "seed": 1})
        net, sched, _ = experiments.load_prepared(experiments.prepare(conn, name, cfg))
        runs = {"numpy_time_step": engine.run(net, sched, cfg.neuron, steps, "time_step", record_spikes=True),
                "compiled_time_step": engine.run(net, sched, cfg.neuron, steps, "time_step", backend="numba", record_spikes=True),
                "numpy_event_driven": engine.run(net, sched, cfg.neuron, steps, "event_driven", aggregation="auto", record_spikes=True),
                "compiled_event_driven": engine.run(net, sched, cfg.neuron, steps, "event_driven", backend="numba", variant=variant,
                                                    record_spikes=True)}
        for a, b in (("numpy_time_step", "compiled_time_step"), ("numpy_event_driven", "compiled_event_driven"),
                     ("compiled_time_step", "compiled_event_driven")):
            ra, rb = runs[a], runs[b]
            rows.append({"dtype": dtype, "a": a, "b": b, "first_divergent_step": metrics.first_divergent_step(ra.raster, rb.raster),
                         "spikes_a": ra.counters["spikes"], "spikes_b": rb.counters["spikes"],
                         "bit_identical": _identical(ra, rb)["identical"],
                         "wall_s": {a: ra.wall_s, b: rb.wall_s}})
            _log(f"{dtype} {a} vs {b}: first divergent step {rows[-1]['first_divergent_step']}, bit-identical {rows[-1]['bit_identical']}")
    _write("experiments/long_equivalence.json", {"subgraph": name, "steps": steps, "input_rate": rate, "variant": variant,
                                                 "rows": rows, "run": runinfo.collect()})


# ---- full connectome ----------------------------------------------------------------------------------------------------------
FULL = "full"
FULL_DIR = "full_connectome"


def full_gain() -> dict[str, float]:
    path = results_dir() / FULL_DIR / "calibration.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {FULL: data["baseline_gain"]} if data["baseline_gain"] is not None else {}


def _build_full_worker() -> None:
    """Isolated process: extract the full subgraph and build its network; prints peak RSS (JSON)."""
    import resource
    import sys as _sys

    from ..connectome.store import Connectome
    from . import network, subgraph
    from .pipeline import BASE

    t0 = time.perf_counter()
    conn = Connectome.load("flywire_fafb_v783")
    sub = subgraph.load(conn, FULL)
    t1 = time.perf_counter()
    peak_sub = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    net = network.build(conn, sub, BASE.replace(weights={"gain": 0.03}))
    t2 = time.perf_counter()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if _sys.platform == "darwin" else 1024
    print(json.dumps({"peak_rss_after_subgraph": peak_sub * scale, "peak_rss_after_network_build": peak * scale,
                      "subgraph_s": t1 - t0, "network_build_s": t2 - t1, "neurons": net.n, "simulated_edges": net.m,
                      "network_arrays_bytes": sum(net.memory().values())}))


def step_full_build() -> None:
    import sys as _sys

    from ..connectome.store import Connectome
    from . import subgraph

    conn = Connectome.load("flywire_fafb_v783")
    path = subgraph.manifests_dir(FULL) / f"{FULL}.json"
    if not path.is_file():
        t0 = time.perf_counter()
        subgraph.save(subgraph.build(conn, FULL, subgraph.CANONICAL[FULL]))
        _log(f"full subgraph extracted in {time.perf_counter() - t0:.1f}s")
    manifest = json.loads(path.read_text())
    proc = subprocess.run([_sys.executable, "-c", "from biobrain.snn import m25; m25._build_full_worker()"], capture_output=True,
                          text=True, timeout=3600, cwd=paths.project_root())
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    build = json.loads(proc.stdout.strip().splitlines()[-1])
    _write(f"{FULL_DIR}/build.json", {"manifest": {k: manifest[k] for k in ("neurons", "edges", "synapses", "weak_components",
                                                                          "largest_weak_component", "strong_components",
                                                                          "largest_strong_component", "sha256")},
                                      "isolated_build": build, "run": runinfo.collect()})
    _log(f"full: {manifest['neurons']:,} neurons, {manifest['edges']:,} edges; build peak RSS "
         f"{build['peak_rss_after_network_build'] / 2**20:.0f} MiB in {build['subgraph_s'] + build['network_build_s']:.1f}s")


def _fit(rows: list[dict], columns) -> tuple[np.ndarray, dict]:
    from scipy.optimize import nnls

    y = np.array([r["y"] for r in rows])
    X = np.array([[r[c] for c in columns] for r in rows], dtype=float)
    coef, _ = nnls(X / y[:, None], np.ones_like(y))
    rel = np.abs(X @ coef - y) / y
    return coef, {"terms": list(columns), "seconds": coef.tolist(), "runs": len(rows),
                  "median_abs_relative_error": float(np.median(rel)), "p90_abs_relative_error": float(np.quantile(rel, 0.9)),
                  "max_abs_relative_error": float(rel.max())}


def cost_models(records: list[dict]) -> dict:
    """Per-step cost models fitted on isolated compiled and NumPy runs (relative least squares, non-negative terms).
    TS: base + N + E; ED: base + A (updated neurons) + E. Empirical, not a physical law."""
    out = {}
    for label in BACKENDS:
        rows = []
        for r in records:
            if r.get("backend_label") != label:
                continue
            m = r["metrics"]
            steps = m["steps"]
            rows.append({"y": m["wall_s"] / steps, "base": 1.0, "N": m["neurons"], "A": m["neuron_updates"] / steps,
                         "E": m["synaptic_events"] / steps})
        if len(rows) >= 4:
            _, model = _fit(rows, ("base", "N", "E") if label.endswith("time_step") else ("base", "A", "E"))
            out[label] = model
    return out


def step_full_estimate() -> None:
    """ESTIMATE before the full-connectome run: time from cost models fitted on the 1k-50k matrix, RAM from arrays."""
    matrix = experiments.load_jsonl(results_dir() / "benchmarks" / "matrix.jsonl")
    build = json.loads((results_dir() / FULL_DIR / "build.json").read_text())
    models = cost_models(matrix)
    n, m = build["isolated_build"]["neurons"], build["isolated_build"]["simulated_edges"]
    empty = json.loads((results_dir() / "experiments" / "empty_process_rss.json").read_text())["median_bytes"]
    worst_fraction = 1 / 3  # refractory-limited maximum spike fraction per step (ref = 2)
    scenarios = []
    for spike_fraction in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, worst_fraction):
        events = spike_fraction * n * m / n
        touched = min(n, events) if spike_fraction < worst_fraction else n
        row = {"spike_fraction_per_step": spike_fraction, "events_per_step": events}
        for label, model in models.items():
            c = model["seconds"]
            row[f"{label}_s_per_step"] = c[0] + c[1] * (n if label.endswith("time_step") else touched) + c[2] * events
        scenarios.append(row)
    worst = scenarios[-1]
    calib_s = 19 * 3 * 600 * worst["compiled_time_step_s_per_step"]
    smoke_compiled_s = 5 * 3 * 1000 * (worst["compiled_time_step_s_per_step"] + worst["compiled_event_driven_s_per_step"])
    realistic = next(r for r in scenarios if r["spike_fraction_per_step"] == 1e-2)
    smoke_numpy_s = 5 * 3 * 1000 * (realistic["numpy_time_step_s_per_step"] + realistic["numpy_event_driven_s_per_step"])
    net_bytes = (n + 1) * 8 + m * (4 + 4)
    ram = {"numba_empty_worker": empty["numba"], "numpy_empty_worker": empty["numpy"], "network_arrays": net_bytes,
           "neuron_state_and_buffers_upper": n * (4 + 4 + 4 + 8 + 1 + 4 + 4 + 4),
           "input_schedule_at_10pct_input": int(0.1 * n * 1000 * 4 + 1001 * 8),
           "build_peak_rss_measured": build["isolated_build"]["peak_rss_after_network_build"]}
    ram["worker_upper_bytes"] = ram["numba_empty_worker"] + net_bytes + ram["neuron_state_and_buffers_upper"] + ram["input_schedule_at_10pct_input"]
    decision = {"ram_below_8_gib": max(ram["worker_upper_bytes"], ram["build_peak_rss_measured"]) < 8 * 2**30,
                "calibration_worst_case_s": calib_s, "smoke_compiled_worst_case_s": smoke_compiled_s,
                "smoke_numpy_at_1pct_spikes_s": smoke_numpy_s}
    decision["run_full_connectome"] = decision["ram_below_8_gib"] and calib_s + smoke_compiled_s < 3600
    decision["include_numpy_backends"] = smoke_numpy_s < 1800
    decision["include_10pct_input"] = decision["run_full_connectome"]
    _write(f"{FULL_DIR}/estimate_before_run.json", {
        "label": "ESTIMATE — made before the full-connectome run from the 1k-50k matrix; compared with measurements afterwards",
        "neurons": n, "simulated_edges": m, "cost_models": models, "scenarios": scenarios, "ram_bytes": ram, "decision": decision,
        "run": runinfo.collect()})
    _log(f"estimate: worker <= {ram['worker_upper_bytes'] / 2**30:.2f} GiB, build {ram['build_peak_rss_measured'] / 2**30:.2f} GiB; "
         f"calibration worst case {calib_s / 60:.1f} min, compiled smoke worst {smoke_compiled_s / 60:.1f} min, "
         f"NumPy smoke at 1% spikes {smoke_numpy_s / 60:.1f} min -> {decision}")


def step_full_calibrate() -> None:
    from ..connectome.store import Connectome
    from . import engine
    from .pipeline import BASE

    est = json.loads((results_dir() / FULL_DIR / "estimate_before_run.json").read_text())["decision"]
    if not est["run_full_connectome"]:
        raise SystemExit("ESTIMATE says the full-connectome run is not safe; not running (see estimate_before_run.json)")
    conn = Connectome.load("flywire_fafb_v783")
    spot = []
    for gain in (0.03, 0.3):  # NumPy time-step spot check on the full brain before trusting compiled calibration
        cfg = BASE.replace(weights={"gain": gain}, inputs={"rate": 0.01, "on_steps": 300}, run={"steps": 600, "seed": 1})
        net, sched, _ = experiments.load_prepared(experiments.prepare(conn, FULL, cfg))
        a = engine.run(net, sched, cfg.neuron, 600, "time_step", record_spikes=True, record_voltage_every=10)
        b = engine.run(net, sched, cfg.neuron, 600, "time_step", backend="numba", record_spikes=True, record_voltage_every=10)
        spot.append({"gain": gain, "spikes": a.counters["spikes"], "numpy_wall_s": a.wall_s, "compiled_wall_s": b.wall_s,
                     **_identical(a, b), "voltage_records_identical": a.voltage == b.voltage})
        _log(f"spot check gain {gain}: identical {spot[-1]['identical']}, NumPy {a.wall_s:.2f}s, compiled {b.wall_s:.2f}s")
    if not all(s["identical"] and s["voltage_records_identical"] for s in spot):
        raise SystemExit("compiled time-step differs from NumPy on the full brain; calibration not run")
    t0 = time.perf_counter()
    result = experiments.calibrate(conn, FULL, BASE, log=_log, backend="numba")
    result.update(backend="numba (bit-identical to NumPy time-step: tests, spot check)", numpy_spot_check=spot,
                  calibration_wall_s=time.perf_counter() - t0, run=runinfo.collect())
    _write(f"{FULL_DIR}/calibration.json", result)
    _log(f"full: stable gains {result['widest_stable_range']} -> baseline {result['baseline_gain']}")


def step_full_regime() -> None:
    from ..connectome.store import Connectome
    from .pipeline import BASE

    g = full_gain()
    if not g:
        raise SystemExit("no baseline gain for the full connectome (see calibration.json)")
    result = experiments.calibrate(Connectome.load("flywire_fafb_v783"), FULL, BASE, gains=(g[FULL],), log=_log, backend="numba")
    _write(f"{FULL_DIR}/baseline_regime.json", {"baseline_gain": g[FULL], "rows": result["rows"], "probe": result["probe"],
                                                "run": runinfo.collect()})


def step_full_smoke() -> None:
    from ..connectome.store import Connectome

    g = full_gain()
    est = json.loads((results_dir() / FULL_DIR / "estimate_before_run.json").read_text())["decision"]
    rates = (0.00001, 0.0001, 0.001, 0.01) + ((0.1,) if est["include_10pct_input"] else ())
    backends = tuple(BACKENDS) if est["include_numpy_backends"] else ("compiled_time_step", "compiled_event_driven")
    bench_matrix(Connectome.load("flywire_fafb_v783"), "full_smoke", {FULL: g[FULL]}, rates, BENCH_SEEDS, BENCH_STEPS, backends,
                 order_seed=ORDER_SEED + 139_255)


def step_full_probe(rates=(0.0001, 0.001, 0.01), seeds=BENCH_SEEDS, on_steps: int = 500, off_steps: int = 500) -> None:
    """Activity over time with the input switched off halfway (self-sustained recurrent activity; not memory)."""
    from ..connectome.store import Connectome
    from . import engine, metrics
    from .pipeline import BASE

    g = full_gain()
    variant = selected_variant()
    conn = Connectome.load("flywire_fafb_v783")
    steps = on_steps + off_steps
    rows = []
    for rate in rates:
        for seed in seeds:
            cfg = BASE.replace(weights={"gain": g[FULL]}, inputs={"rate": rate, "on_steps": on_steps}, run={"steps": steps, "seed": seed})
            net, sched, _ = experiments.load_prepared(experiments.prepare(conn, FULL, cfg))
            res = engine.run(net, sched, cfg.neuron, steps, "event_driven", backend="numba", variant=variant)
            summary = metrics.summarize(res, cfg.neuron)
            cls = metrics.classify(res, cfg.neuron, on_steps, int(sched.indptr[on_steps]))
            off = res.spikes_per_step[on_steps:]
            silent_after = np.flatnonzero(off == 0)
            rows.append({"input_rate": rate, "seed": seed, "on_steps": on_steps, "off_steps": off_steps,
                         "spikes_per_step": res.spikes_per_step.tolist(), "events_per_step": res.events_per_step.tolist(),
                         "unique_targets_per_step": res.extra["_unique_targets_per_step"].tolist(),
                         "updates_per_step": res.updates_per_step.tolist(),
                         "population_rate_hz_on": float(res.spikes_per_step[:on_steps].sum() / (net.n * on_steps / 1000)),
                         "population_rate_hz_last_third_off": float(off[-(off_steps // 3):].sum() / (net.n * (off_steps // 3) / 1000)),
                         "first_silent_step_after_input_off": int(silent_after[0]) if silent_after.size else None,
                         "classification": cls, "summary": summary})
            _log(f"probe {rate:g} s{seed}: on {rows[-1]['population_rate_hz_on']:.2f} Hz, last third off "
                 f"{rows[-1]['population_rate_hz_last_third_off']:.2f} Hz, regime {cls['regime']}, self-sustained {cls['self_sustained']}")
    _write(f"{FULL_DIR}/probe.json", {"rows": rows, "gain": g[FULL], "variant": variant,
                                      "note": "self-sustained = recurrent activity that persists without external input; not memory",
                                      "run": runinfo.collect()})


STEPS = {"freeze-baseline": step_freeze_baseline, "profile-numpy": step_profile_numpy, "check-ts": step_check_ts,
         "check-ed": step_check_ed, "baseline-rss": step_baseline_rss, "bench": step_bench,
         "profile-compiled": step_profile_compiled, "equivalence": step_equivalence, "long-equivalence": step_long_equivalence,
         "full-build": step_full_build, "full-estimate": step_full_estimate, "full-calibrate": step_full_calibrate,
         "full-regime": step_full_regime, "full-smoke": step_full_smoke, "full-probe": step_full_probe}


def run_step(name: str) -> int:
    if name in ("summary", "figures"):
        from . import report25

        print(getattr(report25, name)())
        return 0
    if name not in STEPS:
        raise SystemExit(f"unknown step {name!r}; steps: {', '.join(STEPS)}")
    STEPS[name]()
    return 0
