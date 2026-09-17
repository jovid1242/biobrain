"""Milestone 2 experiments: provenance records, isolated benchmark runs, calibration, equivalence,
sensitivity, null controls and profiling. Raw results go to results/milestone2/ (JSON / JSON Lines).

Isolation: a network + input schedule is prepared once and cached as .npz; every benchmark run then happens in a
fresh Python process that only loads those arrays and simulates, so its peak RSS is the simulator's own.
"""

from __future__ import annotations

import datetime as dt
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .. import paths, runinfo
from ..connectome.store import Connectome, sha256_file
from . import engine, metrics, network, nulls, subgraph
from .config import SimConfig
from .inputs import InputSchedule, generate


def results_dir() -> Path:
    return paths.results_dir() / "milestone2"


def cache_dir() -> Path:
    return paths.data_dir() / "cache" / "m2"


def _rss() -> int:
    import psutil

    return psutil.Process().memory_info().rss


def _peak_rss() -> int:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


# ---- preparation ---------------------------------------------------------------------------------------------------
def build_network(conn: Connectome, subgraph_name: str, cfg: SimConfig, topology: str = "real", topology_seed: int = 0):
    sub = subgraph.load(conn, subgraph_name)
    return nulls.make(network.build(conn, sub, cfg), topology, topology_seed), sub


def _savez(path: Path, **arrays) -> None:
    cache_dir().mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def _meta_bytes(meta: dict) -> np.ndarray:
    return np.frombuffer(json.dumps(meta, default=float).encode(), dtype=np.uint8)


def prepare(conn: Connectome, subgraph_name: str, cfg: SimConfig, topology: str = "real", topology_seed: int = 0) -> Path:
    """Cache the network (per model) and the input schedule (per stimulus) separately; returns a small descriptor."""
    manifest_path = subgraph.manifests_dir(subgraph_name) / f"{subgraph_name}.json"
    sub_sha = json.loads(manifest_path.read_text())["sha256"] if manifest_path.is_file() else None
    net_key = hashlib.sha256(json.dumps([subgraph_name, sub_sha, topology, topology_seed, cfg.model_hash]).encode()).hexdigest()[:20]
    net_path = cache_dir() / f"net-{subgraph_name}-{topology}-{net_key}.npz"
    if not net_path.is_file():
        net, sub = build_network(conn, subgraph_name, cfg, topology, topology_seed)
        meta = {"network": net.meta, "dataset": sub.manifest["dataset"], "topology": topology, "topology_seed": topology_seed,
                "subgraph": {k: sub.manifest[k] for k in ("name", "method", "params", "neurons", "edges", "synapses", "sha256",
                                                          "largest_strong_component", "composition")}}
        _savez(net_path, indptr=net.indptr, indices=net.indices, weights=net.weights, meta=_meta_bytes(meta))
        n = net.n
    else:
        with np.load(net_path) as data:
            n = int(data["indptr"].size - 1)
    sched_key = hashlib.sha256(json.dumps([n, cfg.stimulus_hash]).encode()).hexdigest()[:20]
    sched_path = cache_dir() / f"input-n{n}-{sched_key}.npz"
    if not sched_path.is_file():
        sched = generate(cfg.inputs, n, cfg.run.steps, cfg.run.seed)
        _savez(sched_path, indptr=sched.indptr, neurons=sched.neurons, meta=_meta_bytes({"input": sched.info, "weight": sched.weight}))
    descriptor = cache_dir() / f"run-{net_key}-{sched_key}.json"
    descriptor.write_text(json.dumps({"network": net_path.name, "input": sched_path.name}))
    return descriptor


def load_prepared(descriptor: Path) -> tuple[network.Network, InputSchedule, dict]:
    files = json.loads(Path(descriptor).read_text())
    with np.load(cache_dir() / files["network"]) as data:
        meta = json.loads(bytes(data["meta"]).decode())
        net = network.Network(int(data["indptr"].size - 1), data["indptr"], data["indices"], data["weights"], meta["network"])
    with np.load(cache_dir() / files["input"]) as data:
        imeta = json.loads(bytes(data["meta"]).decode())
        sched = InputSchedule(data["indptr"], data["neurons"], imeta["weight"], imeta["input"])
    meta["input"] = imeta["input"]
    return net, sched, meta


# ---- records ---------------------------------------------------------------------------------------------------------
def experiment_id(kind: str, subgraph_name: str, topology: str, cfg: SimConfig, backend: str = "numpy",
                  variant: str = "touched", prefix: str = "m2") -> str:
    if backend == "numpy":
        mode = cfg.run.mode + ("_auto" if cfg.run.mode == "event_driven" and cfg.run.aggregation == "auto" else "")
    else:
        mode = f"{backend}_{cfg.run.mode}" + (f"_{variant}" if cfg.run.mode == "event_driven" else "")
    return (f"{prefix}-{kind}-{subgraph_name}-{topology}-{mode}-{cfg.inputs.pattern}{cfg.inputs.rate:g}"
            f"-g{cfg.weights.gain:g}-{cfg.neuron.dtype}-s{cfg.run.seed}-{cfg.digest()[:8]}")


def make_record(kind: str, cfg: SimConfig, meta: dict, res: engine.RunResult, extra: dict | None = None,
                backend: str = "numpy", variant: str = "touched", prefix: str = "m2") -> dict:
    summary = metrics.summarize(res, cfg.neuron)
    return {
        "experiment_id": experiment_id(kind, meta["subgraph"]["name"], meta["topology"], cfg, backend, variant, prefix),
        "kind": kind, "backend": backend,
        "variant": variant if backend != "numpy" and cfg.run.mode == "event_driven" else None,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "mode": cfg.run.mode, "aggregation": cfg.run.aggregation if cfg.run.mode == "event_driven" else None,
        "seed": cfg.run.seed,
        "config": cfg.to_dict(), "config_hash": cfg.digest(), "model_hash": cfg.model_hash, "stimulus_hash": cfg.stimulus_hash,
        "dataset": meta["dataset"], "subgraph": meta["subgraph"], "topology": meta["topology"],
        "network": {k: v for k, v in meta["network"].items() if k != "signs"}, "input": meta["input"],
        "metrics": summary, "memory": res.memory, "engine_extra": {k: v for k, v in res.extra.items() if not k.startswith("_")},
        "phases": res.phases,
        "run": runinfo.collect(), **(extra or {}),
    }


def simulate(path: Path, cfg: SimConfig, kind: str = "benchmark", tag: str = "", backend: str = "numpy",
             variant: str = "touched", prefix: str = "m2") -> dict:
    from . import energy

    compile_s = None
    power_before = energy.power_state()
    if backend == "numba":  # compile or load the cached kernels before anything is measured
        from . import compiled

        compile_s = compiled.warmup(cfg.neuron.dtype, modes=(cfg.run.mode,), variants=(variant,))
    rss0 = _rss()
    peak0 = _peak_rss()
    t0 = time.perf_counter()
    net, sched, meta = load_prepared(path)
    setup_s = time.perf_counter() - t0
    rss_loaded = _rss()
    e0 = energy.snapshot()
    kw = {"variant": variant} if backend == "numba" else {}
    res = engine.run(net, sched, cfg.neuron, cfg.run.steps, cfg.run.mode, backend=backend, aggregation=cfg.run.aggregation,
                     profile=cfg.run.profile, record_voltage_every=cfg.run.record_voltage_every, **kw)
    e1 = energy.snapshot()
    gc.collect()
    process = {"baseline_rss": rss0, "peak_rss_before_load": peak0, "rss_after_load": rss_loaded, "rss_after_run": _rss(),
               "peak_rss": _peak_rss(), "setup_s": setup_s, "compile_or_cache_load_s": compile_s, **energy.power_state(),
               "power_before_run": power_before}
    record = make_record(kind, cfg, meta, res, {"process": process, "prepared": path.name,
                                                "os_cpu_energy_estimate": energy.delta(e0, e1, res.wall_s)},
                         backend=backend, variant=variant, prefix=prefix)
    if tag:
        record["experiment_id"] += f"-{tag}"
        record["tag"] = tag
    return record


def run_isolated(path: Path, cfg: SimConfig, kind: str = "benchmark", timeout: float = 3600, tag: str = "",
                 backend: str = "numpy", variant: str = "touched", prefix: str = "m2") -> dict:
    proc = subprocess.run([sys.executable, "-m", "biobrain.snn.experiments", "--prepared", str(path),
                           "--config", json.dumps(cfg.to_dict()), "--kind", kind, "--tag", tag, "--backend", backend,
                           "--variant", variant, "--prefix", prefix],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed: {proc.stderr.strip()[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def empty_process_rss(backend: str = "numpy", dtype: str = "float32") -> int:
    """Peak RSS of a worker that imports the simulator (and, for numba, loads the compiled kernels) and exits."""
    proc = subprocess.run([sys.executable, "-m", "biobrain.snn.experiments", "--baseline", "--backend", backend,
                           "--dtype", dtype], capture_output=True, text=True, timeout=600, check=True)
    return int(json.loads(proc.stdout.strip().splitlines()[-1])["peak_rss"])


def _done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {json.loads(line)["experiment_id"] for line in path.read_text().splitlines() if line.strip()}


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=float) + "\n")


def _write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=float) + "\n")
    return path


# ---- calibration -------------------------------------------------------------------------------------------------------
CALIBRATION_GAINS = tuple(float(g) for g in np.geomspace(0.003, 3.0, 19))


def calibrate(conn: Connectome, subgraph_name: str, base: SimConfig, gains=CALIBRATION_GAINS, seeds=(1, 2, 3),
              on_steps: int = 300, off_steps: int = 300, rate: float = 0.01, log=print) -> dict:
    """Gain sweep under a fixed probe input; the baseline is the geometric middle of the widest contiguous gain range
    in which every seed is STABLE. Time-step mode only (the regime does not depend on the execution mode)."""
    rows = []
    steps = on_steps + off_steps
    sub = subgraph.load(conn, subgraph_name)
    for gain in gains:
        cfg = base.replace(weights={"gain": gain}, inputs={"pattern": "poisson", "rate": rate, "on_steps": on_steps},
                           run={"steps": steps, "mode": "time_step", "record_voltage_every": 10})
        net = network.build(conn, sub, cfg)
        for seed in seeds:
            sched = generate(cfg.inputs, net.n, steps, seed)
            res = engine.run(net, sched, cfg.neuron, steps, "time_step", record_voltage_every=10)
            external_on = int(sched.indptr[on_steps])
            cls = metrics.classify(res, cfg.neuron, on_steps, external_on)
            summary = metrics.summarize(res, cfg.neuron)
            rows.append({"gain": gain, "seed": seed, **cls, "population_rate_hz": summary["population_rate_hz"],
                         "fraction_neurons_active": summary["fraction_neurons_active"],
                         "silent_neurons": summary["silent_neurons"], "refractory_occupancy": summary["refractory_occupancy"],
                         "synaptic_events": summary["synaptic_events"], "membrane_max": summary.get("membrane_max"),
                         "membrane_mean_max": summary.get("membrane_mean_max")})
        labels = [r["regime"] for r in rows if r["gain"] == gain]
        log(f"  {subgraph_name} gain {gain:.4g}: {labels}")
    stable = [g for g in gains if all(r["regime"] == "STABLE" for r in rows if r["gain"] == g)]
    runs, current = [], []
    for g in gains:
        if g in stable:
            current.append(g)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    best = max(runs, key=len) if runs else None
    baseline = float(np.sqrt(best[0] * best[-1])) if best else None
    return {"subgraph": subgraph_name, "neurons": sub.n, "edges_in_subgraph": sub.m, "config": base.to_dict(),
            "probe": {"pattern": "poisson", "rate": rate, "on_steps": on_steps, "off_steps": off_steps, "seeds": list(seeds)},
            "gains": list(gains), "rows": rows, "stable_gains": stable, "widest_stable_range": [best[0], best[-1]] if best else None,
            "baseline_gain": baseline,
            "rule": "baseline = geometric mean of the ends of the widest contiguous gain range where all seeds are STABLE",
            "regime_rules": metrics.RULES, "run": runinfo.collect()}


# ---- equivalence, sensitivity, nulls, profiling ------------------------------------------------------------------------
def equivalence(conn: Connectome, subgraph_name: str, base: SimConfig, rates, dtypes=("float64", "float32"), steps=500,
                seed=1) -> list[dict]:
    rows = []
    sub = subgraph.load(conn, subgraph_name)
    for dtype in dtypes:
        cfg = base.replace(neuron={"dtype": dtype}, run={"steps": steps, "seed": seed})
        net = network.build(conn, sub, cfg)
        for rate in rates:
            sched = generate(cfg.inputs.__class__(**{**cfg.to_dict()["inputs"], "rate": rate}), net.n, steps, seed)
            ts = engine.run(net, sched, cfg.neuron, steps, "time_step", record_spikes=True)
            for aggregation in ("sparse", "auto"):
                ed = engine.run(net, sched, cfg.neuron, steps, "event_driven", record_spikes=True, aggregation=aggregation)
                cmp = metrics.compare(ts, ed, metrics.V_TOLERANCE[dtype])
                rows.append({"subgraph": subgraph_name, "dtype": dtype, "input_rate": rate, "aggregation": aggregation,
                             "spikes": ts.counters["spikes"], "synaptic_events": ts.counters["synaptic_events"], **cmp})
    return rows


SENSITIVITY_VARIANTS = {
    "baseline": {},
    "transform=sqrt": {"weights": {"transform": "sqrt"}},
    "transform=log1p": {"weights": {"transform": "log1p"}},
    "transform=clipped(20)": {"weights": {"transform": "clipped", "clip": 20.0}},
    "glutamate=+1": {"signs": {"glutamate": 1.0}},
    "modulators=+1": {"signs": {"dopamine": 1.0, "serotonin": 1.0, "octopamine": 1.0}},
    "modulators=-1": {"signs": {"dopamine": -1.0, "serotonin": -1.0, "octopamine": -1.0}},
    "sign source=edge": {"signs": {"source": "edge"}},
    "min_synapses=5": {"weights": {"min_synapses": 5}},
    "tau_m=10ms": {"neuron": {"tau_m_ms": 10.0}},
    "tau_m=40ms": {"neuron": {"tau_m_ms": 40.0}},
    "delay=2ms": {"neuron": {"delay_ms": 2.0}},
    "delay=5ms": {"neuron": {"delay_ms": 5.0}},
    "refractory=5ms": {"neuron": {"refractory_ms": 5.0}},
    "gain x0.5": {"_gain_factor": 0.5},
    "gain x2": {"_gain_factor": 2.0},
}


def sensitivity(conn: Connectome, subgraph_name: str, base: SimConfig, seeds=(1, 2, 3), on_steps=300, off_steps=300,
                rate=0.01, log=print) -> dict:
    sub = subgraph.load(conn, subgraph_name)
    steps = on_steps + off_steps
    rows = []
    for name, change in SENSITIVITY_VARIANTS.items():
        change = dict(change)
        factor = change.pop("_gain_factor", 1.0)
        cfg = base.replace(**change) if change else base
        cfg = cfg.replace(weights={"gain": base.weights.gain * factor}, inputs={"pattern": "poisson", "rate": rate, "on_steps": on_steps},
                          run={"steps": steps, "mode": "time_step"})
        net = network.build(conn, sub, cfg)
        for seed in seeds:
            sched = generate(cfg.inputs, net.n, steps, seed)
            res = engine.run(net, sched, cfg.neuron, steps, "time_step")
            cls = metrics.classify(res, cfg.neuron, on_steps, int(sched.indptr[on_steps]))
            s = metrics.summarize(res, cfg.neuron)
            rows.append({"variant": name, "seed": seed, **cls, "population_rate_hz": s["population_rate_hz"],
                         "fraction_neurons_active": s["fraction_neurons_active"], "simulated_edges": net.m,
                         "excitatory_edges": net.meta["excitatory_edges"], "inhibitory_edges": net.meta["inhibitory_edges"],
                         "removed_zero_sign": net.meta["removed_zero_sign"]})
        log(f"  {name}: {[r['regime'] for r in rows if r['variant'] == name]}")
    return {"subgraph": subgraph_name, "baseline_config": base.to_dict(), "probe": {"rate": rate, "on_steps": on_steps,
            "off_steps": off_steps, "seeds": list(seeds)}, "rows": rows, "run": runinfo.collect()}


def null_controls(conn: Connectome, subgraph_name: str, base: SimConfig, seeds=(1, 2, 3), on_steps=300, off_steps=300,
                  rate=0.01, log=print) -> dict:
    steps = on_steps + off_steps
    cfg = base.replace(inputs={"pattern": "poisson", "rate": rate, "on_steps": on_steps}, run={"steps": steps})
    rows = []
    for topology in ("real", "degree_preserving", "reciprocity_preserving"):
        net, _ = build_network(conn, subgraph_name, cfg, topology, topology_seed=11)
        for seed in seeds:
            sched = generate(cfg.inputs, net.n, steps, seed)
            ts = engine.run(net, sched, cfg.neuron, steps, "time_step", record_spikes=True)
            ed = engine.run(net, sched, cfg.neuron, steps, "event_driven", record_spikes=True)
            cls = metrics.classify(ts, cfg.neuron, on_steps, int(sched.indptr[on_steps]))
            s = metrics.summarize(ts, cfg.neuron)
            rows.append({"topology": topology, "seed": seed, "edges": net.m, "reciprocity": nulls.reciprocity(net),
                         "topology_info": net.meta.get("topology"), **cls, "population_rate_hz": s["population_rate_hz"],
                         "equivalence": metrics.compare(ts, ed, metrics.V_TOLERANCE[cfg.neuron.dtype]),
                         "ts_wall_s": ts.wall_s, "ed_wall_s": ed.wall_s})
        log(f"  {topology}: {[r['regime'] for r in rows if r['topology'] == topology]}")
    return {"subgraph": subgraph_name, "config": cfg.to_dict(), "rows": rows,
            "note": "infrastructure check only; no conclusion about fly topology is drawn in Milestone 2", "run": runinfo.collect()}


def profile_phases(conn: Connectome, subgraph_name: str, base: SimConfig, rates, steps=500, seed=1) -> list[dict]:
    rows = []
    sub = subgraph.load(conn, subgraph_name)
    net = network.build(conn, sub, base)
    for rate in rates:
        sched = generate(base.inputs.__class__(**{**base.to_dict()["inputs"], "rate": rate}), net.n, steps, seed)
        for mode, aggregation in (("time_step", "sparse"), ("event_driven", "sparse"), ("event_driven", "auto")):
            engine.run(net, sched, base.neuron, min(steps, 50), mode, aggregation=aggregation)  # warm-up
            res = engine.run(net, sched, base.neuron, steps, mode, aggregation=aggregation, profile=True)
            total = sum(res.phases.values())
            rows.append({"subgraph": subgraph_name, "input_rate": rate, "mode": mode, "aggregation": aggregation, "steps": steps,
                         "wall_s": res.wall_s, "phases_s": res.phases, "phase_share": {k: v / total for k, v in res.phases.items()},
                         "spikes": res.counters["spikes"], "synaptic_events": res.counters["synaptic_events"],
                         "neuron_updates": res.counters["neuron_updates"]})
    return rows


# ---- benchmark matrix ---------------------------------------------------------------------------------------------------
MODES = (("time_step", "sparse"), ("event_driven", "sparse"), ("event_driven", "auto"))


def benchmark(conn: Connectome, out_name: str, subgraphs: dict[str, float], rates, seeds, steps: int, base: SimConfig,
              modes=MODES, input_changes: dict | None = None, log=print) -> Path:
    """subgraphs: name -> gain. Every run in its own process; resumable (existing experiment ids are skipped)."""
    out = results_dir() / "benchmarks" / f"{out_name}.jsonl"
    done = _done(out)
    for name, gain in subgraphs.items():
        for rate in rates:
            for seed in seeds:
                cfg = base.replace(weights={"gain": gain}, inputs={"rate": rate, **(input_changes or {})},
                                   run={"steps": steps, "seed": seed})
                path = prepare(conn, name, cfg)
                for mode, aggregation in modes:
                    run_cfg = cfg.replace(run={"mode": mode, "aggregation": aggregation})
                    eid = experiment_id("benchmark", name, "real", run_cfg)
                    if eid in done:
                        continue
                    t = time.monotonic()
                    record = run_isolated(path, run_cfg)
                    _append(out, record)
                    m = record["metrics"]
                    log(f"  {name} rate {rate:g} seed {seed} {mode}/{aggregation}: wall {m['wall_s']:.3f}s "
                        f"updates {m['update_fraction']:.3f} spikes/step {m['spike_fraction_per_step']:.4f} "
                        f"peak RSS {record['process']['peak_rss'] / 2**20:.0f} MiB ({time.monotonic() - t:.1f}s)")
    return out


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ---- worker entry point --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared")
    parser.add_argument("--config")
    parser.add_argument("--kind", default="benchmark")
    parser.add_argument("--tag", default="")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--variant", default="touched")
    parser.add_argument("--prefix", default="m2")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()
    if args.baseline:
        if args.backend == "numba":
            from . import compiled

            compiled.warmup(args.dtype, variants=compiled.VARIANTS)
        print(json.dumps({"peak_rss": _peak_rss(), "rss": _rss()}))
    else:
        print(json.dumps(simulate(Path(args.prepared), SimConfig.from_dict(json.loads(args.config)), args.kind, args.tag,
                                  args.backend, args.variant, args.prefix), default=float))
