"""Micro-benchmark behind the event-queue design (docs/SIMULATOR.md): grouping a batch of (target, weight) events
into per-neuron input, as one event-driven step must. Writes results/milestone2/benchmarks/aggregation_microbench.json.

    .venv/bin/python tools/aggregation_microbench.py
"""

import json
import time

import numpy as np

from biobrain import paths, runinfo


def bench(fn, reps):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1e6


def main():
    rng = np.random.default_rng(0)
    rows = []
    for n in (1_000, 10_000, 50_000, 139_255):
        acc = np.zeros(n, np.float32)
        for e in (100, 1_000, 10_000, 100_000, 1_000_000):
            if e > 30 * n:
                continue
            tg = rng.integers(0, n, e).astype(np.int32)
            w = rng.random(e).astype(np.float32)
            reps = max(3, int(2e5 // e))

            def unique_bincount():
                u, inv = np.unique(tg, return_inverse=True)
                return u, np.bincount(inv, weights=w)

            def sort_reduceat():
                o = np.argsort(tg, kind="stable")
                s = tg[o]
                b = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
                return s[b], np.add.reduceat(w[o], b)

            def add_at_unique():
                np.add.at(acc, tg, w)
                u = np.unique(tg)
                x = acc[u].copy()
                acc[u] = 0
                return u, x

            def dense_bincount():
                x = np.bincount(tg, weights=w, minlength=n)
                u = np.flatnonzero(x)
                return u, x[u]

            timings = {f.__name__: bench(f, reps) for f in (unique_bincount, sort_reduceat, add_at_unique, dense_bincount)}
            rows.append({"neurons": n, "events": e, "microseconds": timings, "fastest": min(timings, key=timings.get)})
            print(rows[-1])
    out = paths.results_dir() / "milestone2" / "benchmarks" / "aggregation_microbench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "run": runinfo.collect()}, indent=1) + "\n")


if __name__ == "__main__":
    main()
