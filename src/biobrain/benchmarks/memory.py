"""Memory / time benchmark of graph representations (feeds docs/MEMORY.md).

Every case runs in a fresh subprocess, so RSS belongs to that representation alone. Per case:
- `structure_bytes`: exact size of the representation's buffers (tracemalloc for Python objects);
- `rss_after_build_bytes`: RSS growth right after building it (source arrays are read without memory mapping
  and freed, so file-backed pages and temporaries do not inflate the number);
- `rss_after_access_bytes` / `peak_rss_growth_bytes`: after, and at the worst point of, build + access tests;
- access costs: random neighbour lookups, one propagation pass (1 % of neurons spike and their outgoing
  synapses are delivered), and a full scan.
Python-object cases are built on 1,000,000 edges and extrapolated linearly — they are marked as such.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CASES = ("baseline", "python_slots_objects", "python_dict_of_dicts", "arrow_raw_rows", "coo_int64", "coo_compact",
         "csr", "csr_csc", "store_mmap", "store_eager", "delta_varint", "zstd_csr", "weights_quantized", "region_local_ids")
SAMPLE_EDGES = 1_000_000


def _rss() -> int:
    import psutil

    return psutil.Process().memory_info().rss


def _peak() -> int:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _timed(fn, repeat: int = 1):
    times, result = [], None
    for _ in range(repeat):
        t = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t)
    return result, float(np.median(times))


def _csr_gather(indptr, indices, weights, rows):
    lo = indptr[rows]
    counts = indptr[rows + 1] - lo
    idx = np.repeat(lo - (np.cumsum(counts) - counts), counts) + np.arange(int(counts.sum()))
    return indices[idx], weights[idx]


def worker(case: str, dataset: str, seed: int) -> dict:
    from ..connectome.store import Connectome

    conn = Connectome.load(dataset)  # memory-mapped, untouched
    n, e = conn.n_neurons, conn.n_edges
    rng = np.random.default_rng(seed)
    active = np.sort(rng.choice(n, size=n // 100, replace=False))
    queries = rng.integers(0, n, 10_000)

    def eager(name: str) -> np.ndarray:
        return np.load(conn.root / conn.manifest["arrays"][name]["file"])

    def csr_access(indptr, indices, weights) -> dict:
        def lookups():
            for i in queries:
                indices[indptr[i]:indptr[i + 1]], weights[indptr[i]:indptr[i + 1]]
        _, t_lookup = _timed(lookups)
        (targets, w), t_gather = _timed(lambda: _csr_gather(indptr, indices, weights, active), 5)
        _, t_deliver = _timed(lambda: np.bincount(targets, weights=w, minlength=n), 5)
        _, t_scan = _timed(lambda: np.bincount(indices, weights=weights, minlength=n), 3)
        return {"lookup_us": 1e6 * t_lookup / queries.size, "propagate_ms": 1e3 * (t_gather + t_deliver),
                "synaptic_events_per_pass": int(targets.size), "full_scan_ms": 1e3 * t_scan}

    gc.collect()
    base_rss, base_peak = _rss(), _peak()
    res: dict = {"case": case, "baseline_rss_bytes": base_rss}
    keep: list = []  # the representation, alive while RSS is measured
    access = None
    structure: int | None = 0
    t0 = time.perf_counter()

    if case in ("python_slots_objects", "python_dict_of_dicts"):
        import tracemalloc

        indptr = eager("out_indptr")
        src = np.repeat(np.arange(n), np.diff(indptr))[:SAMPLE_EDGES].tolist()
        dst = eager("out_indices")[:SAMPLE_EDGES].tolist()
        w = eager("syn_count")[:SAMPLE_EDGES].tolist()
        del indptr
        tracemalloc.start()
        if case == "python_slots_objects":
            class Edge:
                __slots__ = ("pre", "post", "weight")

                def __init__(self, pre, post, weight):
                    self.pre, self.post, self.weight = pre, post, weight

            edges = [Edge(a + 1_000_000, b + 1_000_000, c) for a, b, c in zip(src, dst, w)]  # ids > 256: no small-int cache
        else:
            edges = {}
            for a, b, c in zip(src, dst, w):
                edges.setdefault(a + 1_000_000, {})[b + 1_000_000] = {"weight": c}
        current, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del src, dst, w
        keep.append(edges)
        structure = int(current * e / SAMPLE_EDGES)
        res["extrapolated_from_edges"] = SAMPLE_EDGES
        if case == "python_dict_of_dicts":
            keys = list(edges)[:10_000]

            def access():
                _, t = _timed(lambda: [[(p, d["weight"]) for p, d in edges[k].items()] for k in keys])
                return {"lookup_us": 1e6 * t / len(keys)}
    elif case == "arrow_raw_rows":
        import pyarrow as pa
        import pyarrow.feather as feather

        from ..connectome.catalog import load_catalog

        before = pa.total_allocated_bytes()
        table = feather.read_table(load_catalog(dataset).local_path("connections"))
        structure = pa.total_allocated_bytes() - before
        keep.append(table)
        res["rows"] = table.num_rows
    elif case in ("coo_int64", "coo_compact"):
        wide = case == "coo_int64"
        indptr = eager("out_indptr")
        pre = np.repeat(np.arange(n, dtype=np.int64 if wide else np.int32), np.diff(indptr))
        del indptr
        post = eager("out_indices").astype(np.int64 if wide else np.int32)
        weight = eager("syn_count").astype(np.int64 if wide else np.uint16)
        keep += [pre, post, weight]
        structure = pre.nbytes + post.nbytes + weight.nbytes

        def access():
            mask = np.zeros(n, dtype=bool)
            mask[active] = True
            def propagate():
                sel = mask[pre]
                return np.bincount(post[sel], weights=weight[sel], minlength=n)
            _, t_prop = _timed(propagate, 3)
            _, t_lookup = _timed(lambda: [post[pre == i] for i in queries[:20]])
            _, t_scan = _timed(lambda: np.bincount(post, weights=weight, minlength=n), 3)
            return {"lookup_us": 1e6 * t_lookup / 20, "propagate_ms": 1e3 * t_prop, "full_scan_ms": 1e3 * t_scan,
                    "note": "unsorted COO: a neighbour lookup scans all edges"}
    elif case in ("csr", "csr_csc"):
        indptr, indices, weights = eager("out_indptr"), eager("out_indices"), eager("syn_count")
        keep += [indptr, indices, weights]
        structure = indptr.nbytes + indices.nbytes + weights.nbytes
        if case == "csr_csc":
            in_indptr, in_indices, in_edge = eager("in_indptr"), eager("in_indices"), eager("in_edge")
            keep += [in_indptr, in_indices, in_edge]
            structure += in_indptr.nbytes + in_indices.nbytes + in_edge.nbytes

        def access():
            out = csr_access(indptr, indices, weights)
            if case == "csr_csc":
                def in_lookups():
                    for i in queries:
                        s = slice(in_indptr[i], in_indptr[i + 1])
                        in_indices[s], weights[in_edge[s]]
                out["in_lookup_us"] = 1e6 * _timed(in_lookups)[1] / queries.size
            return out
    elif case == "store_mmap":
        structure = sum(conn.manifest["arrays"][k]["bytes"] for k in conn.arrays)

        def access():
            return csr_access(conn["out_indptr"], conn["out_indices"], conn["syn_count"])
    elif case == "store_eager":
        loaded = Connectome.load(dataset, mmap=False)
        keep.append(loaded)
        structure = sum(a.nbytes for a in loaded.arrays.values())

        def access():
            return csr_access(loaded["out_indptr"], loaded["out_indices"], loaded["syn_count"])
    elif case == "delta_varint":
        from .varint import DeltaVarintCSR

        indices, weights = eager("out_indices"), eager("syn_count")
        enc = DeltaVarintCSR(eager("out_indptr"), indices, weights)
        del indices, weights
        keep.append(enc)
        structure = enc.nbytes
        res["bytes_index_stream"] = int(enc.idx_stream.nbytes)
        res["bytes_weight_stream"] = int(enc.w_stream.nbytes)

        def access():
            _, t_lookup = _timed(lambda: [enc.rows(np.array([i])) for i in queries[:2000]])
            (targets, w), t_decode = _timed(lambda: enc.rows(active), 5)
            _, t_deliver = _timed(lambda: np.bincount(targets, weights=w.astype(np.float64), minlength=n), 5)
            _, t_scan = _timed(lambda: enc.rows(np.arange(n)))
            return {"lookup_us": 1e6 * t_lookup / 2000, "propagate_ms": 1e3 * (t_decode + t_deliver),
                    "synaptic_events_per_pass": int(targets.size), "full_scan_ms": 1e3 * t_scan}
    elif case == "zstd_csr":
        import pyarrow as pa

        sizes, compressed = {}, {}
        for k in ("out_indptr", "out_indices", "syn_count"):
            array = eager(k)
            sizes[k] = array.nbytes
            compressed[k] = pa.compress(array.tobytes(), codec="zstd", asbytes=True)
            del array
        keep.append(compressed)
        structure = sum(len(c) for c in compressed.values())
        res["uncompressed_bytes"] = int(sum(sizes.values()))

        def access():
            _, t = _timed(lambda: {k: pa.decompress(c, decompressed_size=sizes[k], codec="zstd", asbytes=True)
                                   for k, c in compressed.items()}, 3)
            return {"full_decompress_ms": 1e3 * t, "note": "whole-array compression: no random access without chunking"}
    elif case == "weights_quantized":
        structure = None
        w = eager("syn_count").astype(np.int64)
        total = w.sum()
        u8 = np.minimum(w, 255)
        log_levels = np.round(np.log2(w) * 8).astype(np.int64)  # 1/8-octave log bins
        log_back = np.round(2 ** (log_levels / 8)).astype(np.int64)
        f16 = np.minimum(w, 65504).astype(np.float16).astype(np.int64)  # float16 max is 65504
        res["variants"] = {
            "uint32": {"bytes_per_edge": 4, "lossless": True},
            "uint16": {"bytes_per_edge": 2, "lossless": bool(w.max() <= 65535), "max_count": int(w.max())},
            "uint8_saturating": {"bytes_per_edge": 1, "edges_clipped": int(np.sum(w > 255)),
                                 "synapse_mass_lost_fraction": float((total - u8.sum()) / total)},
            "uint8_log_1/8_octave": {"bytes_per_edge": 1, "max_level": int(log_levels.max()),
                                     "edges_changed": int(np.sum(log_back != w)),
                                     "max_relative_error": float(np.max(np.abs(log_back - w) / w)),
                                     "synapse_mass_error_fraction": float((log_back.sum() - total) / total)},
            "float16": {"bytes_per_edge": 2, "edges_changed": int(np.sum(f16 != w))},
            "float32": {"bytes_per_edge": 4, "lossless_for_integers_below": 2 ** 24},
        }
        del w, u8, log_levels, log_back, f16
    elif case == "region_local_ids":
        from ..analysis.regions import home_blocks

        blocks, _ = home_blocks(conn)
        indptr = eager("out_indptr")
        intra = int(np.sum(blocks[np.repeat(np.arange(n), np.diff(indptr))] == blocks[eager("out_indices")]))
        sizes = np.bincount(blocks)
        structure = intra * 2 + (e - intra) * 4 + 2 * e + (n + 1) * 8 + n * 4 + n * 4
        res.update({"size_only": True, "intra_block_edges": intra, "intra_block_fraction": intra / e,
                    "largest_block_neurons": int(sizes.max()), "uint16_local_ids_possible": bool(sizes.max() <= 65536),
                    "layout": "per row: intra-block targets as uint16 local ids, then inter-block targets as int32 "
                              "global ids; + int64 indptr, int32 split pointer per row, int32 local-id map; uint16 weights"})
        del indptr, blocks
    elif case != "baseline":
        raise ValueError(case)

    res["build_s"] = time.perf_counter() - t0
    gc.collect()
    res["rss_after_build_bytes"] = _rss() - base_rss
    res["access"] = access() if access else {}
    gc.collect()
    res["rss_after_access_bytes"] = _rss() - base_rss
    if case == "store_mmap":
        for array in conn.arrays.values():
            np.asarray(array).sum()  # touches every page of every mapping
        res["rss_after_touching_everything_bytes"] = _rss() - base_rss
    res["peak_rss_growth_bytes"] = _peak() - base_peak
    res["structure_bytes"] = structure
    if structure:
        res["bytes_per_neuron"] = structure / n
        res["bytes_per_edge"] = structure / e
        res["bytes_per_synapse"] = structure / conn.manifest["counts"]["synapses"]
    del keep
    return res


def run(dataset: str, out_dir: Path, cases: tuple[str, ...] = CASES, seed: int = 20260916, log=print) -> dict:
    from .. import runinfo
    from ..connectome.catalog import load_catalog
    from ..connectome.store import Connectome, processed_dir

    results = []
    for case in cases:
        t = time.monotonic()
        proc = subprocess.run([sys.executable, "-m", "biobrain.benchmarks.memory", "--worker", case, "--dataset", dataset,
                               "--seed", str(seed)], capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            results.append({"case": case, "error": proc.stderr.strip().splitlines()[-1] if proc.stderr else "failed"})
            log(f"{case}: FAILED {results[-1]['error']}")
            continue
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        r = results[-1]
        log(f"{case}: {r.get('bytes_per_edge', 0):.2f} B/edge, RSS after build +{r['rss_after_build_bytes'] / 2**20:.0f} MiB, "
            f"peak +{r['peak_rss_growth_bytes'] / 2**20:.0f} MiB ({time.monotonic() - t:.0f} s)")
    catalog = load_catalog(dataset)
    conn = Connectome.load(dataset)
    raw = {e.id: catalog.local_path(e).stat().st_size for e in catalog.files if catalog.local_path(e).exists()}
    store = processed_dir(dataset)
    out = {"dataset": dataset, "counts": conn.manifest["counts"],
           "disk": {"raw_files": raw, "raw_total": sum(raw.values()),
                    "processed_total": sum(p.stat().st_size for p in store.rglob("*") if p.is_file()),
                    "processed_arrays": {k: v["bytes"] for k, v in conn.manifest["arrays"].items()}},
           "results": results, "run": runinfo.collect(seed=seed)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memory_benchmark.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=CASES)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    print(json.dumps(worker(args.worker, args.dataset, args.seed), default=float))
