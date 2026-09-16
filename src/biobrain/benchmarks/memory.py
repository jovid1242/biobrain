"""Memory / time benchmark of graph representations (feeds docs/MEMORY.md).

Every case runs in a fresh subprocess, so RSS and peak RSS belong to that representation alone.
Each case reports exact structure bytes where they can be computed, RSS growth, build time, and three
access costs: random neighbour lookups, one propagation pass (1 % of neurons spike, their outgoing
synapses are delivered), and a full scan. Python-object cases are built on 1,000,000 edges and
extrapolated linearly — they are marked as such.
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

    conn = Connectome.load(dataset)
    n, e = conn.n_neurons, conn.n_edges
    rng = np.random.default_rng(seed)
    active = np.sort(rng.choice(n, size=n // 100, replace=False))
    queries = rng.integers(0, n, 10_000)
    gc.collect()
    base_rss, base_peak = _rss(), _peak()
    res: dict = {"case": case}
    access: dict = {}

    def csr_access(indptr, indices, weights):
        def lookups():
            for i in queries:
                indices[indptr[i]:indptr[i + 1]], weights[indptr[i]:indptr[i + 1]]
        _, t_lookup = _timed(lookups)
        (tg, wg), t_prop = _timed(lambda: _csr_gather(indptr, indices, weights, active), 5)
        _, t_deliver = _timed(lambda: np.bincount(tg, weights=wg, minlength=n), 5)
        _, t_scan = _timed(lambda: np.bincount(indices, weights=weights, minlength=n), 3)
        return {"lookup_us": 1e6 * t_lookup / queries.size, "propagate_ms": 1e3 * (t_prop + t_deliver),
                "synaptic_events_per_pass": int(tg.size), "full_scan_ms": 1e3 * t_scan}

    t0 = time.perf_counter()
    if case == "baseline":
        structure = 0
    elif case in ("python_slots_objects", "python_dict_of_dicts"):
        import tracemalloc

        src = conn.edge_sources()[:SAMPLE_EDGES].tolist()
        dst = conn["out_indices"][:SAMPLE_EDGES].tolist()
        w = conn["syn_count"][:SAMPLE_EDGES].tolist()
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
        structure = current * e / SAMPLE_EDGES
        res["extrapolated_from_edges"] = SAMPLE_EDGES
        if case == "python_dict_of_dicts":
            keys = list(edges)
            def lookups():
                for k in keys[:10_000]:
                    [(p, d["weight"]) for p, d in edges[k].items()]
            _, t = _timed(lookups)
            access["lookup_us"] = 1e6 * t / min(10_000, len(keys))
    elif case == "arrow_raw_rows":
        import pyarrow as pa
        import pyarrow.feather as feather

        from ..connectome.catalog import load_catalog

        before = pa.total_allocated_bytes()
        table = feather.read_table(load_catalog(dataset).local_path("connections"))
        structure = pa.total_allocated_bytes() - before
        res["rows"] = table.num_rows
    elif case in ("coo_int64", "coo_compact"):
        wide = case == "coo_int64"
        pre = conn.edge_sources().astype(np.int64 if wide else np.int32)
        post = np.array(conn["out_indices"], dtype=np.int64 if wide else np.int32)
        weight = np.array(conn["syn_count"], dtype=np.int64 if wide else np.uint16)
        structure = pre.nbytes + post.nbytes + weight.nbytes
        mask = np.zeros(n, dtype=bool)
        mask[active] = True
        def propagate():
            sel = mask[pre]
            return np.bincount(post[sel], weights=weight[sel], minlength=n)
        _, t_prop = _timed(propagate, 3)
        _, t_lookup = _timed(lambda: [post[pre == i] for i in queries[:20]])
        _, t_scan = _timed(lambda: np.bincount(post, weights=weight, minlength=n), 3)
        access = {"lookup_us": 1e6 * t_lookup / 20, "propagate_ms": 1e3 * t_prop, "full_scan_ms": 1e3 * t_scan,
                  "note": "unsorted COO: a neighbour lookup scans all edges"}
    elif case in ("csr", "csr_csc"):
        indptr = np.array(conn["out_indptr"])
        indices = np.array(conn["out_indices"])
        weights = np.array(conn["syn_count"])
        structure = indptr.nbytes + indices.nbytes + weights.nbytes
        if case == "csr_csc":
            in_indptr, in_indices, in_edge = (np.array(conn[k]) for k in ("in_indptr", "in_indices", "in_edge"))
            structure += in_indptr.nbytes + in_indices.nbytes + in_edge.nbytes
            def in_lookups():
                for i in queries:
                    s = slice(in_indptr[i], in_indptr[i + 1])
                    in_indices[s], weights[in_edge[s]]
            _, t = _timed(in_lookups)
            access["in_lookup_us"] = 1e6 * t / queries.size
        access.update(csr_access(indptr, indices, weights))
    elif case == "store_mmap":
        structure = sum(conn.manifest["arrays"][k]["bytes"] for k in conn.arrays)
        res["rss_after_open"] = _rss() - base_rss
        access = csr_access(conn["out_indptr"], conn["out_indices"], conn["syn_count"])
        res["rss_after_csr_access"] = _rss() - base_rss
        for array in conn.arrays.values():
            np.asarray(array).sum()  # touches every page of the mapping
        res["rss_after_touching_everything"] = _rss() - base_rss
    elif case == "store_eager":
        eager = Connectome.load(dataset, mmap=False)
        structure = sum(a.nbytes for a in eager.arrays.values())
        access = csr_access(eager["out_indptr"], eager["out_indices"], eager["syn_count"])
    elif case == "delta_varint":
        from .varint import DeltaVarintCSR

        indptr, indices, weights = (np.array(conn[k]) for k in ("out_indptr", "out_indices", "syn_count"))
        enc, t_build = _timed(lambda: DeltaVarintCSR(indptr, indices, weights))
        del indices, weights
        gc.collect()
        structure = enc.nbytes
        res["encode_s"] = t_build
        _, t_lookup = _timed(lambda: [enc.rows(np.array([i])) for i in queries[:2000]])
        (tg, wg), t_prop = _timed(lambda: enc.rows(active), 5)
        _, t_deliver = _timed(lambda: np.bincount(tg, weights=wg.astype(np.float64), minlength=n), 5)
        _, t_scan = _timed(lambda: enc.rows(np.arange(n)))
        access = {"lookup_us": 1e6 * t_lookup / 2000, "propagate_ms": 1e3 * (t_prop + t_deliver),
                  "synaptic_events_per_pass": int(tg.size), "full_scan_ms": 1e3 * t_scan,
                  "bytes_index_stream": int(enc.idx_stream.nbytes), "bytes_weight_stream": int(enc.w_stream.nbytes)}
    elif case == "zstd_csr":
        import pyarrow as pa

        arrays = {k: np.array(conn[k]) for k in ("out_indptr", "out_indices", "syn_count")}
        compressed = {k: pa.compress(a.tobytes(), codec="zstd", asbytes=True) for k, a in arrays.items()}
        structure = sum(len(c) for c in compressed.values())
        _, t = _timed(lambda: {k: pa.decompress(c, decompressed_size=arrays[k].nbytes, codec="zstd", asbytes=True)
                               for k, c in compressed.items()}, 3)
        access = {"full_decompress_ms": 1e3 * t, "note": "whole-array compression: no random access without chunking"}
        res["uncompressed_bytes"] = int(sum(a.nbytes for a in arrays.values()))
    elif case == "weights_quantized":
        w = np.array(conn["syn_count"], dtype=np.int64)
        total = w.sum()
        u8 = np.minimum(w, 255)
        log_levels = np.round(np.log2(w) * 8).astype(np.int64)  # 1/8-octave log bins
        log_back = np.round(2 ** (log_levels / 8)).astype(np.int64)
        f16 = np.minimum(w, 65504).astype(np.float16).astype(np.int64)  # float16 max is 65504
        structure = 0
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
    elif case == "region_local_ids":
        from ..analysis.regions import home_blocks

        blocks, _ = home_blocks(conn)
        src_block = blocks[conn.edge_sources()]
        dst_block = blocks[np.asarray(conn["out_indices"])]
        intra = int(np.sum(src_block == dst_block))
        sizes = np.bincount(blocks)
        fits = bool(sizes.max() <= 65536)
        structure = (intra * 2 + (e - intra) * 4) + conn["syn_count"].nbytes + (n + 1) * 8 + n * 4 + n * 4
        res["size_only"] = True
        res["intra_block_edges"] = intra
        res["intra_block_fraction"] = intra / e
        res["largest_block_neurons"] = int(sizes.max())
        res["uint16_local_ids_possible"] = fits
        res["layout"] = "per row: intra-block targets as uint16 local ids, then inter-block targets as int32 global ids; " \
                        "+ int64 indptr, int32 split pointer per row, int32 local-id map; weights uint16"
    else:
        raise ValueError(case)
    res["build_s"] = time.perf_counter() - t0 if "encode_s" not in res else res["encode_s"]
    res["structure_bytes"] = int(structure)
    res["rss_growth_bytes"] = _rss() - base_rss
    res["peak_rss_growth_bytes"] = _peak() - base_peak
    res["baseline_rss_bytes"] = base_rss
    res["access"] = access
    synapses = conn.manifest["counts"]["synapses"]
    if structure:
        res["bytes_per_neuron"] = structure / n
        res["bytes_per_edge"] = structure / e
        res["bytes_per_synapse"] = structure / synapses
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
        log(f"{case}: {results[-1].get('bytes_per_edge', 0):.2f} B/edge, RSS +{results[-1]['rss_growth_bytes'] / 2**20:.0f} MiB "
            f"({time.monotonic() - t:.0f} s)")
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
