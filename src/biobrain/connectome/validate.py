"""Data validation: structural invariants, published numbers, and an independent re-derivation.

Statuses: PASS (matches), FAIL (invariant broken or unexplained mismatch), WARN (known/explained
difference or a data-quality flag), INFO (observation without an expectation). The overall result is
FAIL if any check fails. Expected numbers always carry their source; nothing is tuned to pass.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from .. import paths, runinfo
from ..telemetry import MemoryBudget, fmt_bytes, peak_rss_bytes
from .catalog import Catalog
from .preprocess import map_ids, read_annotations
from .store import Connectome, sha256_file

SOURCES = {
    "dorkenwald2024": "Dorkenwald et al. 2024, Nature 634:124-138, doi:10.1038/s41586-024-07558-y",
    "schlegel2024": "Schlegel et al. 2024, Nature 634:139-152, doi:10.1038/s41586-024-07686-5",
    "lin2024": "Lin et al. 2024, Nature 634:153-165, doi:10.1038/s41586-024-07968-y",
    "yu2025": "Yu et al. 2025, bioRxiv doi:10.1101/2025.07.11.664377 (preprint; totals for the Buhmann synapse set)",
    "zenodo": "Zenodo record 10676866 (file description; row count read from the Arrow footer by the research pass)",
    "codex_stats": "Codex neuropil_stats.csv for v783 (totals measured by the research pass; file not part of this pipeline)",
    "invariant": "structural invariant of the data model",
}

# Superclass counts as printed in Schlegel et al. 2024 (Results). NOTE: they sum to 127,864, not
# 139,255 — see the check note; they are compared, not assumed to describe release 783.
SCHLEGEL_SUPERCLASS = {"central": 32388, "optic": 77536, "visual_projection": 8053, "visual_centrifugal": 524,
                       "sensory": 5512, "ascending": 2362, "descending": 1303, "endocrine": 80, "motor": 106}


@dataclass
class Check:
    id: str
    category: str
    description: str
    status: str
    observed: object
    expected: object = None
    source: str = ""
    note: str = ""


class Report:
    def __init__(self):
        self.checks: list[Check] = []

    def add(self, id: str, category: str, description: str, observed, expected=None, *, source: str = "",
            note: str = "", status: str | None = None, tolerance: float = 0, on_mismatch: str = "FAIL") -> Check:
        if status is None:
            if expected is None:
                status = "INFO"
            elif isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
                status = "PASS" if abs(observed - expected) <= tolerance else on_mismatch
            else:
                status = "PASS" if observed == expected else on_mismatch
        check = Check(id, category, description, status, _plain(observed), _plain(expected), source, note)
        self.checks.append(check)
        return check

    @property
    def status(self) -> str:
        return "FAIL" if any(c.status == "FAIL" for c in self.checks) else "PASS"


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _present(values: np.ndarray) -> np.ndarray:
    """Non-missing mask for a stored annotation column (code 0 / NaN / -1 mean missing)."""
    if values.dtype.kind == "u":
        return values != 0
    if values.dtype.kind == "f":
        return np.isfinite(values)
    return values >= 0


def _hist(values: np.ndarray, edges: list[int]) -> dict[str, int]:
    counts = np.histogram(values, bins=[*edges, np.iinfo(np.int64).max])[0]
    labels = [f"{lo}" if hi - lo == 1 else f"{lo}-{hi - 1}" for lo, hi in zip(edges, edges[1:])] + [f">={edges[-1]}"]
    return dict(zip(labels, counts.tolist()))


def check_store(conn: Connectome, report: Report) -> None:
    n, e = conn.n_neurons, conn.n_edges
    ip, idx = conn["out_indptr"], conn["out_indices"]
    report.add("E1", "store", "out_indptr starts at 0, ends at E, non-decreasing",
               bool(ip[0] == 0 and ip[-1] == e and np.all(np.diff(ip) >= 0)), True, source="invariant")
    report.add("E2", "store", "out_indices within [0, N)", bool(idx.min() >= 0 and idx.max() < n), True, source="invariant")
    boundary = np.zeros(e, dtype=bool)
    boundary[ip[1:-1][ip[1:-1] < e]] = True
    report.add("E3", "store", "post ids strictly increasing inside every CSR row (no duplicate pairs)",
               int(np.sum((np.diff(idx) <= 0) & ~boundary[1:])), 0, source="invariant")
    in_edge = conn["in_edge"]
    perm_ok = np.bincount(in_edge, minlength=e).max() == 1 and in_edge.size == e
    post_of_slot = np.repeat(np.arange(n, dtype=np.int32), np.diff(conn["in_indptr"]))
    report.add("E4", "store", "CSC is an exact permutation of CSR (post and pre agree for every slot)",
               bool(perm_ok and np.array_equal(idx[in_edge], post_of_slot)
                    and np.array_equal(conn["in_indices"], conn.edge_sources()[in_edge])), True, source="invariant")
    row_edge = conn["row_edge"]
    per_edge = np.bincount(row_edge, weights=conn["row_syn_count"], minlength=e)
    report.add("E5", "store", "rows sorted by edge, cover every edge, and their synapses sum to the edge count",
               bool(np.all(np.diff(row_edge) >= 0) and np.unique(row_edge).size == e
                    and np.array_equal(per_edge.astype(np.int64), conn["syn_count"].astype(np.int64))), True, source="invariant")
    q_sum = conn["nt_prob_q"].sum(axis=1, dtype=np.int64)
    report.add("E6", "store", "quantized transmitter probabilities of an edge sum to 255 +/- 3 (rounding)",
               int(np.sum(np.abs(q_sum - 255) > 3)), 0, source="invariant")
    bad = [name for name, meta in conn.manifest["arrays"].items() if sha256_file(conn.root / meta["file"]) != meta["sha256"]]
    report.add("E7", "store", "every array matches its manifest sha256", bad, [], source="invariant")


def independent_totals(path: Path, budget: MemoryBudget) -> dict:
    """Re-derive pair/synapse totals with Arrow's hash aggregation (a different code path from preprocess)."""
    budget.check("validate: Arrow group_by over the connection table", 3 << 30)
    table = feather.read_table(path, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    pairs = table.group_by(["pre_pt_root_id", "post_pt_root_id"]).aggregate([("syn_count", "sum")])
    syn = pairs.column("syn_count_sum")
    strong = pairs.filter(pc.greater_equal(syn, 5))
    neurons_strong = pc.count_distinct(pa.chunked_array(strong.column("pre_pt_root_id").chunks + strong.column("post_pt_root_id").chunks))
    return {
        "rows": table.num_rows,
        "pairs": pairs.num_rows,
        "synapses": pc.sum(syn).as_py(),
        "pairs_ge_5": strong.num_rows,
        "neurons_in_pairs_ge_5": neurons_strong.as_py(),
        "self_connection_pairs": pc.sum(pc.cast(pc.equal(pairs.column("pre_pt_root_id"), pairs.column("post_pt_root_id")), pa.int64())).as_py() or 0,
    }


def run(catalog: Catalog, *, budget: MemoryBudget, out_dir: Path | None = None, log=print) -> dict:
    t0 = time.monotonic()
    out_dir = out_dir or paths.results_dir()
    conn = Connectome.load(catalog.id)
    m, obs = conn.manifest, conn.manifest["observed"]
    report = Report()
    n, e = conn.n_neurons, conn.n_edges
    lock = catalog.load_lock()

    # ---- provenance ------------------------------------------------------------------------------
    stale = [fid for fid, src in m["sources"].items() if lock.get(fid, {}).get("sha256") != src["sha256"]]
    report.add("A1", "provenance", "processed store was built from the raw files recorded in the lock file", stale, [],
               source="invariant", note="lists raw file ids whose sha256 differs between lock and manifest")

    # ---- neurons ---------------------------------------------------------------------------------
    ro = obs["root_ids"]
    report.add("B1", "neurons", "proofread neurons in release 783", ro["count"], 139255, source="dorkenwald2024",
               note='"139,255 neurons"')
    report.add("B2", "neurons", "root ids are unique", ro["unique"], ro["count"], source="invariant")

    # ---- connection table rows ---------------------------------------------------------------------
    co = obs["connections"]
    report.add("C1", "rows", "rows in proofread_connections_783 (one per pre, post, neuropil)", co["rows"], 16847997,
               source="zenodo")
    report.add("C2", "rows", "rows whose pre neuron is not a proofread neuron", co["rows_pre_not_in_root_ids"], 0,
               source="zenodo", note="the table is documented as the proofread subset")
    report.add("C3", "rows", "rows whose post neuron is not a proofread neuron", co["rows_post_not_in_root_ids"], 0, source="zenodo")
    report.add("C4", "rows", "rows with syn_count <= 0", co["rows_syn_count_le_0"], 0, source="zenodo",
               note='"one entry per neuron-neuron pair and neuropil if there is 1 or more synapses"')
    report.add("C5", "rows", "duplicate (pre, post, neuropil) rows", co["duplicate_pre_post_neuropil_rows"], 0, source="invariant")
    report.add("C6", "rows", "rows without a neuropil assignment", co["rows_unassigned_neuropil"],
               note="Dorkenwald 2024: synapses farther than 10 um from any neuropil 'were left unassigned'")
    report.add("C7", "rows", "distinct neuropil names in the connection table", co["neuropil_names_in_table"], 78,
               source="dorkenwald2024", note='"78 fly brain regions known as neuropils"')
    tr = obs["transmitters"]
    report.add("C8", "rows", "transmitter probabilities outside [0, 1] or NaN", tr["values_outside_0_1_or_nan"], 0, source="invariant")
    report.add("C9", "rows", "rows whose six mean transmitter probabilities do not sum to 1 (|dev| > 1e-3)",
               tr["rows_sum_dev_gt_1e-3"], 0, source="invariant", on_mismatch="WARN",
               note=f"max |sum - 1| = {tr['row_sum_max_abs_dev_from_1']:.2e}")

    # ---- aggregated graph --------------------------------------------------------------------------
    ed = obs["edges"]
    syn = conn["syn_count"]
    report.add("D1", "graph", "connected neuron pairs (no threshold)", ed["pairs"], 15091983, source="yu2025",
               note='also Schlegel 2024: "139,255 nodes and around 15.1 million weighted edges"')
    report.add("D2", "graph", "synapses between proofread neurons", ed["synapses"], 54492922, source="yu2025",
               note='also Dorkenwald 2024: "54.5 million synapses between these neurons"')
    report.add("D3", "graph", "neuron pairs with >= 5 synapses", ed["pairs_ge_5"], 2700513, source="dorkenwald2024",
               on_mismatch="WARN", note='"2,700,513 such connections"; Lin et al. 2024 states 2,701,601 for v783')
    strong = syn >= 5
    in_strong = np.zeros(n, dtype=bool)
    in_strong[conn.edge_sources()[strong]] = True
    in_strong[conn["out_indices"][strong]] = True
    report.add("D4", "graph", "neurons taking part in >= 5-synapse pairs", int(in_strong.sum()), 134181,
               source="dorkenwald2024", on_mismatch="WARN", note='"between 134,181 identified neurons"')
    report.add("D5", "graph", "self-connection pairs (pre == post) kept in the store", ed["self_connection_pairs"],
               note=f"{ed['self_connection_synapses']:,} synapses; not described in the sources; graph statistics drop them")
    deg = conn.out_degree() + conn.in_degree()
    report.add("D6", "graph", "neurons with no connection at all (isolated)", int(np.sum(deg == 0)))
    report.add("D7", "graph", "neurons with no outgoing / no incoming connection",
               {"no_out": int(np.sum(conn.out_degree() == 0)), "no_in": int(np.sum(conn.in_degree() == 0))})
    ind = independent_totals(catalog.local_path("connections"), budget)
    mine = {"rows": co["rows"], "pairs": ed["pairs"], "synapses": ed["synapses"], "pairs_ge_5": ed["pairs_ge_5"],
            "neurons_in_pairs_ge_5": int(in_strong.sum()), "self_connection_pairs": ed["self_connection_pairs"]}
    report.add("D8", "graph", "independent re-derivation from raw (Arrow group_by) equals the processed store", ind, mine,
               source="invariant", note="different code path: hash aggregation on root ids, no index mapping")
    check_store(conn, report)
    log(f"store + graph checks done ({time.monotonic() - t0:.0f} s)")

    # ---- synapse totals and completeness ----------------------------------------------------------------
    for side, graph_syn in (("pre", conn.out_synapses()), ("post", conn.in_synapses())):
        o = obs[f"neuropil_counts_{side}"]
        k = "F1" if side == "pre" else "F2"
        report.add(f"{k}a", "completeness", f"{side}-side synapse counts summed over all segments (incl. fragments)",
                   o["synapses_all_segments"], 130054535, source="zenodo", on_mismatch="WARN",
                   note="130,054,535 = rows of flywire_synapses_783 (research pass); Dorkenwald: '~130 million synapses'")
        report.add(f"{k}b", "completeness", f"{side}-side synapses assigned to one of the neuropils (all segments)",
                   o["synapses_all_segments"] - o["synapses_unassigned_neuropil_all_segments"], 130038118,
                   source="codex_stats", on_mismatch="WARN")
        expected = 121904312 if side == "pre" else 58096025
        report.add(f"{k}c", "completeness", f"synapses whose {side}synaptic side lies on a proofread neuron",
                   o["synapses_proofread_neurons"], expected, source="codex_stats", on_mismatch="WARN",
                   note="attachment rates 93.7 % pre / 44.7 % post (Dorkenwald 2024)")
        ip = conn[f"np_{side}_indptr"]
        cum = np.concatenate([[0], np.cumsum(conn[f"np_{side}_count"], dtype=np.int64)])
        totals = cum[ip[1:]] - cum[ip[:-1]]
        report.add(f"{k}d", "completeness",
                   f"neurons whose {side}-side synapses inside the proofread graph exceed their all-partner total",
                   int(np.sum(graph_syn > totals)), 0, source="invariant")
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(totals > 0, graph_syn / totals, np.nan)
        report.add(f"{k}e", "completeness",
                   f"share of a neuron's {side}-side synapses whose partner is a proofread neuron (percentiles)",
                   {f"p{p}": round(float(np.nanpercentile(frac, p)), 4) for p in (5, 25, 50, 75, 95)},
                   note="the remainder goes to unproofread fragments; this is data the graph does not contain")
        report.add(f"{k}f", "completeness", f"duplicate (neuron, neuropil) rows in the {side} count file",
                   o["duplicate_neuron_neuropil_rows"], 0, source="invariant")

    # ---- annotations: the version used for analysis -----------------------------------------------
    an = obs["annotations"]
    report.add("G1", "annotations", f"annotation rows ({m['annotation_version']}) vs proofread neurons", an["rows"], n,
               on_mismatch="WARN", note=f"{an['neurons_without_row']} neurons have no row; see G1b")
    report.add("G2", "annotations", "annotation rows whose root id is not a proofread neuron", an["rows_not_in_root_ids"], 0,
               source="invariant")
    report.add("G3", "annotations", "neurons with more than one annotation row", an["duplicate_rows_same_neuron"], 0,
               source="invariant")
    has_row = conn["ann_has_row"]
    columns = [name.removeprefix("ann_") for name in m["arrays"] if name.startswith("ann_") and name != "ann_has_row"]
    report.add("G4", "annotations", "fraction of neurons with a non-empty value, per column",
               {c: round(float(np.mean(_present(conn[f"ann_{c}"]))), 4) for c in columns})
    super_class = conn.labels("super_class")
    values, counts = np.unique(super_class, return_counts=True)
    report.add("G5", "annotations", f"super_class counts ({m['annotation_version']})", dict(zip(values.tolist(), counts.tolist())))

    paper_path = catalog.raw_dir / "annotations" / "v2.1.0" / "Supplemental_file1_neuron_annotations.tsv"
    if paper_path.is_file():
        paper = read_annotations(paper_path)
        pids = paper.column("root_id").to_numpy()
        pos, hit = map_ids(conn["root_id"], pids)
        report.add("H1", "annotations-paper", "v2.1.0 rows / rows that are proofread neurons / unique",
                   [paper.num_rows, int(hit.sum()), int(np.unique(pids).size)], [n, n, n], source="invariant")
        p_super = np.array([s or "" for s in paper.column("super_class").to_pylist()], dtype=object)
        pv, pc_ = np.unique(p_super.astype(str), return_counts=True)
        paper_counts = dict(zip(pv.tolist(), pc_.tolist()))
        report.add("H2", "annotations-paper", "v2.1.0 super_class counts vs the counts printed in Schlegel 2024",
                   {k: paper_counts.get(k, 0) for k in SCHLEGEL_SUPERCLASS}, SCHLEGEL_SUPERCLASS, source="schlegel2024",
                   on_mismatch="WARN",
                   note="the printed counts sum to 127,864 (release 783 has 139,255; release 630 had 127,978), "
                        f"so they most likely describe an earlier snapshot — inference, not stated. All v2.1.0 values: {paper_counts}")
        cell_type = np.array(paper.column("cell_type").to_pylist(), dtype=object)
        typed = np.array([bool(t) for t in cell_type])
        report.add("H3", "annotations-paper", "v2.1.0 distinct cell_type values", int(np.unique(cell_type[typed].astype(str)).size),
                   8453, source="schlegel2024", on_mismatch="WARN", note='"8,453 annotated cell types"')
        report.add("H4", "annotations-paper", "v2.1.0 share of neurons with a cell_type", round(float(typed.mean()), 4), 0.964,
                   source="schlegel2024", tolerance=0.0005, on_mismatch="WARN", note='"cell types for 96.4% of all neurons"')
        missing = conn["root_id"][~has_row]
        in_paper = np.isin(pids, missing)
        report.add("G1b", "annotations", f"neurons without a {m['annotation_version']} row, as annotated in v2.1.0",
                   [{"root_id": int(r), "super_class": s, "cell_type": t} for r, s, t in
                    zip(pids[in_paper], p_super[in_paper], cell_type[in_paper])])

    # ---- distributions ---------------------------------------------------------------------------
    edges = [1, 2, 3, 4, 5, 10, 20, 50, 100, 1000]
    report.add("I1", "distributions", "synapses per connected pair (histogram)", _hist(syn, edges))
    report.add("I2", "distributions", "synapses per pair: mean / median / p99 / max",
               {"mean": round(float(syn.mean()), 3), "median": float(np.median(syn)), "p99": float(np.percentile(syn, 99)),
                "max": int(syn.max())})
    per_np = np.bincount(conn["row_neuropil"], weights=conn["row_syn_count"], minlength=len(conn.vocab("neuropil")))
    names = conn.vocab("neuropil")
    top = np.argsort(per_np)[::-1][:10]
    report.add("I3", "distributions", "synapses between proofread neurons per neuropil (top 10; '' = unassigned)",
               {names[i]: int(per_np[i]) for i in top})

    result = {
        "status": report.status,
        "dataset": catalog.id,
        "counts": {"neurons": n, "edges": e, "rows": m["counts"]["rows"], "synapses": m["counts"]["synapses"]},
        "summary": {s: sum(c.status == s for c in report.checks) for s in ("PASS", "FAIL", "WARN", "INFO")},
        "sources": SOURCES,
        "checks": [asdict(c) for c in report.checks],
        "run": {**runinfo.collect(), "seconds": round(time.monotonic() - t0, 1), "peak_rss": peak_rss_bytes()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_validation.json").write_text(json.dumps(result, indent=1) + "\n")
    (out_dir / "data_validation.md").write_text(render_markdown(result))
    log(f"validation {result['status']}: {result['summary']} -> {out_dir / 'data_validation.md'}")
    return result


def render_markdown(result: dict) -> str:
    run = result["run"]
    lines = [
        f"# Data validation — {result['dataset']}",
        "",
        f"**Overall: {result['status']}** — {result['summary']}",
        "",
        f"Generated {run['timestamp_utc']} · commit `{run['git_commit'] or 'n/a'}`{' (dirty)' if run['git_dirty'] else ''} · "
        f"{run['cpu']} · {run['seconds']} s · peak RSS {fmt_bytes(run['peak_rss'])}",
        "",
        "Statuses: PASS = matches; FAIL = invariant broken or unexplained mismatch; WARN = known/explained difference "
        "or quality flag; INFO = observation.",
        "",
        "| counts | value |", "|---|---:|",
        *[f"| {k} | {v:,} |" for k, v in result["counts"].items()],
    ]
    category = None
    for c in result["checks"]:
        if c["category"] != category:
            category = c["category"]
            lines += ["", f"## {category}", "", "| id | status | check | observed | expected | source / note |", "|---|---|---|---|---|---|"]
        note = "; ".join(x for x in (result["sources"].get(c["source"], c["source"]) if c["source"] != "invariant" else "", c["note"]) if x)
        lines.append(f"| {c['id']} | {c['status']} | {c['description']} | {_cell(c['observed'])} | "
                     f"{_cell(c['expected']) if c['expected'] is not None else ''} | {note} |")
    return "\n".join(lines) + "\n"


def _cell(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    text = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    return text.replace("|", "\\|")
