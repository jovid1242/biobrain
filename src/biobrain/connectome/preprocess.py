"""Raw FlyWire release -> compact processed store.

Processing choices are documented in docs/DATASET.md and docs/ASSUMPTIONS.md. Everything noticed
on the way (row counts, rows outside the neuron set, duplicates, self-connections, transmitter
probability sums, ...) is kept under `observed` in the manifest; `biobrain validate` compares those
observations with published numbers and re-derives the key totals independently from the raw files.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.feather as feather

from .. import runinfo
from ..telemetry import MemoryBudget, fmt_bytes, peak_rss_bytes, rss_bytes
from .catalog import Catalog
from .store import StoreWriter, processed_dir, smallest_uint

NT_COLUMNS = ("gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg")
NT_LEVELS = 255
# The release spells "no neuropil assignment" as UNASGD (connection table) and as the string "None"
# (per-neuron count files); both are stored as the empty neuropil name (code 0).
UNASSIGNED_NEUROPIL = ("UNASGD", "None")
ANN_NUMERIC = ("pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z", "top_nt_conf")
ANN_IDS = ("supervoxel_id", "nucleus_id")
ANN_SKIP = ("root_id", "vfb_id", "fbbt_id", "matching_notes", "synonyms")  # external ids / free text
REQUIRED = ("root_ids", "connections", "neuropil_counts_pre", "neuropil_counts_post", "annotations")


class Vocab:
    """String -> dense integer code; '' (missing) is always code 0. New strings are appended.
    `missing` lists raw spellings that also mean "no value"."""

    def __init__(self, values: list[str] | None = None, missing: tuple[str, ...] = ()):
        self.index = {v: i for i, v in enumerate(values or [""])}
        self.missing = set(missing)

    @property
    def values(self) -> list[str]:
        return list(self.index)

    def encode(self, strings: pa.ChunkedArray | pa.Array) -> np.ndarray:
        if isinstance(strings, pa.ChunkedArray):
            strings = strings.combine_chunks()
        enc = pc.dictionary_encode(strings)
        values = ["" if v in self.missing else v or "" for v in enc.dictionary.to_pylist()]
        local = np.array([self.index.setdefault(v, len(self.index)) for v in values] + [0], dtype=np.int64)
        return local[pc.fill_null(enc.indices, len(values)).to_numpy()]

    def sorted(self) -> tuple["Vocab", np.ndarray]:
        """A vocab with the same strings in sorted order ('' first) and the old->new code map."""
        ordered = Vocab([""] + sorted(v for v in self.index if v), tuple(self.missing))
        remap = np.array([ordered.index[v] for v in self.index], dtype=np.int64)
        return ordered, remap


def label_counts(strings: pa.ChunkedArray | pa.Array, labels: tuple[str, ...], weights: np.ndarray | None = None) -> dict:
    """Rows (or summed weights) carrying each of `labels`."""
    out = {}
    for label in labels:
        mask = pc.fill_null(pc.equal(strings, label), False)
        mask = np.asarray(mask.to_numpy(zero_copy_only=False), dtype=bool)
        out[label] = int(mask.sum() if weights is None else weights[mask].sum())
    return out


def map_ids(sorted_ids: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions of `ids` in `sorted_ids` and a mask of which ids were found."""
    pos = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
    return pos, sorted_ids[pos] == ids


def indptr_from(sorted_rows: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(np.bincount(sorted_rows, minlength=n), dtype=np.int64)])


def read_annotations(path: Path) -> pa.Table:
    """Annotation TSV (no quoting). Integer ids are parsed from text so that ids above 2**53 stay exact,
    and releases that print them as '2453924.0' still load; a real fractional part raises."""
    header = path.open().readline().rstrip("\n").split("\t")
    types = {c: pa.string() for c in header}
    types.update({c: pa.float64() for c in ANN_NUMERIC if c in types})
    table = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                           convert_options=pacsv.ConvertOptions(column_types=types, strings_can_be_null=True))
    for column in ("root_id", *ANN_IDS):
        if column in table.column_names:
            text = pc.replace_substring_regex(table.column(column), pattern=r"\.0+$", replacement="")
            table = table.set_column(table.column_names.index(column), column, pc.cast(text, pa.int64()))
    return table


def _neuropil_counts(path: Path, id_col: str, root: np.ndarray, vocab: Vocab):
    """Per-segment neuropil synapse counts -> per-neuron CSR (proofread neurons only) + totals."""
    reader = pa.ipc.open_file(path)
    obs = dict(rows=0, synapses_all_segments=0, synapses_unassigned_neuropil_all_segments=0, rows_count_le_0=0,
               null_neuropil_rows=0, unassigned_labels=dict.fromkeys(UNASSIGNED_NEUROPIL, 0))
    parts = []
    for b in range(reader.num_record_batches):
        batch = reader.get_batch(b)
        ids = batch.column(id_col).to_numpy()
        count = batch.column("count").to_numpy()
        names = batch.column("neuropil")
        codes = vocab.encode(names)
        pos, hit = map_ids(root, ids)
        obs["rows"] += int(ids.size)
        obs["synapses_all_segments"] += int(count.sum())
        obs["synapses_unassigned_neuropil_all_segments"] += int(count[codes == 0].sum())
        obs["rows_count_le_0"] += int((count <= 0).sum())
        obs["null_neuropil_rows"] += names.null_count
        for label, synapses in label_counts(names, UNASSIGNED_NEUROPIL, count).items():
            obs["unassigned_labels"][label] += synapses
        parts.append((pos[hit].astype(np.int32), codes[hit], count[hit]))
    idx, codes, count = (np.concatenate(p) for p in zip(*parts))
    order = np.lexsort((codes, idx))
    idx, codes, count = idx[order], codes[order], count[order]
    obs["rows_proofread_neurons"] = int(idx.size)
    obs["synapses_proofread_neurons"] = int(count.sum())
    obs["synapses_proofread_neurons_assigned_neuropil"] = int(count[codes != 0].sum())
    obs["duplicate_neuron_neuropil_rows"] = int(np.sum((idx[1:] == idx[:-1]) & (codes[1:] == codes[:-1])))
    return indptr_from(idx, root.size), codes, count, obs


def run(catalog: Catalog, *, budget: MemoryBudget, log=print) -> Path:
    t0 = time.monotonic()

    def lap(msg: str) -> None:
        log(f"[{time.monotonic() - t0:6.1f} s | RSS {fmt_bytes(rss_bytes()):>10}] {msg}")

    for file_id in REQUIRED:
        if not catalog.local_path(file_id).exists():
            raise FileNotFoundError(f"{catalog.local_path(file_id)} is missing; run `biobrain download` first")
    lock = catalog.load_lock()
    writer = StoreWriter(processed_dir(catalog.id))
    observed: dict = {}

    # ---- neurons -------------------------------------------------------------------------------
    raw_ids = np.load(catalog.local_path("root_ids"))
    if raw_ids.size and int(raw_ids.max()) >= 2**63:
        raise ValueError("root id does not fit int64")
    root = np.unique(raw_ids.astype(np.int64))
    n = int(root.size)
    observed["root_ids"] = {"count": int(raw_ids.size), "unique": n, "dtype": raw_ids.dtype.str,
                            "sorted_in_file": bool(np.all(raw_ids[1:] >= raw_ids[:-1]))}
    writer.add("root_id", root, "FlyWire 783 root id of neuron i (ascending)")
    lap(f"neurons: {n:,} root ids ({observed['root_ids']['count']:,} in file)")

    # ---- connection table: one row per (pre, post, neuropil) ----------------------------------
    path = catalog.local_path("connections")
    budget.check("preprocess: connection-table columns and sort workspace", 3 << 30)
    table = feather.read_table(path, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    pre_pos, pre_ok = map_ids(root, table.column("pre_pt_root_id").to_numpy())
    post_pos, post_ok = map_ids(root, table.column("post_pt_root_id").to_numpy())
    syn = table.column("syn_count").to_numpy()
    del table
    rows_total = int(syn.size)
    raw_neuropil = Vocab(missing=UNASSIGNED_NEUROPIL)
    neuropil_column = feather.read_table(path, columns=["neuropil"]).column("neuropil")
    unassigned_rows = label_counts(neuropil_column, UNASSIGNED_NEUROPIL)
    unassigned_synapses = label_counts(neuropil_column, UNASSIGNED_NEUROPIL, syn)
    np_codes = raw_neuropil.encode(neuropil_column)
    null_neuropil_rows = neuropil_column.null_count
    del neuropil_column
    neuropils, remap = raw_neuropil.sorted()
    np_codes = remap[np_codes]
    if len(neuropils.index) > 256:
        raise ValueError("more than 255 neuropil names; widen the row sort key")
    keep = np.flatnonzero(pre_ok & post_ok & (syn > 0))
    observed["connections"] = {
        "rows": rows_total,
        "rows_pre_not_in_root_ids": int((~pre_ok).sum()),
        "rows_post_not_in_root_ids": int((~post_ok).sum()),
        "rows_syn_count_le_0": int((syn <= 0).sum()),
        "rows_dropped": rows_total - int(keep.size),
        "rows_unassigned_neuropil": int((np_codes == 0).sum()),
        "unassigned_neuropil_labels_rows": unassigned_rows,
        "unassigned_neuropil_labels_synapses": unassigned_synapses,
        "null_neuropil_rows": null_neuropil_rows,
        "neuropil_names_in_table": len(neuropils.index) - 1,
        "synapses_all_rows": int(syn.sum()),
        "syn_count_row_max": int(syn.max()),
    }
    lap(f"connection table: {rows_total:,} rows, {observed['connections']['neuropil_names_in_table']} neuropil names")

    key = pre_pos[keep] * n + post_pos[keep]
    del pre_pos, post_pos, pre_ok, post_ok
    order = np.argsort(key * 256 + np_codes[keep])
    rows = keep[order]
    key = key[order]
    del keep, order
    np_row = np_codes[rows].astype(np.uint8)
    syn_row = syn[rows]
    del np_codes, syn
    same_pair = key[1:] == key[:-1]
    observed["connections"]["duplicate_pre_post_neuropil_rows"] = int(np.sum(same_pair & (np_row[1:] == np_row[:-1])))
    starts = np.flatnonzero(np.concatenate([[True], ~same_pair]))
    del same_pair
    n_edges = int(starts.size)
    row_edge = np.repeat(np.arange(n_edges, dtype=np.int32), np.diff(np.append(starts, key.size)))
    edge_pre = (key[starts] // n).astype(np.int32)
    edge_post = (key[starts] % n).astype(np.int32)
    del key
    syn_edge = np.add.reduceat(syn_row, starts)
    observed["edges"] = {
        "pairs": n_edges,
        "synapses": int(syn_edge.sum()),
        "self_connection_pairs": int(np.sum(edge_pre == edge_post)),
        "self_connection_synapses": int(syn_edge[edge_pre == edge_post].sum()),
        "syn_count_pair_max": int(syn_edge.max()),
        "pairs_ge_5": int(np.sum(syn_edge >= 5)),
    }
    lap(f"aggregated {n_edges:,} neuron pairs, {observed['edges']['synapses']:,} synapses")

    writer.add("out_indptr", indptr_from(edge_pre, n), "CSR row pointer over edges sorted by (pre, post)")
    writer.add("out_indices", edge_post, "post neuron of edge e")
    writer.add("syn_count", syn_edge.astype(smallest_uint(int(syn_edge.max()))), "synapses of edge e, summed over neuropils")
    writer.add("row_edge", row_edge, "edge id of each (pre, post, neuropil) row; rows sorted by (edge, neuropil)")
    writer.add("row_neuropil", np_row, "neuropil code of each row (vocab 'neuropil'; 0 = unassigned)")
    writer.add("row_syn_count", syn_row.astype(smallest_uint(int(syn_row.max()))), "synapses of each row")
    del row_edge

    # transmitter probabilities: synapse-weighted mean over the pair's rows that HAVE a prediction
    # (some rows carry NaN in all six columns), quantized to uint8; an edge with no predicted row stays all zeros
    def nt_column(column: str) -> np.ndarray:
        return feather.read_table(path, columns=[column]).column(column).to_numpy()[rows]

    nan_count = np.zeros(rows.size, dtype=np.int8)
    for column in NT_COLUMNS:
        nan_count += np.isnan(nt_column(column))
    predicted = nan_count == 0
    denominator = np.add.reduceat(np.where(predicted, syn_row, 0), starts)
    has = denominator > 0
    nt = np.zeros((n_edges, len(NT_COLUMNS)), dtype=np.uint8)
    row_sum = np.zeros(rows.size)
    outside = 0
    for k, column in enumerate(NT_COLUMNS):
        p = np.where(predicted, nt_column(column), 0.0)
        outside += int(np.sum((p < 0) | (p > 1)))
        row_sum += p
        weighted = np.add.reduceat(p * syn_row, starts)
        nt[has, k] = np.rint(weighted[has] / denominator[has] * NT_LEVELS).astype(np.uint8)
    dev = np.abs(row_sum[predicted] - 1)
    predicted_rows_per_edge = np.add.reduceat(predicted.astype(np.int64), starts)
    observed["transmitters"] = {
        "rows_without_prediction": int((~predicted).sum()),
        "rows_partially_nan": int(np.sum((nan_count > 0) & (nan_count < len(NT_COLUMNS)))),
        "synapses_without_prediction": int(syn_row[~predicted].sum()),
        "edges_without_prediction": int((~has).sum()),
        "edges_partially_predicted": int(np.sum(has & (predicted_rows_per_edge < np.diff(np.append(starts, rows.size))))),
        "predicted_values_outside_0_1": outside,
        "predicted_row_sum_max_abs_dev_from_1": float(dev.max()) if dev.size else 0.0,
        "predicted_rows_sum_dev_gt_1e-3": int(np.sum(dev > 1e-3)),
    }
    writer.add("nt_prob_q", nt, f"edge transmitter probabilities {list(NT_COLUMNS)}, q = rint(p*{NT_LEVELS}); "
                                "all zeros = no synapse of the edge has a prediction")
    del nt, row_sum, dev, rows, syn_row, starts, nan_count, predicted
    lap("transmitter probabilities aggregated")

    in_edge = np.argsort(edge_post, kind="stable").astype(np.int32)
    writer.add("in_indptr", indptr_from(edge_post, n), "CSC column pointer (in-edges grouped by post)")
    writer.add("in_indices", edge_pre[in_edge], "pre neuron of each CSC slot")
    writer.add("in_edge", in_edge, "CSR edge id of each CSC slot")
    del in_edge, edge_pre, edge_post
    lap("CSR + CSC written")

    # ---- per-neuron neuropil synapse counts (all partners, incl. unproofread fragments) -------
    for side, file_id, id_col in (("pre", "neuropil_counts_pre", "pre_pt_root_id"),
                                  ("post", "neuropil_counts_post", "post_pt_root_id")):
        indptr, codes, count, obs = _neuropil_counts(catalog.local_path(file_id), id_col, root, neuropils)
        observed[f"neuropil_counts_{side}"] = obs
        writer.add(f"np_{side}_indptr", indptr, f"per-neuron {side}synaptic counts by neuropil: row pointer")
        writer.add(f"np_{side}_neuropil", codes.astype(np.uint8), "neuropil code")
        writer.add(f"np_{side}_count", count.astype(smallest_uint(int(count.max()))), "synapses (all partners)")
        lap(f"neuropil counts ({side}): {obs['rows']:,} rows, {obs['rows_proofread_neurons']:,} for proofread neurons")
    writer.add_vocab("neuropil", neuropils.values)
    observed["neuropil_names_total"] = len(neuropils.index) - 1

    # ---- annotations ---------------------------------------------------------------------------
    ann_path = catalog.local_path("annotations")
    ann = read_annotations(ann_path)
    ids = ann.column("root_id")
    ann_ids = pc.fill_null(ids, -1).to_numpy()
    pos, hit = map_ids(root, ann_ids)
    present = np.zeros(n, dtype=bool)
    present[pos[hit]] = True
    observed["annotations"] = {
        "file": str(ann_path.relative_to(catalog.raw_dir)),
        "rows": ann.num_rows,
        "columns": ann.column_names,
        "root_id_null": ids.null_count,
        "rows_not_in_root_ids": int((~hit).sum()),
        "rows_not_in_root_ids_ids": ann_ids[~hit][:50].tolist(),
        "duplicate_rows_same_neuron": int(hit.sum() - present.sum()),
        "neurons_without_row": int(n - present.sum()),
        "neurons_without_row_ids": root[~present][:50].tolist(),
        "null_counts": {c: ann.column(c).null_count for c in ann.column_names},
    }
    writer.add("ann_has_row", present, "neuron has a row in the annotation file")
    for column in ann.column_names:
        if column in ANN_SKIP:
            continue
        values = ann.column(column)
        if column in ANN_NUMERIC:
            full = np.full(n, np.nan, dtype=np.float32)
            full[pos[hit]] = pc.fill_null(values, np.nan).to_numpy()[hit]
        elif column in ANN_IDS:
            full = np.full(n, -1, dtype=np.int64)
            full[pos[hit]] = pc.fill_null(values, -1).to_numpy()[hit]
        else:
            raw_vocab = Vocab()
            codes = raw_vocab.encode(values)
            vocab, remap = raw_vocab.sorted()
            full = np.zeros(n, dtype=smallest_uint(len(vocab.index) - 1))
            full[pos[hit]] = remap[codes][hit]
            writer.add_vocab(column, vocab.values)
        writer.add(f"ann_{column}", full, f"annotation '{column}' ({ann_path.parent.name})")
    lap(f"annotations: {ann.num_rows:,} rows, {observed['annotations']['neurons_without_row']} neurons without a row")

    manifest = {
        "dataset": catalog.id,
        "title": catalog.title,
        "license": catalog.license,
        "citations": list(catalog.citations),
        "annotation_version": ann_path.parent.name,
        "sources": {fid: {"file": str(catalog.local_path(fid).relative_to(catalog.raw_dir)),
                          "sha256": lock.get(fid, {}).get("sha256")} for fid in REQUIRED},
        "counts": {"neurons": n, "edges": n_edges, "rows": observed["connections"]["rows"] - observed["connections"]["rows_dropped"],
                   "synapses": observed["edges"]["synapses"]},
        "params": {
            "neuron_index": "position of root_id in ascending order",
            "edge": "ordered neuron pair (pre, post) with >= 1 synapse in any neuropil; self-connections kept",
            "edge_syn_count": "sum of syn_count over the pair's neuropil rows",
            "nt_classes": [c.removesuffix("_avg") for c in NT_COLUMNS],
            "nt_prob_q": f"synapse-weighted mean of the rows' *_avg columns, stored as rint(p*{NT_LEVELS}) (uint8)",
            "rows_dropped_rule": "pre or post not in root ids, or syn_count <= 0",
        },
        "observed": observed,
        "build": {**runinfo.collect(), "seconds": round(time.monotonic() - t0, 1), "peak_rss": peak_rss_bytes()},
    }
    final = writer.commit(manifest)
    lap(f"store committed: {final}")
    return final
