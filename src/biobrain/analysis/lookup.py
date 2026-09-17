"""Neuron lookup: annotations, home neuropil, degrees and strongest partners of one neuron."""

from __future__ import annotations

import numpy as np

from ..connectome.store import Connectome
from .regions import home_blocks, primary_neuropil

NT_NAMES = ("gaba", "ach", "glut", "oct", "ser", "da")


def describe(conn: Connectome, root_id: int, top: int = 10) -> dict:
    i = int(conn.index_of(root_id)[0])
    names = conn.vocab("neuropil")
    blocks, _ = home_blocks(conn)
    ann = {name.removeprefix("ann_"): conn[name][i] for name in conn.arrays if name.startswith("ann_")}
    annotations = {}
    for key, value in ann.items():
        if key in conn.manifest["vocab"]:
            annotations[key] = conn.vocab(key)[int(value)] or None
        elif isinstance(value, np.floating):
            annotations[key] = None if np.isnan(value) else float(value)
        else:
            annotations[key] = value.item() if isinstance(value, np.generic) else value

    lo, hi = conn["out_indptr"][i], conn["out_indptr"][i + 1]
    out_idx, out_syn = conn["out_indices"][lo:hi], conn["syn_count"][lo:hi].astype(np.int64)
    out_nt = conn["nt_prob_q"][lo:hi].astype(np.int64)
    clo, chi = conn["in_indptr"][i], conn["in_indptr"][i + 1]
    in_edges = conn["in_edge"][clo:chi]
    in_idx, in_syn = conn["in_indices"][clo:chi], conn["syn_count"][in_edges].astype(np.int64)

    def partners(idx, syn, k):
        order = np.argsort(syn, kind="stable")[::-1][:k]
        return [{"root_id": int(conn["root_id"][j]), "synapses": int(syn[o]), "cell_type": conn.labels("cell_type", j) or None,
                 "super_class": conn.labels("super_class", j) or None, "home_neuropil": names[blocks[j]] or None}
                for o, j in zip(order, idx[order])]

    def neuropil_counts(side):
        a, b = conn[f"np_{side}_indptr"][i], conn[f"np_{side}_indptr"][i + 1]
        codes, counts = conn[f"np_{side}_neuropil"][a:b], conn[f"np_{side}_count"][a:b]
        order = np.argsort(counts)[::-1]
        return {names[codes[o]] or "(unassigned)": int(counts[o]) for o in order}

    predicted = out_nt.sum(axis=1) > 0  # an all-zero row means no synapse of that edge has a prediction
    weights = out_syn[predicted]
    nt_mean = (out_nt[predicted] * weights[:, None]).sum(axis=0) / max(weights.sum(), 1) / 255
    return {
        "root_id": int(root_id), "index": i, "annotations": annotations,
        "home_neuropil": names[blocks[i]] or None,
        "primary_input_neuropil": names[primary_neuropil(conn, "post")[i]] or None,
        "out_degree": int(hi - lo), "in_degree": int(chi - clo),
        "out_synapses_to_proofread": int(out_syn.sum()), "in_synapses_from_proofread": int(in_syn.sum()),
        "presynapses_all_partners_by_neuropil": neuropil_counts("pre"),
        "postsynapses_all_partners_by_neuropil": neuropil_counts("post"),
        "outgoing_transmitter_mean_probability": {n: round(float(p), 3) for n, p in zip(NT_NAMES, nt_mean)},
        "strongest_outgoing": partners(out_idx, out_syn, top),
        "strongest_incoming": partners(in_idx, in_syn, top),
    }


def render(info: dict) -> str:
    a = info["annotations"]
    lines = [f"neuron {info['root_id']} (index {info['index']})",
             f"  type {a.get('cell_type')} | super class {a.get('super_class')} | class {a.get('cell_class')} | flow {a.get('flow')} | side {a.get('side')}",
             f"  predicted transmitter (annotation) {a.get('top_nt')} ({a.get('top_nt_conf')}) | home neuropil {info['home_neuropil']} | main input neuropil {info['primary_input_neuropil']}",
             f"  out: {info['out_degree']:,} partners, {info['out_synapses_to_proofread']:,} synapses | in: {info['in_degree']:,} partners, {info['in_synapses_from_proofread']:,} synapses",
             f"  presynapses by neuropil (all partners): {info['presynapses_all_partners_by_neuropil']}",
             f"  postsynapses by neuropil (all partners): {info['postsynapses_all_partners_by_neuropil']}",
             f"  outgoing transmitter probabilities (synapse-weighted): {info['outgoing_transmitter_mean_probability']}"]
    for key in ("strongest_outgoing", "strongest_incoming"):
        lines.append(f"  {key.replace('_', ' ')}:")
        lines += [f"    {p['synapses']:>5}  {p['root_id']}  {p['cell_type'] or '-':<16} {p['super_class'] or '-':<20} {p['home_neuropil'] or '-'}"
                  for p in info[key]]
    return "\n".join(lines)
