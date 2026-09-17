"""Writes an analysis run to disk: summary.json, CSV tables, figures and a human-readable REPORT.md."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..telemetry import fmt_bytes

# Published values used for comparison. Lin et al. 2024 analysed release 630 (127,978 neurons) at >= 5 synapses,
# so agreement in magnitude — not identity — is the expectation for release 783.
REFERENCE = {
    "lin2024": "Lin et al. 2024, Nature 634:153 (release 630, pairs >= 5 synapses)",
    "dorkenwald2024": "Dorkenwald et al. 2024, Nature 634:124 (release 783)",
    "values": [
        ("connections >= 5 synapses", "2,700,513", "dorkenwald2024"),
        ("density (>= 5)", "0.000161 (text) / 0.000160 (Table 2)", "lin2024"),
        ("reciprocity (>= 5)", "0.138", "lin2024"),
        ("reciprocity (>= 1)", "0.2655 (repository code output)", "lin2024"),
        ("reciprocity ratio to ER / to degree-preserving null (>= 5)", "x858 / x43.8", "lin2024"),
        ("neurons with a reciprocal partner (>= 5)", "77,607 of 127,978", "lin2024"),
        ("global clustering (>= 5)", "0.0477 (text) / 0.0463 (Table 2)", "lin2024"),
        ("clustering ratio to ER / to degree-preserving null", "x144 / x7.57", "lin2024"),
        ("largest SCC / WCC", "93.3 % / 98.8 % of neurons", "lin2024"),
        ("mean path length: directed in giant SCC / undirected in giant WCC", "4.42 / 3.91 hops", "lin2024"),
        ("small-worldness S_delta", "141", "lin2024"),
        ("rich-club cut-off (total degree) / neurons above", "37 / 40,218", "lin2024"),
        ("intrinsic neurons: median in / out degree (>= 5)", "11 / 13", "dorkenwald2024"),
        ("intrinsic neurons: in/out degree Pearson R", "0.76", "lin2024"),
    ],
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def figures(out: dict, analyzer, conn, fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .graph import Graph

    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=130)
        plt.close(fig)
        made.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for t, ax in zip(sorted(analyzer.s.thresholds), axes):
        g = Graph.from_connectome(conn, min_synapses=t)
        for label, deg in (("in-degree", g.in_degree()), ("out-degree", g.out_degree())):
            values = np.sort(deg[deg > 0])
            ccdf = 1 - np.arange(values.size) / values.size
            ax.loglog(values, ccdf, label=label)
        ax.set(title=f"degree CCDF, {g.name}", xlabel="degree", ylabel="P(X >= x)")
        ax.legend()
    save(fig, "degree_ccdf.png")

    syn = np.asarray(conn["syn_count"])
    bins = np.unique(np.round(np.logspace(0, np.log10(syn.max() + 1), 40)))
    hist, edges = np.histogram(syn, bins=bins)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(edges[:-1][hist > 0], hist[hist > 0] / np.diff(edges)[hist > 0], "o-", ms=3)
    ax.set(title="synapses per connected pair", xlabel="synapses", ylabel="pairs per unit bin")
    save(fig, "synapses_per_pair.png")

    reg = out["regions"]
    labels = [x or "(none)" for x in reg["super_class_labels"]]
    mat = np.array(reg["super_class_synapses"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.log10(mat + 1), cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(title=f"synapses between super classes (log10), {reg['graph']}", xlabel="post", ylabel="pre")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "super_class_matrix.png")

    names = [x or "(unassigned)" for x in analyzer.neuropils]
    mat = analyzer.arrays["home_neuropil_synapse_matrix"].astype(np.float64)
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(np.log10(mat + 1), cmap="magma")
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=5)
    ax.set_yticks(range(len(names)), names, fontsize=5)
    ax.set(title="synapses between home neuropils (log10); pre = row, post = column")
    fig.colorbar(im, ax=ax, shrink=0.6)
    save(fig, "home_neuropil_matrix.png")

    summary = out["nulls"]["summary"]
    if "motifs" in summary:
        motifs = summary["motifs"]
        names = list(motifs)
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(names))
        for offset, key, label in ((-0.2, "ratio_to_null", "vs degree-preserving null"), (0.2, "ratio_to_er", "vs ER")):
            ax.bar(x + offset, [np.log2(motifs[n][key]) if motifs[n][key] else 0 for n in names], width=0.4, label=label)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, names)
        ax.set(title=f"three-node motif counts, log2(real / null), {out['nulls']['graph']}", ylabel="log2 ratio")
        ax.legend()
        save(fig, "motifs.png")

    rc = summary["rich_club"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogx(rc["ks"], rc["phi_normalized"], "o-", ms=3)
    ax.axhline(1.0, color="k", lw=0.8)
    if rc["cutoff_total_degree"]:
        ax.axvline(rc["cutoff_total_degree"], color="r", ls="--", lw=0.8, label=f"cut-off k = {rc['cutoff_total_degree']}")
        ax.legend()
    ax.set(title=f"normalized rich-club coefficient, {out['nulls']['graph']}", xlabel="total degree k", ylabel="phi / phi_null")
    save(fig, "rich_club.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    series = [(f"{name}, {kind}", g[f"paths_{kind}"]) for name, g in out["graphs"].items()
              for kind in ("directed_giant_scc", "undirected_giant_wcc") if g[f"paths_{kind}"]]
    if out["nulls"]["er"]["paths_directed_giant_scc"]:
        series.append(("ER null, directed_giant_scc", out["nulls"]["er"]["paths_directed_giant_scc"]))
    for label, p in series:
        hist = {int(k): v for k, v in p["histogram"].items()}
        hops = np.array(sorted(hist))
        counts = np.array([hist[h] for h in hops], dtype=float)
        ax.plot(hops, counts / counts.sum(), "o-", ms=3, label=f"{label} (mean {p['mean']:.2f})")
    ax.set(title="shortest-path length distribution (sampled sources)", xlabel="hops", ylabel="fraction of pairs")
    ax.legend(fontsize=7)
    save(fig, "path_lengths.png")

    sizes = out["communities"]["largest_sizes"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(1, len(sizes) + 1), sizes)
    ax.set(title=f"largest Leiden communities, {out['communities']['graph']}", xlabel="rank", ylabel="neurons")
    save(fig, "community_sizes.png")
    return made


def write(out: dict, analyzer, conn, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1, default=_json_default) + "\n")
    for name, rows in analyzer.tables.items():
        write_csv(out_dir / "tables" / f"{name}.csv", rows)
    made = figures(out, analyzer, conn, out_dir / "figures")
    (out_dir / "REPORT.md").write_text(markdown(out, made, analyzer.tables))
    return out_dir


def _f(x, digits=4):
    if x is None:
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}g}"


def _cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return _f(value)
    return str(value).replace("|", "\\|")


def rerender(out_dir: Path) -> Path:
    """Rebuild REPORT.md from summary.json and tables/*.csv without re-running the analysis."""
    out = json.loads((out_dir / "summary.json").read_text())
    tables = {}
    for path in sorted((out_dir / "tables").glob("*.csv")):
        with open(path, newline="") as fh:
            tables[path.stem] = list(csv.DictReader(fh))
    made = sorted(p.name for p in (out_dir / "figures").glob("*.png"))
    (out_dir / "REPORT.md").write_text(markdown(out, made, tables))
    return out_dir / "REPORT.md"


def _table(rows: list[dict], columns: list[str] | None = None) -> list[str]:
    if not rows:
        return ["(none)"]
    columns = columns or list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_cell(row.get(c, "")) for c in columns) + " |" for row in rows]
    return lines


def markdown(out: dict, made: list[str], tables: dict[str, list[dict]]) -> str:
    run = out["run"]
    L = [f"# Connectome analysis — {out['dataset']}", "",
         f"Annotations {out['annotation_version']} · commit `{run['git_commit'] or 'n/a'}`{' (dirty)' if run['git_dirty'] else ''} · "
         f"{run['cpu']}, {fmt_bytes(run['ram_total_bytes'])} RAM · {run['seconds']:.0f} s · peak RSS {fmt_bytes(run['peak_rss'])} "
         f"(budget {fmt_bytes(run['memory_budget'])}) · seed {out['settings']['seed']}", "",
         "All numbers are generated by `biobrain analyze`; nothing in this file is edited by hand. "
         "Self-connections are excluded from every graph statistic "
         f"({out['self_connections_dropped']:,} self-connection pairs dropped).", "",
         "## Graph overview", ""]
    cols = list(out["graphs"])
    rows = [("neurons", lambda g: g["neurons"]), ("connections (ordered pairs)", lambda g: g["connections"]),
            ("synapses", lambda g: g["synapses"]), ("density", lambda g: g["density"]),
            ("neurons with >= 1 connection", lambda g: g["components"]["neurons_with_edges"]),
            ("reciprocity P(j->i | i->j)", lambda g: g["reciprocity"]["reciprocity"]),
            ("reciprocity / ER expectation", lambda g: g["reciprocity"]["ratio_to_er"]),
            ("reciprocity / degree-preserving expectation (analytic)", lambda g: g["reciprocity"]["ratio_to_cfg_analytic"]),
            ("neurons with a reciprocal partner", lambda g: g["reciprocity"]["neurons_with_a_reciprocal_partner"]),
            ("largest WCC (share of all neurons)", lambda g: g["components"]["weak"]["largest_fraction_of_all"]),
            ("largest SCC (share of all neurons)", lambda g: g["components"]["strong"]["largest_fraction_of_all"]),
            ("largest SCC (share of neurons with edges)", lambda g: g["components"]["strong"]["largest_fraction_of_neurons_with_edges"]),
            ("weak components / strong components", lambda g: f"{g['components']['weak']['components']:,} / {g['components']['strong']['components']:,}"),
            ("global clustering (undirected view)", lambda g: g["clustering"]["global_clustering"] if g["clustering"] else None),
            ("mean directed path length, giant SCC", lambda g: g["paths_directed_giant_scc"]["mean"] if g["paths_directed_giant_scc"] else None),
            ("max observed directed path length", lambda g: g["paths_directed_giant_scc"]["max_observed"] if g["paths_directed_giant_scc"] else None),
            ("mean undirected path length, giant WCC", lambda g: g["paths_undirected_giant_wcc"]["mean"] if g["paths_undirected_giant_wcc"] else None),
            ("in/out degree Pearson (intrinsic neurons)", lambda g: g["degrees_intrinsic_neurons"]["in_out_degree_pearson"]),
            ("median in / out degree (intrinsic neurons)", lambda g: f"{g['degrees_intrinsic_neurons']['in_degree']['median']:g} / {g['degrees_intrinsic_neurons']['out_degree']['median']:g}"),
            ("mean in degree (intrinsic neurons)", lambda g: g["degrees_intrinsic_neurons"]["in_degree"]["mean"]),
            ("max in / out degree (all)", lambda g: f"{g['degrees_all_neurons']['in_degree']['max']:,.0f} / {g['degrees_all_neurons']['out_degree']['max']:,.0f}"),
            ("in-degree Gini (all neurons)", lambda g: g["degrees_all_neurons"]["in_degree"]["gini"]),
            ("share of total degree held by the top 1 % of neurons", lambda g: g["degrees_all_neurons"]["top1pct_share_of_total_degree"])]
    L += ["| metric | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for label, fn in rows:
        L.append(f"| {label} | " + " | ".join(_cell(fn(out["graphs"][c])) for c in cols) + " |")
    L += ["", "Path lengths use sampled sources (see `summary.json` for the sample size and the standard error).", "",
          "## Published reference values (for comparison, not for tuning)", "",
          "| quantity | published | source |", "|---|---|---|"]
    L += [f"| {q} | {v} | {REFERENCE[s]} |" for q, v, s in REFERENCE["values"]]

    nulls = out["nulls"]
    s = nulls["summary"]
    L += ["", f"## Null models ({nulls['graph']})", "", s["note"], "",
          "| quantity | real | ER | degree-preserving (mean ± sd) |", "|---|---:|---:|---:|"]
    for key in ("reciprocity", "clustering"):
        if key in s:
            dp = s[key]["degree_preserving"]
            L.append(f"| {key} | {_f(s[key]['real'])} | {_f(s[key]['er'])} | {_f(dp['mean'])} ± {_f(dp['std'])} |")
    if "small_world" in out:
        sw = out["small_world"]
        L += ["", f"Small-worldness S_delta = **{sw['S_delta']:.1f}** (C/C_ER = {sw['clustering_ratio_to_er']:.1f}, "
                  f"L/L_ER = {sw['path_length_ratio_to_er']:.3f}); {sw['definition']}."]
    if "motifs" in s:
        L += ["", "### Three-node motifs (triad census, MAN notation; 030T = feed-forward loop, 030C = 3-cycle)", "",
              "| motif | real | ER | null mean | real / null | real / ER | z (null) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, m in s["motifs"].items():
            L.append(f"| {name} | {m['real']:,} | {m['er']:,} | {m['null_mean']:,.0f} | {_f(m['ratio_to_null'], 3)} | "
                     f"{_f(m['ratio_to_er'], 3)} | {_f(m['z_null'], 3)} |")
    rc = s["rich_club"]
    L += ["", f"Rich club: {rc['rule']} → cut-off total degree **{rc['cutoff_total_degree']}**, "
              f"**{_f(rc['neurons_in_rich_club'])}** neurons above it."]

    com = out["communities"]
    L += ["", f"## Communities ({com['graph']})", "",
          f"Leiden, {com['objective']}: modularity **{com['modularity']:.4f}**, {com['communities_with_edges']:,} communities "
          f"(stopped: {com['stopped']}; {len(com['history'])} iterations). Largest: {com['largest_sizes'][:10]}.", "",
          f"- agreement with home neuropil: NMI {com['agreement_with_home_neuropil']['nmi']:.3f}, ARI {com['agreement_with_home_neuropil']['ari']:.3f}",
          f"- agreement with super class: NMI {com['agreement_with_super_class']['nmi']:.3f}, ARI {com['agreement_with_super_class']['ari']:.3f}",
          f"- modularity of the home-neuropil partition itself: {com['modularity_of_home_neuropil_partition']:.4f}; "
          f"of the super-class partition: {com['modularity_of_super_class_partition']:.4f}",
          f"- unweighted Leiden, real vs degree-preserving null: {com['unweighted_vs_degree_preserving_null']['real']:.4f} vs "
          f"{com['unweighted_vs_degree_preserving_null']['null']:.4f}", ""]
    L += _table(com["composition_top15"])

    reg = out["regions"]
    L += ["", f"## Regions ({reg['graph']})", "", f"Home neuropil rule: {reg['home_neuropil_rule']['rule']}. "
          f"Assigned from presynapses: {reg['home_neuropil_rule']['from_presynapses']:,}; postsynapse fallback: "
          f"{reg['home_neuropil_rule']['from_postsynapse_fallback']:,}; unassigned: {reg['home_neuropil_rule']['unassigned']:,}.", "",
          f"Synapses whose pre and post neuron share a home neuropil: **{reg['synapses_within_home_neuropil_fraction']:.1%}**.", "",
          "Strongest projections between different home neuropils:", ""]
    L += _table(reg["top_between_home_neuropils"])
    L += ["", "Super classes — share of neurons vs share of incoming synapses (`tables/super_class_flow.csv`):", ""]
    L += _table(tables.get("super_class_flow", []))

    hubs = out["hubs"]
    for key in ("in_degree", "out_degree", "in_synapses", "out_synapses"):
        L += ["", f"## Hubs by {key.replace('_', ' ')} ({hubs['graph']})", ""]
        L += _table(hubs[key])
    L += ["", f"## Strongest connections ({hubs['graph']})", ""]
    L += _table(hubs["strongest_connections"])

    b = out["bottlenecks"]
    L += ["", "## Potential bottlenecks: sampled betweenness", ""]
    if b.get("skipped"):
        L.append("Skipped by the cost gate.")
    else:
        L += [f"{b['method']} ({b['graph']}; samples: {[m['sources'] for m in b['samples']]} sources). "
              f"Stability between the two samples: top-100 Jaccard {b['stability']['top100_jaccard']:.2f}, Spearman on the "
              f"union of top-1000 {b['stability']['spearman_on_union_of_top1000']:.2f}.", "",
              "Top 1 % by betweenness — super-class share [in top 1 %, among all neurons]: "
              + ", ".join(f"{k} {v[0]:.1%} vs {v[1]:.1%}" for k, v in b["top1pct_super_class_share_vs_all_neurons"].items()), ""]
        L += _table(b["top"])

    L += ["", "## Degrees by super class", ""]
    L += _table(out["degrees_by_super_class"])
    L += ["", "## Cost control", "", f"Calibration ({out['calibration']['graph']}): triad census "
          f"{out['calibration']['triad_census_wedges_per_s']:.2e} wedges/s, clustering {out['calibration']['clustering_wedges_per_s']:.2e} wedges/s; "
          f"safety factor {out['settings']['safety_factor']}, step limit {out['settings']['step_time_limit_s']:.0f} s.", ""]
    L += _table([{k: row.get(k, "") for k in ("step", "estimate_s", "actual_s", "decision", "wedges", "sources_used")}
                 for row in out["costs"]])
    L += ["", "## Figures", ""] + [f"![{name}](figures/{name})" for name in made]
    return "\n".join(L) + "\n"
