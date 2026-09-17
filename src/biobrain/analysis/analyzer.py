"""ConnectomeAnalyzer: the Milestone 1 graph analyses, with an explicit cost gate on every expensive step.

Cost policy: cheap (linear) statistics always run exactly. Wedge-bound algorithms (clustering, triad
census) are estimated from a throughput calibration on a random graph measured on this machine at
the start of the run; BFS-based ones (path lengths, betweenness) from a timed pilot. A step whose
estimate exceeds `step_time_limit_s` is skipped or down-sampled, and the decision, the estimate and
the actual time are all written to the output.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import spearmanr

from .. import runinfo
from ..connectome.store import Connectome
from ..telemetry import MemoryBudget, fmt_bytes, peak_rss_bytes, rss_bytes
from . import regions, stats
from .graph import Graph, degree_preserving_null, erdos_renyi, undirected_edges


@dataclass
class Settings:
    seed: int = 20260916
    thresholds: tuple[int, ...] = (1, 5)
    path_sources: int = 200
    betweenness_sources: int = 256
    null_samples: int = 5
    step_time_limit_s: float = 900
    top: int = 20
    safety_factor: float = 2.0
    rich_club_ratio: float = 1.01  # Lin et al. 2024: normalized rich-club coefficient > 1.01


class CostLog:
    def __init__(self, limit_s: float):
        self.limit = limit_s
        self.rows: list[dict] = []

    def decide(self, step: str, estimate_s: float, **proxy) -> bool:
        run = estimate_s <= self.limit
        self.rows.append({"step": step, **proxy, "estimate_s": round(float(estimate_s), 1),
                          "decision": "run" if run else f"skipped: estimate above the {self.limit:.0f} s limit"})
        return run

    def actual(self, seconds: float) -> None:
        self.rows[-1]["actual_s"] = round(seconds, 1)


def calibrate(seed: int) -> dict:
    """Wedges/second of the two wedge-bound algorithms on a 50k-node, 1M-edge random graph."""
    g = erdos_renyi(50_000, 1_000_000, np.random.default_rng(seed))
    wedges = stats.wedge_count(g)
    t0 = time.monotonic()
    stats.triad_census(g)
    t1 = time.monotonic()
    stats.clustering(g)
    t2 = time.monotonic()
    return {"graph": "directed G(n=50,000, m=1,000,000)", "wedges": wedges,
            "triad_census_wedges_per_s": wedges / max(t1 - t0, 1e-3),
            "clustering_wedges_per_s": wedges / max(t2 - t1, 1e-3)}


class ConnectomeAnalyzer:
    def __init__(self, conn: Connectome, settings: Settings | None = None, budget: MemoryBudget | None = None, log=print):
        self.conn = conn
        self.s = settings or Settings()
        self.budget = budget or MemoryBudget.from_env()
        self.costs = CostLog(self.s.step_time_limit_s)
        self.t0 = time.monotonic()
        self._log = log
        self.tables: dict[str, list[dict]] = {}
        self.arrays: dict[str, np.ndarray] = {}

    def log(self, msg: str) -> None:
        self._log(f"[{time.monotonic() - self.t0:7.1f} s | RSS {fmt_bytes(rss_bytes()):>10}] {msg}")

    # ---- helpers --------------------------------------------------------------------------------
    def annotate(self, index: np.ndarray, **values) -> list[dict]:
        c = self.conn
        rows = []
        for j, i in enumerate(np.asarray(index)):
            rows.append({"root_id": int(c["root_id"][i]), "cell_type": c.labels("cell_type", i),
                         "super_class": c.labels("super_class", i), "side": c.labels("side", i),
                         "home_neuropil": self.neuropils[self.blocks[i]] or "-",
                         **{k: (v[j].item() if isinstance(v[j], np.generic) else v[j]) for k, v in values.items()}})
        return rows

    def top(self, values: np.ndarray, name: str) -> list[dict]:
        idx = np.argsort(values, kind="stable")[::-1][: self.s.top]
        return self.annotate(idx, **{name: values[idx]})

    def paths(self, step: str, g: Graph, mask: np.ndarray, directed: bool, sources: int) -> dict | None:
        if not directed:
            self.budget.check(f"{step}: symmetrized sparse matrix", 24 * g.m)
        pilot = stats.path_lengths(g, mask, k=8, seed=self.s.seed, directed=directed, batch=8)
        per_source = max(pilot["seconds"], 0.05) / pilot["sources"]
        k = int(min(sources, self.s.step_time_limit_s / per_source))
        if not self.costs.decide(step, per_source * max(k, 16), sources_requested=sources, sources_used=k,
                                 pilot_seconds_per_source=round(per_source, 4)) or k < 16:
            return None
        t = time.monotonic()
        result = stats.path_lengths(g, mask, k=k, seed=self.s.seed, directed=directed)
        self.costs.actual(time.monotonic() - t)
        return result

    def wedge_step(self, step: str, g: Graph, wedges: float, rate_key: str, fn):
        estimate = self.s.safety_factor * wedges / self.calibration[rate_key]
        if not self.costs.decide(step, estimate, wedges=wedges):
            return None
        self.budget.check(f"{step}: igraph graph", 80 * g.m + 64 * g.n)
        t = time.monotonic()
        result = fn(g)
        self.costs.actual(time.monotonic() - t)
        return result

    # ---- analyses -------------------------------------------------------------------------------
    def describe(self, g: Graph, intrinsic: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        key = g.name
        out = {"name": g.name, "neurons": g.n, "connections": g.m, "synapses": int(g.weights.sum()),
               "density": g.m / (g.n * (g.n - 1)),
               "degrees_all_neurons": stats.degrees(g),
               "degrees_intrinsic_neurons": stats.degrees(g, intrinsic),
               "reciprocity": stats.reciprocity(g)}
        comp, weak, strong = stats.components(g)
        out["components"] = comp
        self.log(f"{key}: {g.m:,} connections, reciprocity {out['reciprocity']['reciprocity']:.4f}, "
                 f"giant SCC {comp['strong']['largest']:,}, giant WCC {comp['weak']['largest']:,}")
        wedges = stats.wedge_count(g)
        out["wedges_undirected"] = wedges
        out["clustering"] = self.wedge_step(f"clustering [{key}]", g, wedges, "clustering_wedges_per_s", stats.clustering)
        out["paths_directed_giant_scc"] = self.paths(f"path lengths, directed, giant SCC [{key}]", g, strong, True,
                                                     self.s.path_sources)
        out["paths_undirected_giant_wcc"] = self.paths(f"path lengths, undirected, giant WCC [{key}]", g, weak, False,
                                                       self.s.path_sources)
        out["triads"] = self.wedge_step(f"triad census [{key}]", g, wedges, "triad_census_wedges_per_s", stats.triad_census)
        self.log(f"{key}: clustering, paths, triads done")
        return out, weak, strong

    def nulls(self, g: Graph, real: dict) -> dict:
        rng = np.random.default_rng(self.s.seed + 1)
        er = erdos_renyi(g.n, g.m, rng, name="ER")
        _, _, er_scc = stats.components(er)
        er_out = {"reciprocity": stats.reciprocity(er)["reciprocity"],
                  "clustering": stats.clustering(er)["global_clustering"],
                  "paths_directed_giant_scc": self.paths("path lengths, directed [ER null]", er, er_scc, True, 64),
                  "triads": self.wedge_step("triad census [ER null]", er, stats.wedge_count(er), "triad_census_wedges_per_s",
                                            stats.triad_census)}
        self.log("ER null done")
        ks = np.unique(np.round(np.logspace(0, 3.3, 70)).astype(np.int64))
        cfg = []
        for i in range(self.s.null_samples):
            seed = self.s.seed + 100 + i
            t = time.monotonic()
            h = degree_preserving_null(g, seed=seed)
            sample = {"seed": seed, "rewire_seconds": round(time.monotonic() - t, 1),
                      "reciprocity": stats.reciprocity(h)["reciprocity"],
                      "clustering": stats.clustering(h)["global_clustering"],
                      "rich_club_phi": stats.rich_club(h, ks).tolist()}
            triads = self.wedge_step(f"triad census [degree-preserving null {i + 1}]", h, stats.wedge_count(h),
                                     "triad_census_wedges_per_s", stats.triad_census)
            sample["triads"] = triads["counts"] if triads else None
            cfg.append(sample)
            self.log(f"degree-preserving null {i + 1}/{self.s.null_samples} done")
        return {"er": er_out, "degree_preserving": cfg, "rich_club_ks": ks.tolist(),
                "summary": self.null_summary(g, real, er_out, cfg, ks)}

    def null_summary(self, g: Graph, real: dict, er: dict, cfg: list[dict], ks: np.ndarray) -> dict:
        def spread(values):
            v = np.array(values, dtype=np.float64)
            return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else None}

        out = {"samples": len(cfg), "note": "degree-preserving nulls keep every in- and out-degree; "
                                           f"{len(cfg)} samples (Lin et al. used 100), so z-scores are indicative only"}
        r = real["reciprocity"]["reciprocity"]
        out["reciprocity"] = {"real": r, "er": er["reciprocity"], "degree_preserving": spread([c["reciprocity"] for c in cfg])}
        if real["clustering"]:
            c = real["clustering"]["global_clustering"]
            out["clustering"] = {"real": c, "er": er["clustering"], "degree_preserving": spread([x["clustering"] for x in cfg])}
        if real["triads"] and er["triads"] and all(x["triads"] for x in cfg):
            motif = {}
            for name in stats.CONNECTED_TRIADS:
                real_n = real["triads"]["counts"][name]
                null = np.array([x["triads"][name] for x in cfg], dtype=np.float64)
                sd = null.std(ddof=1) if null.size > 1 else np.nan
                motif[name] = {"real": real_n, "er": er["triads"]["counts"][name], "null_mean": float(null.mean()),
                               "null_std": float(sd), "ratio_to_null": real_n / null.mean() if null.mean() else None,
                               "ratio_to_er": real_n / er["triads"]["counts"][name] if er["triads"]["counts"][name] else None,
                               "z_null": float((real_n - null.mean()) / sd) if sd and np.isfinite(sd) and sd > 0 else None}
            out["motifs"] = motif
        phi = stats.rich_club(g, ks)
        null_phis = np.array([x["rich_club_phi"] for x in cfg], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):  # k beyond the maximum degree leaves no rich club
            phi_null = np.where(np.isnan(null_phis).all(axis=0), np.nan, np.nansum(null_phis, axis=0) /
                                np.maximum(np.sum(~np.isnan(null_phis), axis=0), 1))
            norm = phi / phi_null
        deg = g.in_degree() + g.out_degree()
        n_k = deg.size - np.searchsorted(np.sort(deg), ks, side="right")
        out["rich_club"] = {"ks": ks.tolist(), "phi": phi.tolist(), "phi_null_mean": phi_null.tolist(),
                            "phi_normalized": norm.tolist(), "neurons_above_k": n_k.tolist(),
                            "regime": stats.rich_club_regime(ks, norm, n_k, self.s.rich_club_ratio)}
        return out

    def communities(self, g: Graph, blocks: np.ndarray, super_class: np.ndarray) -> dict:
        lo, hi, w = undirected_edges(g)
        self.budget.check("Leiden: igraph undirected graph", 80 * lo.size + 64 * g.n)
        active = (g.in_degree() + g.out_degree()) > 0
        weighted = stats.leiden(g.n, lo, hi, w, seed=self.s.seed, time_limit_s=self.s.step_time_limit_s)
        self.costs.rows.append({"step": "Leiden, synapse-weighted", "decision": "run (iteration-wise, time-limited)",
                                "actual_s": weighted["history"][-1]["t"], "stopped": weighted["stopped"]})
        membership = weighted.pop("membership")
        self.arrays["community"] = membership
        sizes = np.bincount(membership[active])
        sizes = np.sort(sizes[sizes > 0])[::-1]
        out = {"graph": g.name, "objective": "modularity (resolution 1), undirected, weight = synapses both directions",
               **weighted, "communities_with_edges": int(sizes.size), "largest_sizes": sizes[:20].tolist(),
               "neurons_in_top10": float(sizes[:10].sum() / active.sum()),
               "agreement_with_home_neuropil": stats.partition_agreement(membership[active], blocks[active]),
               "agreement_with_super_class": stats.partition_agreement(membership[active], super_class[active]),
               "modularity_of_home_neuropil_partition": stats.modularity(g.n, lo, hi, w, blocks),
               "modularity_of_super_class_partition": stats.modularity(g.n, lo, hi, w, super_class.astype(np.int64))}
        self.log(f"Leiden (weighted): Q={out['modularity']:.4f}, {sizes.size} communities")
        unweighted = stats.leiden(g.n, lo, hi, None, seed=self.s.seed, time_limit_s=self.s.step_time_limit_s / 2)
        null = degree_preserving_null(g, seed=self.s.seed + 999)
        nlo, nhi, _ = undirected_edges(null)
        null_q = stats.leiden(g.n, nlo, nhi, None, seed=self.s.seed, time_limit_s=self.s.step_time_limit_s / 2)
        out["unweighted_vs_degree_preserving_null"] = {"real": unweighted["modularity"], "null": null_q["modularity"],
                                                       "real_iterations": len(unweighted["history"]),
                                                       "null_iterations": len(null_q["history"])}
        self.log(f"Leiden unweighted: real Q={unweighted['modularity']:.4f} vs null Q={null_q['modularity']:.4f}")
        composition = []
        for label in np.argsort(np.bincount(membership, weights=active))[::-1][:15]:
            members = membership == label
            composition.append({"community": int(label), "neurons": int(members.sum()),
                                **{f"top_{kind}": self._shares(codes[members], vocab)
                                   for kind, codes, vocab in (("super_class", super_class, self.conn.vocab("super_class")),
                                                              ("home_neuropil", blocks, self.neuropils))}})
        self.tables["communities_top15"] = composition
        out["composition_top15"] = composition
        return out

    @staticmethod
    def _shares(codes: np.ndarray, vocab: list[str], k: int = 3) -> str:
        counts = np.bincount(codes, minlength=len(vocab))
        idx = np.argsort(counts)[::-1][:k]
        return ", ".join(f"{vocab[i] or '(none)'} {counts[i] / codes.size:.0%}" for i in idx if counts[i])

    def regions(self, g: Graph) -> dict:
        c = self.conn
        k = len(self.neuropils)
        syn, con = regions.group_matrices(g, self.blocks, k)
        self.arrays["home_neuropil_synapse_matrix"] = syn
        off = syn.copy()
        np.fill_diagonal(off, 0)
        flat = np.argsort(off, axis=None)[::-1][: self.s.top]
        located = regions.neuropil_synapses(c)
        sc_vocab = c.vocab("super_class")
        sc_syn, sc_con = regions.group_matrices(g, c["ann_super_class"].astype(np.int64), len(sc_vocab))
        self.arrays["super_class_synapse_matrix"] = sc_syn
        fl_vocab = c.vocab("flow")
        fl_syn, _ = regions.group_matrices(g, c["ann_flow"].astype(np.int64), len(fl_vocab))
        home_sizes = np.bincount(self.blocks, minlength=k)
        self.tables["neuropils"] = [{"neuropil": self.neuropils[i] or "(unassigned)", "home_neurons": int(home_sizes[i]),
                                     "synapses_located_here": int(located[i]),
                                     "synapses_within_home_block": int(syn[i, i]),
                                     "synapses_out_of_home_block": int(syn[i].sum() - syn[i, i]),
                                     "synapses_into_home_block": int(syn[:, i].sum() - syn[i, i])}
                                    for i in np.argsort(located)[::-1]]
        in_syn = sc_syn.sum(axis=0)
        out_syn = sc_syn.sum(axis=1)
        counts = np.bincount(c["ann_super_class"], minlength=len(sc_vocab))
        self.tables["super_class_flow"] = [{"super_class": sc_vocab[i] or "(no annotation)", "neurons": int(counts[i]),
                                            "neuron_share": round(counts[i] / c.n_neurons, 5),
                                            "synapses_in": int(in_syn[i]), "synapses_out": int(out_syn[i]),
                                            "synapse_in_share": round(in_syn[i] / sc_syn.sum(), 5),
                                            "synapses_in_per_neuron": round(in_syn[i] / max(counts[i], 1), 1)}
                                           for i in range(len(sc_vocab))]
        return {"graph": g.name, "home_neuropil_rule": self.block_info,
                "neuropils": k - 1, "synapses_within_home_neuropil_fraction": float(np.trace(syn) / syn.sum()),
                "top_between_home_neuropils": [{"from": self.neuropils[i // k] or "(unassigned)", "to": self.neuropils[i % k] or "(unassigned)",
                                                "synapses": int(off.flat[i]), "connections": int(con.flat[i])} for i in flat],
                "synapses_by_location_unassigned": int(located[0]),
                "super_class_labels": sc_vocab,
                "super_class_synapses": sc_syn.tolist(), "super_class_connections": sc_con.tolist(),
                "flow_labels": fl_vocab, "flow_synapses": fl_syn.tolist()}

    def hubs(self, g: Graph) -> dict:
        kin, kout = g.in_degree(), g.out_degree()
        win = np.bincount(g.indices, weights=g.weights, minlength=g.n).astype(np.int64)
        wout = np.bincount(g.sources(), weights=g.weights, minlength=g.n).astype(np.int64)
        out = {"graph": g.name}
        for name, values in (("in_degree", kin), ("out_degree", kout), ("in_synapses", win), ("out_synapses", wout)):
            out[name] = self.top(values, name)
            self.tables[f"hubs_{name}"] = out[name]
        idx = np.argsort(g.weights, kind="stable")[::-1][: self.s.top]
        src = g.sources()[idx]
        pre = self.annotate(src)
        post = self.annotate(g.indices[idx])
        out["strongest_connections"] = [{"pre_root_id": a["root_id"], "pre_cell_type": a["cell_type"], "pre_super_class": a["super_class"],
                                         "post_root_id": b["root_id"], "post_cell_type": b["cell_type"], "post_super_class": b["super_class"],
                                         "synapses": int(g.weights[i])} for a, b, i in zip(pre, post, idx)]
        self.tables["strongest_connections"] = out["strongest_connections"]
        return out

    def degrees_by_class(self, graphs: dict[str, Graph]) -> list[dict]:
        c = self.conn
        vocab = c.vocab("super_class")
        codes = c["ann_super_class"]
        rows = []
        for code, label in enumerate(vocab):
            members = codes == code
            row = {"super_class": label or "(no annotation)", "neurons": int(members.sum())}
            for key, g in graphs.items():
                row[f"median_in_degree {key}"] = float(np.median(g.in_degree()[members])) if members.any() else None
                row[f"median_out_degree {key}"] = float(np.median(g.out_degree()[members])) if members.any() else None
            rows.append(row)
        self.tables["degrees_by_super_class"] = rows
        return rows

    def bottlenecks(self, g: Graph) -> dict:
        pilot, meta = stats.sampled_betweenness(g, k=8, seed=self.s.seed)
        per_source = max(meta["seconds"], 0.05) / meta["sources"]
        k = int(min(self.s.betweenness_sources, self.s.step_time_limit_s / 2 / per_source))
        if not self.costs.decide(f"sampled betweenness, 2 x {k} sources [{g.name}]", 2 * k * per_source,
                                 pilot_seconds_per_source=round(per_source, 3)) or k < 16:
            return {"skipped": True}
        t = time.monotonic()
        b1, m1 = stats.sampled_betweenness(g, k=k, seed=self.s.seed + 11)
        b2, m2 = stats.sampled_betweenness(g, k=k, seed=self.s.seed + 12)
        self.costs.actual(time.monotonic() - t)
        est = (b1 + b2) / 2
        top1, top2 = set(np.argsort(b1)[::-1][:100]), set(np.argsort(b2)[::-1][:100])
        union = np.array(sorted(set(np.argsort(b1)[::-1][:1000]) | set(np.argsort(b2)[::-1][:1000])))
        top = np.argsort(est)[::-1]
        top_pct = top[: max(1, g.n // 100)]
        sc = self.conn["ann_super_class"][top_pct]
        vocab = self.conn.vocab("super_class")
        share = np.bincount(sc, minlength=len(vocab)) / top_pct.size
        base = np.bincount(self.conn["ann_super_class"], minlength=len(vocab)) / g.n
        rows = self.annotate(top[: self.s.top], betweenness_estimate=np.round(est[top[: self.s.top]], 0))
        self.tables["betweenness_top"] = rows
        return {"graph": g.name, "method": "Brandes from random sources, scaled by n/k; two independent samples averaged",
                "samples": [m1, m2],
                "stability": {"top100_jaccard": len(top1 & top2) / len(top1 | top2),
                              "spearman_on_union_of_top1000": float(spearmanr(b1[union], b2[union]).statistic)},
                "top": rows,
                "top1pct_super_class_share_vs_all_neurons": {vocab[i] or "(none)": [round(float(share[i]), 4), round(float(base[i]), 4)]
                                                             for i in np.argsort(share)[::-1] if share[i] > 0}}

    # ---- orchestration --------------------------------------------------------------------------
    def run(self) -> dict:
        c, s = self.conn, self.s
        self.log(f"analyzing {c.manifest['dataset']}: {c.n_neurons:,} neurons, {c.n_edges:,} connections")
        self.calibration = calibrate(s.seed)
        self.log(f"calibration: triad census {self.calibration['triad_census_wedges_per_s']:.2e} wedges/s, "
                 f"clustering {self.calibration['clustering_wedges_per_s']:.2e} wedges/s")
        self.neuropils = c.vocab("neuropil")
        self.blocks, self.block_info = regions.home_blocks(c)
        intrinsic = c.labels("flow") == "intrinsic"
        out = {"dataset": c.manifest["dataset"], "annotation_version": c.manifest["annotation_version"],
               "settings": asdict(s), "calibration": self.calibration, "graphs": {}}
        graphs = {t: Graph.from_connectome(c, min_synapses=t) for t in sorted(s.thresholds)}
        for g in graphs.values():
            out["graphs"][g.name], _, _ = self.describe(g, intrinsic)
        out["self_connections_dropped"] = int(np.sum(c.edge_sources() == c["out_indices"]))
        main, full = graphs[max(graphs)], graphs[min(graphs)]
        out["nulls"] = {"graph": main.name, **self.nulls(main, out["graphs"][main.name])}
        real = out["graphs"][main.name]
        er = out["nulls"]["er"]
        if real["clustering"] and real["paths_directed_giant_scc"] and er["paths_directed_giant_scc"]:
            c_ratio = real["clustering"]["global_clustering"] / er["clustering"]
            l_ratio = real["paths_directed_giant_scc"]["mean"] / er["paths_directed_giant_scc"]["mean"]
            out["small_world"] = {"graph": main.name, "clustering_ratio_to_er": c_ratio, "path_length_ratio_to_er": l_ratio,
                                  "S_delta": c_ratio / l_ratio, "definition": "(C / C_ER) / (L / L_ER), C = global clustering "
                                  "of the undirected view, L = mean directed path length in the giant SCC"}
        out["communities"] = self.communities(main, self.blocks, c["ann_super_class"].astype(np.int64))
        out["regions"] = self.regions(full)
        out["hubs"] = self.hubs(full)
        out["degrees_by_super_class"] = self.degrees_by_class({g.name: g for g in graphs.values()})
        out["bottlenecks"] = self.bottlenecks(main)
        out["costs"] = self.costs.rows
        out["run"] = {**runinfo.collect(), "seconds": round(time.monotonic() - self.t0, 1), "peak_rss": peak_rss_bytes(),
                      "memory_budget": self.budget.limit}
        self.log(f"analysis finished, peak RSS {fmt_bytes(peak_rss_bytes())}")
        return out
