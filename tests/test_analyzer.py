"""End-to-end: synthetic release -> preprocess -> validate -> analyzer -> report files."""

import json

import numpy as np

from biobrain.analysis import report
from biobrain.analysis.analyzer import ConnectomeAnalyzer, Settings
from biobrain.analysis.regions import home_blocks
from biobrain.connectome import preprocess, validate
from biobrain.connectome.store import Connectome


def test_pipeline_end_to_end(synthetic_catalog, budget, tmp_path):
    store = preprocess.run(synthetic_catalog, budget=budget, log=lambda *_: None)
    checks = {c["id"]: c["status"] for c in validate.run(synthetic_catalog, budget=budget, out_dir=tmp_path / "v",
                                                         log=lambda *_: None)["checks"]}
    assert all(checks[k] == "PASS" for k in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "D8", "F1d", "F2d", "C2", "C5"))

    conn = Connectome.load(store)
    settings = Settings(seed=3, path_sources=32, betweenness_sources=32, null_samples=2, step_time_limit_s=120, top=5)
    analyzer = ConnectomeAnalyzer(conn, settings, budget, log=lambda *_: None)
    out = analyzer.run()

    g1, g5 = out["graphs"][">=1 synapse"], out["graphs"][">=5 synapses"]
    assert g1["connections"] > g5["connections"] > 0
    assert 0 <= g5["reciprocity"]["reciprocity"] <= 1
    assert g1["components"]["weak"]["largest"] <= conn.n_neurons
    assert g5["triads"] and sum(g5["triads"]["counts"].values()) == conn.n_neurons * (conn.n_neurons - 1) * (conn.n_neurons - 2) // 6
    # planted neuropil communities must be recovered and agree with the home neuropils
    assert out["communities"]["modularity"] > 0.3
    assert out["communities"]["agreement_with_home_neuropil"]["nmi"] > 0.5
    assert out["regions"]["synapses_within_home_neuropil_fraction"] > 0.5
    assert len(out["nulls"]["degree_preserving"]) == 2 and "motifs" in out["nulls"]["summary"]
    assert not out["bottlenecks"].get("skipped")
    assert all(row["decision"].startswith("run") for row in out["costs"])

    target = report.write(out, analyzer, conn, tmp_path / "analysis")
    saved = json.loads((target / "summary.json").read_text())
    assert saved["graphs"][">=5 synapses"]["connections"] == g5["connections"]
    text = (target / "REPORT.md").read_text()
    assert "## Null models" in text and "## Cost control" in text
    assert {p.name for p in (target / "figures").iterdir()} >= {"degree_ccdf.png", "motifs.png", "home_neuropil_matrix.png"}
    assert (target / "tables" / "hubs_in_degree.csv").is_file()


def test_home_blocks_follow_presynapses(synthetic_catalog, budget):
    conn = Connectome.load(preprocess.run(synthetic_catalog, budget=budget, log=lambda *_: None))
    blocks, info = home_blocks(conn)
    names = conn.vocab("neuropil")
    i = int(np.argmax(conn.out_synapses()))
    lo, hi = conn["np_pre_indptr"][i:i + 2]
    counts = conn["np_pre_count"][lo:hi]
    assert names[blocks[i]] == names[conn["np_pre_neuropil"][lo + int(np.argmax(counts))]]
    assert info["unassigned"] + info["from_presynapses"] + info["from_postsynapse_fallback"] == conn.n_neurons
