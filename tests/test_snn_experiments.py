"""Experiment layer: provenance records, isolated runs, calibration, reproducibility (synthetic store)."""

import json

import numpy as np
import pytest

from biobrain import paths
from biobrain.connectome import preprocess
from biobrain.connectome.store import Connectome
from biobrain.snn import experiments, subgraph
from biobrain.snn.config import SimConfig


@pytest.fixture
def setup(synthetic_catalog, budget, monkeypatch):
    conn = Connectome.load(preprocess.run(synthetic_catalog, budget=budget, log=lambda *_: None))
    monkeypatch.setattr(subgraph, "CANONICAL", {"toy_400": {"method": "expand", "size": 400, "seed": 2}})
    subgraph.save(subgraph.build(conn, "toy_400", subgraph.CANONICAL["toy_400"]))
    return conn


def test_config_roundtrip_and_hashes():
    cfg = SimConfig().replace(weights={"gain": 0.3, "transform": "sqrt"}, run={"mode": "event_driven", "seed": 4})
    again = SimConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg and again.digest() == cfg.digest()
    assert cfg.model_hash == cfg.replace(run={"mode": "time_step"}).model_hash
    assert cfg.digest() != cfg.replace(run={"mode": "time_step"}).digest()
    with pytest.raises(ValueError):
        SimConfig().replace(neuron={"v_reset": 1.5})


def test_prepared_isolated_run_has_full_provenance_and_is_reproducible(setup):
    conn = setup
    cfg = SimConfig().replace(weights={"gain": 0.2}, inputs={"rate": 0.02}, run={"steps": 200, "seed": 3})
    path = experiments.prepare(conn, "toy_400", cfg)
    records = [experiments.run_isolated(path, cfg.replace(run={"mode": mode})) for mode in ("time_step", "event_driven", "time_step")]
    ts, ed, ts_again = records
    for key in ("experiment_id", "git_commit", "timestamp_utc"):
        assert key in ts or key in ts["run"]
    assert ts["dataset"]["store_manifest_sha256"] and ts["subgraph"]["sha256"]["edges"]
    assert not any(str(paths.project_root()) in arg for arg in ts["run"]["command"])  # no machine-specific paths
    assert ts["config_hash"] == cfg.replace(run={"mode": "time_step"}).digest() and ts["seed"] == 3
    assert ts["process"]["peak_rss"] > 0 and ts["memory"]["topology"] > 0
    for key in ("spikes", "synaptic_events", "neuron_updates", "external_events"):
        assert ts["metrics"][key] == ts_again["metrics"][key]  # same config + seed -> same counters
    assert ts["metrics"]["spikes"] == ed["metrics"]["spikes"] and ts["metrics"]["synaptic_events"] == ed["metrics"]["synaptic_events"]
    assert ed["metrics"]["neuron_updates"] <= ts["metrics"]["neuron_updates"]


def test_calibration_reports_regimes_and_a_rule_based_baseline(setup):
    conn = setup
    result = experiments.calibrate(conn, "toy_400", SimConfig(), gains=(0.001, 0.05, 0.3, 5.0, 50.0), seeds=(1, 2),
                                   on_steps=100, off_steps=100, log=lambda *_: None)
    regimes = {r["gain"]: r["regime"] for r in result["rows"]}
    assert regimes[0.001] == "DEAD" and regimes[50.0] == "SATURATED"
    if result["baseline_gain"] is not None:
        lo, hi = result["widest_stable_range"]
        assert lo <= result["baseline_gain"] <= hi


def test_benchmark_matrix_is_resumable(setup, tmp_path, monkeypatch):
    conn = setup
    monkeypatch.setattr(experiments, "results_dir", lambda: tmp_path)
    kw = dict(subgraphs={"toy_400": 0.1}, rates=(0.01,), seeds=(1,), steps=100, base=SimConfig(), log=lambda *_: None)
    out = experiments.benchmark(conn, "mini", **kw)
    first = experiments.load_jsonl(out)
    assert len(first) == 3 and {r["mode"] for r in first} == {"time_step", "event_driven"}
    experiments.benchmark(conn, "mini", **kw)
    assert len(experiments.load_jsonl(out)) == 3


def test_null_controls_and_equivalence_helpers(setup):
    conn = setup
    base = SimConfig().replace(weights={"gain": 0.1})
    nulls = experiments.null_controls(conn, "toy_400", base, seeds=(1,), on_steps=80, off_steps=40, log=lambda *_: None)
    assert {r["topology"] for r in nulls["rows"]} == {"real", "degree_preserving", "reciprocity_preserving"}
    assert all(r["equivalence"]["equivalent"] for r in nulls["rows"])
    rows = experiments.equivalence(conn, "toy_400", base, rates=(0.005, 0.05), steps=150)
    assert all(r["equivalent"] for r in rows) and len(rows) == 8


def test_os_energy_estimate_is_labelled_and_never_claimed_as_measured():
    from biobrain.snn import energy

    before = energy.snapshot()
    sum(i * i for i in range(200_000))
    after = energy.snapshot()
    result = energy.delta(before, after, wall_s=0.01)
    if before is None:  # not macOS: the estimate is simply absent
        assert result is None
    else:
        assert result["energy_nj"] >= 0 and result["cpu_time_s"] > 0
        assert "NOT a measurement" in result["label"] and result["reliable_window"] is False


def test_gains_ignore_variant_calibrations(tmp_path, monkeypatch):
    from biobrain.snn import pipeline

    monkeypatch.setattr(experiments, "results_dir", lambda: tmp_path)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "calibration_a.json").write_text(json.dumps({"subgraph": "a", "baseline_gain": 0.1}))
    (tmp_path / "experiments" / "calibration_variant_b.json").write_text(json.dumps({"subgraph": "b", "baseline_gain": 0.2, "variant": "x"}))
    assert pipeline.gains() == {"a": 0.1}
