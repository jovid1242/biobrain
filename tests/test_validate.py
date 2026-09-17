import json

from biobrain.connectome import preprocess, validate


def statuses(result):
    return {c["id"]: c["status"] for c in result["checks"]}


def test_validation_on_toy_release_flags_what_is_wrong(toy_catalog, budget, tmp_path):
    preprocess.run(toy_catalog, budget=budget, log=lambda *_: None)
    result = validate.run(toy_catalog, budget=budget, out_dir=tmp_path / "out", log=lambda *_: None)
    s = statuses(result)
    # structural invariants of the store hold
    assert all(s[k] == "PASS" for k in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "B2", "C5", "G2", "G3", "F1d", "F2d"))
    # the planted fragment row is caught twice: directly, and by the independent re-derivation
    assert s["C2"] == "FAIL" and s["D8"] == "FAIL"
    # the toy is not FlyWire: published totals must not "pass"
    assert s["B1"] == "FAIL" and s["D1"] == "FAIL"
    assert s["G1"] == "WARN"
    assert result["status"] == "FAIL"
    saved = json.loads((tmp_path / "out" / "data_validation.json").read_text())
    assert saved["summary"] == result["summary"]
    assert "| E4 | PASS |" in (tmp_path / "out" / "data_validation.md").read_text()


def test_independent_totals_match_bruteforce(toy_catalog, budget):
    totals = validate.independent_totals(toy_catalog.local_path("connections"), budget)
    # 9 rows -> pairs: (0,1) (1,0) (2,2) (3,4) (4,5) (5,3) (fragment,3)
    assert totals == {"rows": 9, "pairs": 7, "synapses": 48, "pairs_ge_5": 4, "neurons_in_pairs_ge_5": 6,
                      "self_connection_pairs": 1}
