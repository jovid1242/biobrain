import pytest

from biobrain.benchmarks import memory
from biobrain.connectome import preprocess


@pytest.mark.parametrize("case", ["baseline", "python_slots_objects", "coo_compact", "csr_csc", "store_mmap", "store_eager",
                                  "delta_varint", "zstd_csr", "weights_quantized", "region_local_ids"])
def test_worker_cases_run_and_report_sizes(synthetic_catalog, budget, case):
    preprocess.run(synthetic_catalog, budget=budget, log=lambda *_: None)
    res = memory.worker(case, "toy", seed=1)
    assert res["case"] == case and res["rss_after_build_bytes"] is not None
    if case in ("csr_csc", "store_eager", "delta_varint"):
        assert res["access"]["synaptic_events_per_pass"] > 0 and res["bytes_per_edge"] > 0
    if case == "weights_quantized":
        assert res["variants"]["uint16"]["lossless"]
