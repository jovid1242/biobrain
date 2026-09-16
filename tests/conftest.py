"""A tiny synthetic FlyWire-shaped release (same file formats and columns) for pipeline tests."""

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from biobrain.connectome.catalog import load_catalog
from biobrain.telemetry import MemoryBudget

NT = ("gaba", "ach", "glut", "oct", "ser", "da")
IDS = [720575940000000050, 720575940000000010, 720575940000000040,
       720575940000000020, 720575940000000060, 720575940000000030]  # unsorted on purpose
FRAGMENT = 720575940999999999  # a segment that is not a proofread neuron

# (pre, post, neuropil, syn_count, dominant transmitter index)
ROWS = [
    (IDS[0], IDS[1], "AL_L", 3, 0),
    (IDS[0], IDS[1], "LH_L", 7, 1),     # same pair, second neuropil, different transmitter profile
    (IDS[1], IDS[0], "LH_L", 2, 0),     # reciprocal edge
    (IDS[2], IDS[2], "MB_CA_L", 4, 2),  # self-connection
    (IDS[3], IDS[4], "FB", 12, 1),
    (IDS[4], IDS[5], None, 1, 3),       # unassigned neuropil
    (IDS[5], IDS[3], "FB", 5, 0),
    (FRAGMENT, IDS[3], "FB", 9, 1),     # pre is not a proofread neuron -> dropped, counted
]


def nt_probs(dominant: int) -> list[float]:
    p = np.full(6, 0.04)
    p[dominant] = 0.8
    return list(p)


def write_release(raw):
    raw.mkdir(parents=True)
    np.save(raw / "proofread_root_ids_783.npy", np.array(IDS, dtype=np.uint64))
    cols = {"pre_pt_root_id": [], "post_pt_root_id": [], "neuropil": [], "syn_count": []}
    cols.update({f"{t}_avg": [] for t in NT})
    for pre, post, neuropil, syn, dom in ROWS:
        cols["pre_pt_root_id"].append(pre)
        cols["post_pt_root_id"].append(post)
        cols["neuropil"].append(neuropil)
        cols["syn_count"].append(syn)
        for t, p in zip(NT, nt_probs(dom)):
            cols[f"{t}_avg"].append(p)
    feather.write_feather(pa.table(cols), raw / "proofread_connections_783.feather", compression="zstd")
    # all-partner totals are >= the synapses each neuron has inside the proofread graph
    feather.write_feather(pa.table({"pre_pt_root_id": [IDS[0], IDS[0], FRAGMENT, IDS[3], IDS[1], IDS[2], IDS[4], IDS[5]],
                                    "neuropil": ["AL_L", "LH_L", "FB", None, "LH_L", "MB_CA_L", "FB", "FB"],
                                    "count": [30, 7, 100, 12, 2, 4, 1, 5]}),
                          raw / "per_neuron_neuropil_count_pre_783.feather", chunksize=3)
    feather.write_feather(pa.table({"post_pt_root_id": [IDS[1], FRAGMENT, IDS[4], IDS[1], IDS[0], IDS[2], IDS[5], IDS[3]],
                                    "neuropil": ["LH_L", "FB", "FB", "AL_L", "LH_L", "MB_CA_L", "FB", "FB"],
                                    "count": [10, 50, 12, 3, 2, 4, 1, 14]}),
                          raw / "per_neuron_neuropil_count_post_783.feather")
    ann = raw / "annotations" / "v3.1.0"
    ann.mkdir(parents=True)
    header = ["supervoxel_id", "root_id", "pos_x", "soma_x", "nucleus_id", "flow", "super_class", "cell_type", "top_nt", "top_nt_conf", "vfb_id"]
    lines = ["\t".join(header)]
    # IDS[5] deliberately has no annotation row
    for k, rid in enumerate(IDS[:5]):
        lines.append("\t".join([str(78112261444987077 + k), str(rid), f"{100.5 + k}", "" if k == 2 else str(10 + k),
                                "" if k == 2 else str(2453924 + k), "intrinsic", ["central", "optic", "central", "sensory", "optic"][k],
                                "" if k == 4 else f"T{k % 2}", "acetylcholine", "0.9", f"fw{k}"]))
    (ann / "Supplemental_file1_neuron_annotations.tsv").write_text("\n".join(lines) + "\n")


CATALOG = """
[dataset]
id = "toy"
title = "toy release"
version = "783"
license = "test"
{files}
"""
FILES = [("root_ids", "proofread_root_ids_783.npy", ""), ("connections", "proofread_connections_783.feather", ""),
         ("neuropil_counts_pre", "per_neuron_neuropil_count_pre_783.feather", ""),
         ("neuropil_counts_post", "per_neuron_neuropil_count_post_783.feather", ""),
         ("annotations", "Supplemental_file1_neuron_annotations.tsv", "annotations/v3.1.0")]


def make_catalog(tmp_path, monkeypatch, writer):
    data = tmp_path / "data"
    monkeypatch.setenv("BIOBRAIN_DATA_DIR", str(data))
    monkeypatch.setenv("BIOBRAIN_RESULTS_DIR", str(tmp_path / "results"))
    writer(data / "raw" / "toy")
    entries = "".join(f'\n[[files]]\nid = "{i}"\nfilename = "{f}"\nsubdir = "{s}"\nurl = "file:///dev/null"\nsize = 1\npurpose = "t"\n'
                      for i, f, s in FILES)
    toml = tmp_path / "toy.toml"
    toml.write_text(CATALOG.format(files=entries))
    return load_catalog(toml)


@pytest.fixture
def toy_catalog(tmp_path, monkeypatch):
    return make_catalog(tmp_path, monkeypatch, write_release)


@pytest.fixture
def synthetic_catalog(tmp_path, monkeypatch):
    from synthetic import write_synthetic_release

    return make_catalog(tmp_path, monkeypatch, lambda raw: write_synthetic_release(raw, n=3000, seed=0))


@pytest.fixture
def budget():
    return MemoryBudget.from_env("64GB")
