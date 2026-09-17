"""A synthetic FlyWire-shaped release with planted neuropil communities and heavy-tailed degrees."""

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

NEUROPILS = ["AL_L", "AL_R", "LH_L", "LH_R", "MB_CA_L", "FB"]
SUPER = ["central", "optic", "sensory", "descending"]
NT = ("gaba", "ach", "glut", "oct", "ser", "da")


def write_synthetic_release(raw, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    raw.mkdir(parents=True, exist_ok=True)
    ids = np.sort(rng.choice(np.arange(720575940000000000, 720575940000000000 + 50 * n), size=n, replace=False))
    np.save(raw / "proofread_root_ids_783.npy", ids.astype(np.uint64))
    home = rng.integers(0, len(NEUROPILS), n)
    out_deg = np.minimum(np.round(rng.lognormal(2.5, 0.9, n)).astype(int), n // 3)
    pre, post = [], []
    for i in range(n):
        same = np.flatnonzero(home == home[i])
        k_in = int(out_deg[i] * 0.8)
        pre += [i] * out_deg[i]
        post += list(rng.choice(same, k_in)) + list(rng.integers(0, n, out_deg[i] - k_in))
    pre, post = np.array(pre), np.array(post)
    keep = pre != post
    pairs = np.unique(pre[keep] * n + post[keep])
    pre, post = pairs // n, pairs % n
    syn = 1 + rng.geometric(0.25, pairs.size)
    dominant = rng.integers(0, 6, n)
    rows = {"pre_pt_root_id": [], "post_pt_root_id": [], "neuropil": [], "syn_count": [], **{f"{t}_avg": [] for t in NT}}
    pre_counts, post_counts = {}, {}
    for a, b, s in zip(pre, post, syn):
        split = [(home[a], s)] if s < 4 or home[a] == home[b] or rng.random() < 0.7 else [(home[a], s - 2), (home[b], 2)]
        for np_code, count in split:
            name = NEUROPILS[np_code]
            rows["pre_pt_root_id"].append(ids[a])
            rows["post_pt_root_id"].append(ids[b])
            rows["neuropil"].append(name)
            rows["syn_count"].append(int(count))
            p = np.full(6, 0.04)
            p[dominant[a]] = 0.8
            for t, v in zip(NT, p):
                rows[f"{t}_avg"].append(float(v))
            pre_counts[(ids[a], name)] = pre_counts.get((ids[a], name), 0) + int(count)
            post_counts[(ids[b], name)] = post_counts.get((ids[b], name), 0) + int(count)
    feather.write_feather(pa.table(rows), raw / "proofread_connections_783.feather", compression="zstd")
    for side, counts in (("pre", pre_counts), ("post", post_counts)):
        keys = list(counts)
        feather.write_feather(pa.table({f"{side}_pt_root_id": [k[0] for k in keys] + [1],  # plus one fragment row
                                        "neuropil": [k[1] for k in keys] + ["FB"],
                                        "count": [counts[k] + int(rng.integers(0, 3)) for k in keys] + [7]}),
                              raw / f"per_neuron_neuropil_count_{side}_783.feather")
    ann = raw / "annotations" / "v3.1.0"
    ann.mkdir(parents=True, exist_ok=True)
    super_class = rng.choice(SUPER, n, p=[0.5, 0.35, 0.1, 0.05])
    flow = np.where(super_class == "sensory", "afferent", np.where(super_class == "descending", "efferent", "intrinsic"))
    transmitters = ["acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine", ""]
    top_nt = rng.choice(transmitters, n, p=[0.55, 0.15, 0.2, 0.04, 0.03, 0.01, 0.02])
    lines = ["supervoxel_id\troot_id\tpos_x\tsoma_x\tnucleus_id\tflow\tsuper_class\tcell_type\tside\ttop_nt"]
    for i in range(n - 2):  # the last two neurons have no annotation row
        lines.append(f"{i + 1}\t{ids[i]}\t{rng.random() * 1e5:.1f}\t{int(rng.integers(0, 1e5))}\t{i + 10}\t{flow[i]}\t"
                     f"{super_class[i]}\tT{int(rng.integers(0, 40))}\t{rng.choice(['left', 'right'])}\t{top_nt[i]}")
    (ann / "Supplemental_file1_neuron_annotations.tsv").write_text("\n".join(lines) + "\n")
    return ids, home
