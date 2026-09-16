import numpy as np

from biobrain.analysis.graph import Graph
from biobrain.benchmarks.varint import DeltaVarintCSR, varint_decode, varint_encode


def test_varint_roundtrip_all_byte_lengths():
    values = np.array([0, 1, 127, 128, 16383, 16384, 2**21 - 1, 2**21, 2**28, 2**32 - 1], dtype=np.uint64)
    stream, nbytes = varint_encode(values)
    assert nbytes.tolist() == [1, 1, 1, 2, 2, 3, 3, 4, 5, 5]
    assert np.array_equal(varint_decode(stream), values)


def test_delta_varint_csr_random_rows_match_plain_csr():
    rng = np.random.default_rng(0)
    n = 500
    src, dst = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    g = Graph.from_edges(n, src, dst, rng.integers(1, 300, 20000))
    enc = DeltaVarintCSR(g.indptr, g.indices, g.weights)
    rows = np.array([0, 17, 3, 499, 250, 250])
    ids, w = enc.rows(rows)
    want_ids = np.concatenate([g.indices[g.indptr[r]:g.indptr[r + 1]] for r in rows])
    want_w = np.concatenate([g.weights[g.indptr[r]:g.indptr[r + 1]] for r in rows])
    assert np.array_equal(ids, want_ids) and np.array_equal(w, want_w)
    empty = np.flatnonzero(np.diff(g.indptr) == 0)
    if empty.size:
        ids, _ = enc.rows(np.r_[empty[:1], 5])
        assert np.array_equal(ids, g.indices[g.indptr[5]:g.indptr[6]])
    assert enc.nbytes < g.indices.nbytes + g.weights.nbytes
