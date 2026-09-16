"""Vectorized LEB128 varint + per-row delta coding of sorted CSR neighbour lists (benchmark candidate)."""

from __future__ import annotations

import numpy as np


def varint_encode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unsigned values (< 2**35) -> (byte stream, byte length per value)."""
    v = np.asarray(values, dtype=np.uint64)
    nbytes = np.ones(v.size, dtype=np.int64)
    for k in range(1, 5):
        nbytes += v >= (1 << (7 * k))
    pos = np.concatenate([[0], np.cumsum(nbytes)])
    out = np.zeros(int(pos[-1]), dtype=np.uint8)
    for b in range(5):
        sel = nbytes > b
        byte = ((v[sel] >> np.uint64(7 * b)) & np.uint64(0x7F)).astype(np.uint8)
        byte |= np.where(nbytes[sel] > b + 1, 0x80, 0).astype(np.uint8)
        out[pos[:-1][sel] + b] = byte
    return out, nbytes


def varint_decode(stream: np.ndarray) -> np.ndarray:
    if stream.size == 0:
        return np.zeros(0, dtype=np.uint64)
    ends = (stream & 0x80) == 0
    value_id = np.concatenate([[0], np.cumsum(ends)[:-1]])
    first_byte = np.flatnonzero(np.concatenate([[True], ends[:-1]]))
    shift = ((np.arange(stream.size) - first_byte[value_id]) * 7).astype(np.uint64)
    parts = (stream & 0x7F).astype(np.uint64) << shift
    return np.bincount(value_id, weights=parts.astype(np.float64), minlength=int(ends.sum())).astype(np.uint64)


class DeltaVarintCSR:
    """Sorted neighbour ids stored as per-row deltas in varint bytes; weights as a second varint stream."""

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray):
        deltas = np.diff(indices.astype(np.int64), prepend=0)
        starts = indptr[:-1][np.diff(indptr) > 0]
        deltas[starts] = indices[starts]  # the first neighbour of each row is stored absolute
        self.idx_stream, idx_len = varint_encode(deltas)
        self.w_stream, w_len = varint_encode(weights)
        self.idx_ptr = np.concatenate([[0], np.cumsum(idx_len)])[indptr]
        self.w_ptr = np.concatenate([[0], np.cumsum(w_len)])[indptr]
        self.indptr = np.asarray(indptr)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.idx_stream, self.w_stream, self.idx_ptr, self.w_ptr, self.indptr))

    @staticmethod
    def _gather(stream: np.ndarray, ptr: np.ndarray, rows: np.ndarray) -> np.ndarray:
        lo, lengths = ptr[rows], ptr[rows + 1] - ptr[rows]
        total = int(lengths.sum())
        offsets = np.repeat(lo - (np.cumsum(lengths) - lengths), lengths) + np.arange(total)
        return varint_decode(stream[offsets])

    def rows(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Neighbour ids and weights of the given rows, concatenated in row order."""
        counts = self.indptr[rows + 1] - self.indptr[rows]
        cs = np.cumsum(self._gather(self.idx_stream, self.idx_ptr, rows).astype(np.int64))
        if cs.size == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.uint64)
        starts = np.cumsum(counts) - counts
        base = np.where(starts > 0, cs[np.maximum(starts - 1, 0)], 0)
        return (cs - np.repeat(base, counts)).astype(np.int32), self._gather(self.w_stream, self.w_ptr, rows)
