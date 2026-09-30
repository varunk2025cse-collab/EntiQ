"""Blocker interface + shared retrieval primitives (sparse / dense top-k, key joins)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..features.record_store import RecordStore

PAIR_COLS = ["i", "j", "score"]


def empty_pairs() -> pd.DataFrame:
    return pd.DataFrame({"i": np.array([], np.int64), "j": np.array([], np.int64), "score": np.array([], np.float32)})


def country_partitions(store: RecordStore, target: str, enabled: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """(s1_rows, target_rows) blocks. Countries are discovered from the data (open set).

    A record with a missing country is compared against every partition so it is never lost.
    """
    s1 = store.rows("S1")
    tg = store.rows(target)
    if not enabled:
        return [(s1, tg)]
    c = store.records["country_norm"].to_numpy()
    c1, ct = c[s1], c[tg]
    parts = []
    for country in sorted(set(c1) - {""}):
        a = s1[c1 == country]
        b = tg[(ct == country) | (ct == "")]
        if len(a) and len(b):
            parts.append((a, b))
    miss = s1[c1 == ""]
    if len(miss):
        parts.append((miss, tg))
    return parts


def sparse_topk(A: sp.csr_matrix, B: sp.csr_matrix, k: int, min_score: float, chunk: int = 2000):
    """Top-k columns of A @ B.T per row (exact sparse cosine retrieval, chunked)."""
    BT = B.T.tocsr()
    rows, cols, vals = [], [], []
    n_b = B.shape[0]
    chunk = max(100, min(chunk, int(4e7 // max(1, n_b))))  # bounds a densified chunk to ~160 MB
    for s in range(0, A.shape[0], chunk):
        C = (A[s:s + chunk] @ BT).tocsr()
        C.data[C.data < min_score] = 0
        C.eliminate_zeros()
        if C.nnz == 0:
            continue
        if C.nnz > 0.05 * C.shape[0] * n_b and n_b > k:
            # dense-ish chunk: linear-time selection per row (argpartition) beats sorting every nonzero
            D = C.toarray()
            top = np.argpartition(-D, k - 1, axis=1)[:, :k]
            v = np.take_along_axis(D, top, axis=1)
            keep = v > 0
            r_idx = np.broadcast_to(np.arange(D.shape[0])[:, None], top.shape)
            rows.append(r_idx[keep].astype(np.int64) + s)
            cols.append(top[keep].astype(np.int64))
            vals.append(v[keep].astype(np.float32))
            continue
        # sparse chunk: order entries by (row, -score) and keep the first k of each row
        r_ids = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
        order = np.lexsort((-C.data, r_ids))
        r_sorted = r_ids[order]
        first = np.searchsorted(r_sorted, r_sorted, side="left")
        keep = order[(np.arange(len(order)) - first) < k]
        rows.append(r_ids[keep].astype(np.int64) + s)
        cols.append(C.indices[keep])
        vals.append(C.data[keep])
    if not rows:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(rows), np.concatenate(cols).astype(np.int64), np.concatenate(vals).astype(np.float32)


def dense_topk(E1: np.ndarray, E2: np.ndarray, k: int, min_score: float, chunk: int = 2000):
    """Brute-force (BLAS) nearest neighbours on normalised dense embeddings."""
    k = min(k, E2.shape[0])
    rows, cols, vals = [], [], []
    for s in range(0, E1.shape[0], chunk):
        S = E1[s:s + chunk] @ E2.T
        top = np.argpartition(-S, k - 1, axis=1)[:, :k] if k < S.shape[1] else np.tile(np.arange(S.shape[1]), (S.shape[0], 1))
        v = np.take_along_axis(S, top, axis=1)
        keep = v >= min_score
        r = np.repeat(np.arange(s, s + S.shape[0]), top.shape[1]).reshape(top.shape)
        rows.append(r[keep])
        cols.append(top[keep])
        vals.append(v[keep])
    if not rows:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return (np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64),
            np.concatenate(vals).astype(np.float32))


def key_join(left_rows: np.ndarray, left_keys, right_rows: np.ndarray, right_keys, max_group: int) -> pd.DataFrame:
    """Pairs of rows sharing a non-empty key; keys that are too common are skipped (stop-keys)."""
    L = pd.DataFrame({"i": left_rows, "key": left_keys})
    R = pd.DataFrame({"j": right_rows, "key": right_keys})
    L, R = L[L.key != ""], R[R.key != ""]
    rs = R.groupby("key").size()
    ls = L.groupby("key").size()
    ok = set(rs[rs <= max_group].index) & set(ls[ls <= max_group].index)
    m = L[L.key.isin(ok)].merge(R[R.key.isin(ok)], on="key")
    m["score"] = np.float32(1.0)
    return m[PAIR_COLS]


class Blocker(ABC):
    name: str = "blocker"

    def __init__(self, enabled: bool = True, **params) -> None:
        self.enabled = enabled
        self.params = params

    @abstractmethod
    def pairs(self, store: RecordStore, target: str, partition_by_country: bool, chunk: int) -> pd.DataFrame:
        """Return candidate pairs (global rows i in S1, j in target) with a retrieval score."""
