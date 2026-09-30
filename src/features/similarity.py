"""Vectorised pairwise similarity kernels (numpy / scipy only).

All functions take aligned index arrays (i, j) and return one value per pair, so millions of
candidate pairs are scored without Python-level loops over pairs.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


# ----------------------------------------------------------------------------- sparse / dense
def rowwise_dot(A: sp.csr_matrix, B: sp.csr_matrix, i: np.ndarray, j: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    """sum_k A[i,k] * B[j,k] for each pair (cosine when rows are L2-normalised)."""
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        a = A[i[s:s + chunk]]
        b = B[j[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def dense_rowwise_dot(E1: np.ndarray, E2: np.ndarray, i: np.ndarray, j: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", E1[i[s:s + chunk]], E2[j[s:s + chunk]])
    return out


def set_overlap(Abin: sp.csr_matrix, Bbin: sp.csr_matrix, i: np.ndarray, j: np.ndarray,
                weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Token-set statistics from binary incidence matrices (optionally IDF-weighted)."""
    if weights is not None:
        W = sp.diags(weights.astype(np.float32))
        Aw, Bw = (Abin @ W).tocsr(), (Bbin @ W).tocsr()
        inter = rowwise_dot(Aw, Bbin, i, j)
        na = np.asarray(Aw.sum(axis=1)).ravel()[i]
        nb = np.asarray(Bw.sum(axis=1)).ravel()[j]
    else:
        inter = rowwise_dot(Abin, Bbin, i, j)
        na = np.diff(Abin.indptr)[i].astype(np.float32)
        nb = np.diff(Bbin.indptr)[j].astype(np.float32)
    union = na + nb - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
        ovl = np.where(np.minimum(na, nb) > 0, inter / np.minimum(na, nb), 0.0)
        cov_a = np.where(na > 0, inter / na, 0.0)
        cov_b = np.where(nb > 0, inter / nb, 0.0)
    return {"inter": inter.astype(np.float32), "jaccard": jac.astype(np.float32), "overlap": ovl.astype(np.float32),
            "cov_a": cov_a.astype(np.float32), "cov_b": cov_b.astype(np.float32),
            "na": na.astype(np.float32), "nb": nb.astype(np.float32)}


# ----------------------------------------------------------------------------- character level
def encode_strings(strings, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode strings to a padded (n, max_len) uint32 code-point array + lengths."""
    n = len(strings)
    codes = np.zeros((n, max_len), dtype=np.uint32)
    lens = np.zeros(n, dtype=np.int32)
    for r, s in enumerate(strings):
        s = s[:max_len]
        if s:
            codes[r, : len(s)] = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)
            lens[r] = len(s)
    return codes, lens


def levenshtein_batch(ac: np.ndarray, al: np.ndarray, bc: np.ndarray, bl: np.ndarray) -> np.ndarray:
    """Exact Levenshtein distance for aligned pairs of encoded strings.

    Row recurrence D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+cost). The in-row insertion
    dependency is resolved with a cumulative minimum:  D[i][j] = j + cummin_k(v_k) where
    v_0 = i and v_k = min(D[i-1][k-1]+cost_k, D[i-1][k]+1) - k.  Cost: O(La) vectorised steps.
    """
    n, La = ac.shape
    Lb = bc.shape[1]
    ar = np.arange(Lb + 1, dtype=np.int16)
    prev = np.broadcast_to(ar, (n, Lb + 1)).copy()
    result = prev[np.arange(n), bl].astype(np.int32)  # rows with la == 0
    for i in range(1, La + 1):
        if not (al >= i).any():
            break
        cost = (ac[:, i - 1][:, None] != bc).view(np.int8)  # (n, Lb) 0/1
        best = np.minimum(prev[:, :-1] + cost, prev[:, 1:] + 1)  # (n, Lb) for j = 1..Lb
        v = np.empty((n, Lb + 1), dtype=np.int16)
        v[:, 0] = i
        v[:, 1:] = best - ar[1:]
        cur = np.minimum.accumulate(v, axis=1) + ar
        done = al == i
        if done.any():
            result[done] = cur[done, bl[done]]
        prev = cur
    return result


def levenshtein_pairs(codes: np.ndarray, lens: np.ndarray, i: np.ndarray, j: np.ndarray,
                      chunk: int = 60_000) -> tuple[np.ndarray, np.ndarray]:
    """Levenshtein for pairs of *pre-encoded* records (codes: (n_records, L) cached per record).

    Pairs are ordered by the shorter string's length and processed in chunks trimmed to the
    chunk's own maximum lengths, so the O(La) loop never iterates over padding.
    Returns (distance, normalised similarity = 1 - d / max_len).
    """
    li, lj = lens[i], lens[j]
    swap = li > lj  # put the shorter string on the loop axis
    a = np.where(swap, j, i)
    b = np.where(swap, i, j)
    la, lb = lens[a], lens[b]
    order = np.lexsort((lb, la))
    dist = np.zeros(len(i), dtype=np.float32)
    for s in range(0, len(order), chunk):
        idx = order[s:s + chunk]
        A_len, B_len = la[idx], lb[idx]
        La, Lb = int(A_len.max(initial=0)), int(B_len.max(initial=0))
        if Lb == 0:
            continue
        d = levenshtein_batch(codes[a[idx], :max(La, 1)], A_len, codes[b[idx], :Lb], B_len)
        dist[idx] = d
    m = np.maximum(li, lj).astype(np.float32)
    sim = np.where(m > 0, 1.0 - dist / np.maximum(m, 1), 0.0).astype(np.float32)
    return dist, sim


def levenshtein_sim(a_strings, b_strings, max_len: int = 48, chunk: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """Return (distance, normalised similarity) for aligned string lists."""
    codes_a, la = encode_strings(list(a_strings), max_len)
    codes_b, lb = encode_strings(list(b_strings), max_len)
    codes = np.vstack([codes_a, codes_b])
    lens = np.concatenate([la, lb])
    n = len(la)
    return levenshtein_pairs(codes, lens, np.arange(n), np.arange(n, 2 * n))


def prefix_suffix_pairs(codes: np.ndarray, rcodes: np.ndarray, lens: np.ndarray, i: np.ndarray, j: np.ndarray,
                        chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Common prefix / suffix length / shorter length, from cached forward & reversed encodings."""
    p = np.zeros(len(i), np.float32)
    q = np.zeros(len(i), np.float32)
    L = codes.shape[1]
    pos = np.arange(L)[None, :]
    for s in range(0, len(i), chunk):
        a, b = i[s:s + chunk], j[s:s + chunk]
        mn = np.minimum(lens[a], lens[b])
        valid = pos < mn[:, None]
        pre = np.cumprod((codes[a] == codes[b]) & valid, axis=1).sum(axis=1)
        suf = np.cumprod((rcodes[a] == rcodes[b]) & valid, axis=1).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p[s:s + chunk] = np.where(mn > 0, pre / np.maximum(mn, 1), 0.0)
            q[s:s + chunk] = np.where(mn > 0, suf / np.maximum(mn, 1), 0.0)
    return p, q


def prefix_suffix_sim(a_strings, b_strings, max_len: int = 48) -> tuple[np.ndarray, np.ndarray]:
    a_strings, b_strings = list(a_strings), list(b_strings)
    codes, lens = encode_strings(a_strings + b_strings, max_len)
    rcodes, _ = encode_strings([s[:max_len][::-1] for s in a_strings + b_strings], max_len)
    n = len(a_strings)
    return prefix_suffix_pairs(codes, rcodes, lens, np.arange(n), np.arange(n, 2 * n))


# ----------------------------------------------------------------------------- scalar reference implementations
def levenshtein(a: str, b: str) -> int:
    """Reference implementation used by tests and the explanation UI."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def jaro_winkler(a: str, b: str, p: float = 0.1) -> float:
    """Scalar Jaro-Winkler (explanations / diagnostics)."""
    if a == b:
        return 1.0 if a else 0.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    w = max(la, lb) // 2 - 1
    ma, mb = [False] * la, [False] * lb
    m = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - w), min(lb, i + w + 1)):
            if not mb[j] and b[j] == ca:
                ma[i] = mb[j] = True
                m += 1
                break
    if not m:
        return 0.0
    t, k = 0, 0
    for i in range(la):
        if ma[i]:
            while not mb[k]:
                k += 1
            t += a[i] != b[k]
            k += 1
    jaro = (m / la + m / lb + (m - t / 2) / m) / 3
    pre = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        pre += 1
    return jaro + pre * p * (1 - jaro)
