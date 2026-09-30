"""Fast text vectorisation: parallel feature hashing + TF-IDF derived from cached counts.

sklearn's n-gram analysers are pure Python; with 7+ vectorisations per dataset (features and
blockers) they dominated preparation time at 50K+ records. Here each (field, analyser, n-gram) is
analysed ONCE, in parallel chunks (HashingVectorizer is stateless, so chunks are independent), and every
TF-IDF / binary / max_df-pruned variant is derived from those counts with sparse linear algebra.
2**20 hash buckets make collisions negligible for similarity purposes.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

TOKEN = r"(?u)\b\w+\b"
N_FEATURES = 2 ** 20
PARALLEL_MIN = 40_000


def _vectorizer(analyzer: str, ngram: tuple[int, int], token_pattern: str | None) -> HashingVectorizer:
    kw = dict(n_features=N_FEATURES, alternate_sign=False, norm=None, lowercase=False, dtype=np.float32)
    if analyzer == "word":
        return HashingVectorizer(analyzer="word", token_pattern=token_pattern or TOKEN, ngram_range=ngram, **kw)
    return HashingVectorizer(analyzer=analyzer, ngram_range=ngram, **kw)


def hashed_counts(texts: list[str], analyzer: str = "word", ngram: tuple[int, int] = (1, 1),
                  token_pattern: str | None = None, n_jobs: int = -1, chunk: int = 25_000) -> sp.csr_matrix:
    hv = _vectorizer(analyzer, tuple(ngram), token_pattern)
    if len(texts) < PARALLEL_MIN or n_jobs == 1:
        return hv.transform(texts).tocsr()
    from joblib import Parallel, delayed

    parts = Parallel(n_jobs=n_jobs, backend="loky")(delayed(hv.transform)(texts[s:s + chunk])
                                                    for s in range(0, len(texts), chunk))
    return sp.vstack(parts).tocsr()


def tfidf_from_counts(X: sp.csr_matrix, rows: np.ndarray | None = None, max_df: float = 1.0, min_df: int = 1,
                      sublinear: bool = True) -> sp.csr_matrix:
    """L2-normalised TF-IDF over `rows` of a count matrix, IDF computed on those rows (like fitting a
    TfidfVectorizer on that sub-corpus). Terms outside [min_df, max_df*n] are dropped."""
    Xs = X if rows is None else X[rows]
    n = Xs.shape[0]
    if n == 0:
        return Xs.astype(np.float32)
    df = np.bincount(Xs.indices, minlength=X.shape[1]).astype(np.float64)
    keep = (df >= min_df) & ((df <= max_df * n) if max_df < 1.0 else (df > 0))
    idf = np.where(keep, np.log((1.0 + n) / (1.0 + df)) + 1.0, 0.0).astype(np.float32)
    Y = Xs.astype(np.float32, copy=True)
    if sublinear:
        Y.data = 1.0 + np.log(Y.data)
    Y = (Y @ sp.diags(idf)).tocsr()
    Y.eliminate_zeros()
    return normalize(Y).astype(np.float32).tocsr()


def binary_from_counts(X: sp.csr_matrix) -> sp.csr_matrix:
    B = X.copy().tocsr()
    B.data = np.ones_like(B.data, dtype=np.float32)
    return B


def idf_vector(B: sp.csr_matrix) -> np.ndarray:
    n = B.shape[0]
    df = np.bincount(B.indices, minlength=B.shape[1]).astype(np.float64)
    return (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
