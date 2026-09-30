"""TF-IDF top-k retrieval blockers (word or character n-gram), optionally country-partitioned."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.text_vectors import tfidf_from_counts
from .base import Blocker, country_partitions, empty_pairs, sparse_topk

TOKEN = r"(?u)\b\w+\b"


class TfidfTopKBlocker(Blocker):
    """Generic top-k cosine retrieval over a text field.

    params: field (records column or 'name+addr'), analyzer ('word'|'char_wb'), ngram, k, min_score,
            max_df (drops ubiquitous terms so the sparse product stays sparse), partition (bool override).
    """

    def __init__(self, name: str, field: str, analyzer: str = "word", enabled: bool = True, **params) -> None:
        super().__init__(enabled, **params)
        self.name = name
        self.field = field
        self.analyzer = analyzer

    def pairs(self, store, target, partition_by_country, chunk):
        a_all, b_all = store.rows("S1"), store.rows(target)
        rows = np.concatenate([a_all, b_all])
        ngram = (1, 1) if self.analyzer == "word" else tuple(self.params.get("ngram", (3, 3)))
        counts = store.counts(self.field, self.analyzer, ngram)  # analysed once per dataset, shared
        # IDF / max_df on the S1 ∪ target sub-corpus, as if a vectoriser were fitted on it
        # max_df bounds common terms proportionally; max_df_abs caps them absolutely so per-record work stays
        # constant as the corpus grows (otherwise sparse products become quadratic at 100K+ records)
        max_df = float(self.params.get("max_df", 1.0))
        cap = self.params.get("max_df_abs")
        if cap:
            max_df = min(max_df, float(cap) / max(1, len(rows)))
        X = tfidf_from_counts(counts, rows, max_df=max_df)
        if X.nnz == 0:
            return empty_pairs()
        pos = {g: p for p, g in enumerate(rows)}
        k = int(self.params.get("k", 10))
        ms = float(self.params.get("min_score", 0.0))
        part = self.params.get("partition", partition_by_country)
        out = []
        for a, b in country_partitions(store, target, part):
            A = X[[pos[g] for g in a]]
            B = X[[pos[g] for g in b]]
            ri, ci, v = sparse_topk(A, B, k, ms, chunk)
            if len(ri):
                out.append(pd.DataFrame({"i": a[ri], "j": b[ci], "score": v}))
        return pd.concat(out, ignore_index=True) if out else empty_pairs()
