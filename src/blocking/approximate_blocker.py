"""Dense-embedding retrieval (LSA over character n-grams learned from the corpus)."""
from __future__ import annotations

import pandas as pd

from .base import Blocker, country_partitions, dense_topk, empty_pairs


class DenseEmbeddingBlocker(Blocker):
    name = "semantic_lsa"

    def __init__(self, name: str = "semantic_lsa", key: str = "lsa_combo", enabled: bool = True, **params) -> None:
        super().__init__(enabled, **params)
        self.name = name
        self.key = key

    def pairs(self, store, target, partition_by_country, chunk):
        E = store.dense[self.key]
        k = int(self.params.get("k", 10))
        ms = float(self.params.get("min_score", 0.0))
        out = []
        for a, b in country_partitions(store, target, partition_by_country):
            ri, ci, v = dense_topk(E[a], E[b], k, ms, chunk)
            if len(ri):
                out.append(pd.DataFrame({"i": a[ri], "j": b[ci], "score": v}))
        return pd.concat(out, ignore_index=True) if out else empty_pairs()
