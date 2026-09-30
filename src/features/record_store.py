"""RecordStore: normalised records of all sources + cached text representations.

All sources are stacked into one global row space (S1 rows first, then S2, then S3) so every
pair is simply (global_row_a, global_row_b). Vectorisers are fitted once per dataset on the
union of its record texts. This is unsupervised (no labels are touched) and is applied identically
to train and test, so it cannot leak ground truth; IDF statistics simply describe the corpus being
resolved (e.g. which tokens are common in *this* dataset's names).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from ..data.data_loader import ERDataset
from ..preprocessing.normalizer import Normalizer
from ..utils.logging import get_logger
from .text_vectors import binary_from_counts, hashed_counts, idf_vector, tfidf_from_counts

log = get_logger("record_store")
TOKEN = r"(?u)\b\w+\b"


@dataclass
class RecordStore:
    records: pd.DataFrame                    # stacked normalised records, column `source` in {S1,S2,S3}
    ranges: dict[str, tuple[int, int]]       # source -> [start, end) global rows
    mats: dict[str, sp.csr_matrix] = field(default_factory=dict)
    dense: dict[str, np.ndarray] = field(default_factory=dict)
    idf: dict[str, np.ndarray] = field(default_factory=dict)
    name_freq: np.ndarray | None = None      # how many records share this exact name_core (chain signal)

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(cls, ds: ERDataset, normalizer: Normalizer, lsa_dims: int = 128, seed: int = 42) -> "RecordStore":
        parts, ranges, start = [], {}, 0
        for src, df in ds.sources.items():
            nd = normalizer.transform(df)
            nd.insert(0, "source", src)
            parts.append(nd)
            ranges[src] = (start, start + len(nd))
            start += len(nd)
        rec = pd.concat(parts, ignore_index=True)
        store = cls(rec, ranges)
        store._fit(lsa_dims, seed)
        return store

    def text(self, fld: str) -> list[str]:
        """A records column, or the virtual field 'name+addr'."""
        r = self.records
        if fld == "name+addr":
            return (r["name_core"] + " " + r["addr_core"]).tolist()
        return r[fld].tolist()

    def counts(self, fld: str, analyzer: str = "word", ngram=(1, 1), token_pattern: str | None = None) -> sp.csr_matrix:
        """Hashed n-gram counts of a field over ALL records, analysed once and cached."""
        key = f"{fld}|{analyzer}|{tuple(ngram)}|{token_pattern}"
        cache = self.__dict__.setdefault("_counts", {})
        if key not in cache:
            cache[key] = hashed_counts(self.text(fld), analyzer, tuple(ngram), token_pattern)
        return cache[key]

    def _fit(self, lsa_dims: int, seed: int) -> None:
        r = self.records
        C = self.counts
        # names
        self.mats["name_word"] = tfidf_from_counts(C("name_core"))
        self.mats["name_char"] = tfidf_from_counts(C("name_core", "char_wb", (2, 4)))
        # trade names share the name vocabulary / IDF (fitted on core names)
        alt = C("name_alt", "char_wb", (2, 4))
        base = C("name_core", "char_wb", (2, 4))
        n = base.shape[0]
        df = np.bincount(base.indices, minlength=base.shape[1]).astype(np.float64)
        idf = np.where(df > 0, np.log((1.0 + n) / (1.0 + df)) + 1.0, 0.0).astype(np.float32)
        A = alt.astype(np.float32, copy=True)
        A.data = 1.0 + np.log(A.data)
        self.mats["name_alt_char"] = normalize((A @ sp.diags(idf)).tocsr()).astype(np.float32).tocsr()
        self.mats["name_phon"] = tfidf_from_counts(C("name_phon", "char_wb", (2, 3)))
        self.mats["name_bin"] = binary_from_counts(C("name_core"))
        self.idf["name_bin"] = idf_vector(self.mats["name_bin"])
        self.idf["name_word"] = self.idf["name_bin"]
        # addresses
        self.mats["addr_word"] = tfidf_from_counts(C("addr_core"))
        self.mats["addr_char"] = tfidf_from_counts(C("addr_core", "char_wb", (3, 4)))
        self.mats["addr_phon"] = tfidf_from_counts(C("addr_phon", "char_wb", (2, 3)))
        self.mats["addr_bin"] = binary_from_counts(C("addr_core"))
        self.idf["addr_bin"] = idf_vector(self.mats["addr_bin"])
        self.mats["addr_num"] = binary_from_counts(C("addr_nums", "word", (1, 1), r"\d+"))
        # "semantic" dense embeddings: LSA over character n-grams learned from this corpus. The SVD basis is
        # fitted on a row sample (cost independent of corpus size) and applied to every record.
        combo = tfidf_from_counts(C("name+addr", "char_wb", (2, 4)), min_df=2)
        rng = np.random.RandomState(seed)
        for key, M in (("lsa_combo", combo), ("lsa_name", self.mats["name_char"])):
            nnz_cols = np.unique(M.indices)
            M = M[:, nnz_cols] if len(nnz_cols) else M
            k = int(min(lsa_dims, max(2, M.shape[1] - 1), max(2, M.shape[0] - 1)))
            if M.shape[1] > 2 and M.shape[0] > 2:
                fit_rows = rng.choice(M.shape[0], size=min(M.shape[0], 60_000), replace=False)
                svd = TruncatedSVD(n_components=k, random_state=seed, algorithm="randomized", n_iter=4).fit(M[fit_rows])
                E = svd.transform(M).astype(np.float32)
            else:
                E = np.zeros((M.shape[0], 2), dtype=np.float32)
            self.dense[key] = normalize(E).astype(np.float32)
        counts = r.groupby("name_core")["name_core"].transform("size").to_numpy()
        self.name_freq = counts.astype(np.float32)
        log.info("record store built", extra={"data": {"records": len(r), "name_terms": int(len(np.unique(self.mats["name_word"].indices))),
                                                        "addr_terms": int(len(np.unique(self.mats["addr_word"].indices)))}})

    # ------------------------------------------------------------------ helpers
    def codes(self, column: str, max_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cached (codes, reversed codes, lengths) encodings of a text column, one row per record."""
        key = f"{column}:{max_len}"
        cache = self.__dict__.setdefault("_code_cache", {})
        if key not in cache:
            from .similarity import encode_strings

            vals = [s[:max_len] for s in self.records[column].tolist()]
            c, lens = encode_strings(vals, max_len)
            rc, _ = encode_strings([s[::-1] for s in vals], max_len)
            cache[key] = (c, rc, lens)
        return cache[key]

    def rows(self, source: str) -> np.ndarray:
        a, b = self.ranges[source]
        return np.arange(a, b)

    def ids(self) -> np.ndarray:
        return self.records["entity_id"].to_numpy()

    def source_frame(self, source: str) -> pd.DataFrame:
        a, b = self.ranges[source]
        return self.records.iloc[a:b]

    def id_to_row(self) -> dict[str, int]:
        return {e: k for k, e in enumerate(self.records["entity_id"].tolist())}
