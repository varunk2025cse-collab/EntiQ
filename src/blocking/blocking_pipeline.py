"""Multi-pass blocking: run every enabled blocker for S1->S2 and S1->S3 and UNION the results.

Output candidate frame (one row per unique pair):
  i, j        global record rows (i in S1, j in S2/S3)
  target      'S2' | 'S3'
  blk_mask    bitmask of blockers that retrieved the pair (bit order = BLOCKER_ORDER)
  blk_count   number of blockers that retrieved it
  bs_<name>   retrieval score of each blocker (0 when not retrieved)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..features.record_store import RecordStore
from ..utils.logging import get_logger
from .approximate_blocker import DenseEmbeddingBlocker
from .base import Blocker
from .exact_blocker import AcronymBlocker, ExactNameBlocker, PostalTokenBlocker
from .tfidf_blocker import TfidfTopKBlocker

log = get_logger("blocking")

BLOCKER_ORDER = ["exact_name", "acronym", "name_word_tfidf", "name_char_tfidf", "name_char_global",
                 "address_tfidf", "name_address_char", "postal_token", "semantic_lsa"]
BLOCKER_LABELS = {
    "exact_name": "Normalized name key", "acronym": "Acronym", "name_word_tfidf": "Name token TF-IDF",
    "name_char_tfidf": "Name n-gram TF-IDF", "name_char_global": "Cross-country name n-gram",
    "address_tfidf": "Address TF-IDF", "name_address_char": "Name+address n-gram",
    "postal_token": "Postal code + name token", "semantic_lsa": "Semantic embedding (LSA)",
}


def build_blocker(name: str, params: dict) -> Blocker:
    p = dict(params)
    enabled = bool(p.pop("enabled", True))
    if name == "exact_name":
        return ExactNameBlocker(enabled, **p)
    if name == "acronym":
        return AcronymBlocker(enabled, **p)
    if name == "postal_token":
        return PostalTokenBlocker(enabled, **p)
    if name == "name_word_tfidf":
        return TfidfTopKBlocker(name, "name_core", "word", enabled, **p)
    if name == "name_char_tfidf":
        return TfidfTopKBlocker(name, "name_core", "char_wb", enabled, **p)
    if name == "name_char_global":
        return TfidfTopKBlocker(name, "name_core", "char_wb", enabled, partition=False, **p)
    if name == "address_tfidf":
        return TfidfTopKBlocker(name, "addr_core", "word", enabled, **p)
    if name == "name_address_char":
        return TfidfTopKBlocker(name, "name+addr", "char_wb", enabled, **p)
    if name == "semantic_lsa":
        p.pop("dims", None)
        return DenseEmbeddingBlocker(name, "lsa_combo", enabled, **p)
    raise KeyError(f"unknown blocker {name}")


class BlockingPipeline:
    VERSION = "block-v2"

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.partition = bool(cfg.get("partition_by_country", True))
        self.chunk = int(cfg.get("chunk_size", 2000))
        self.blockers = [build_blocker(n, cfg["blockers"][n]) for n in BLOCKER_ORDER if n in cfg.get("blockers", {})]

    def run(self, store: RecordStore) -> tuple[pd.DataFrame, dict]:
        frames, timings = [], {}
        for target in ("S2", "S3"):
            if store.ranges[target][1] == store.ranges[target][0]:
                continue
            for b in self.blockers:
                if not b.enabled:
                    continue
                t0 = time.perf_counter()
                p = b.pairs(store, target, self.partition, self.chunk)
                timings[f"{b.name}:{target}"] = round(time.perf_counter() - t0, 3)
                if len(p):
                    frames.append((BLOCKER_ORDER.index(b.name), p["i"].to_numpy(np.int64), p["j"].to_numpy(np.int64),
                                   p["score"].to_numpy(np.float32)))
        nb = len(BLOCKER_ORDER)
        if not frames:
            cand = pd.DataFrame({"i": np.array([], np.int64), "j": np.array([], np.int64), "target": [],
                                 "blk_mask": np.array([], np.int64), "blk_count": np.array([], np.int16)})
            for n in BLOCKER_ORDER:
                cand[f"bs_{n}"] = np.array([], np.float32)
            return cand, {"timings": timings}
        # Vectorised union: integer pair key -> unique pairs, then scatter-max each blocker's score.
        n_rec = np.int64(len(store.records))
        keys = np.concatenate([f[1] * n_rec + f[2] for f in frames])
        blk = np.concatenate([np.full(len(f[1]), f[0], np.int8) for f in frames])
        score = np.concatenate([f[3] for f in frames])
        uniq, inv = np.unique(keys, return_inverse=True)   # sorted by (i, j)
        S = np.full((len(uniq), nb), -1.0, np.float32)
        np.maximum.at(S, (inv, blk), score)
        present = S >= 0
        S[~present] = 0.0
        i_arr, j_arr = uniq // n_rec, uniq % n_rec
        s3_lo = store.ranges["S3"][0]
        cand = pd.DataFrame({"i": i_arr, "j": j_arr,
                             "target": pd.Categorical.from_codes((j_arr >= s3_lo).astype(np.int8), ["S2", "S3"])})
        for b, n in enumerate(BLOCKER_ORDER):
            cand[f"bs_{n}"] = S[:, b]
        cand["blk_mask"] = (present * (1 << np.arange(nb))).sum(axis=1).astype(np.int64)
        cand["blk_count"] = present.sum(axis=1).astype(np.int16)
        log.info("blocking done", extra={"data": {"candidates": len(cand), "seconds": round(sum(timings.values()), 2)}})
        return cand, {"timings": timings}


def blocking_report(cand: pd.DataFrame, store: RecordStore, true_pairs: set[tuple[str, str]] | None) -> dict:
    """Per-blocker and union statistics: volume, reduction ratio, recall, unique recall."""
    ids = store.ids()
    n1 = store.ranges["S1"][1] - store.ranges["S1"][0]
    n_other = (store.ranges["S2"][1] - store.ranges["S2"][0]) + (store.ranges["S3"][1] - store.ranges["S3"][0])
    total_space = max(1, n1 * n_other)
    rep = {"union": {"candidates": int(len(cand)), "reduction_ratio": 1 - len(cand) / total_space,
                     "avg_candidates_per_s1": len(cand) / max(1, n1),
                     "s1_without_candidates": int(n1 - cand["i"].nunique()) if len(cand) else int(n1)},
           "blockers": {}}
    is_true = None
    if true_pairs is not None:
        key = pd.Series(list(zip(ids[cand["i"].to_numpy()], ids[cand["j"].to_numpy()]))) if len(cand) else pd.Series([], dtype=object)
        is_true = key.isin(true_pairs).to_numpy() if len(cand) else np.array([], bool)
        n_true = max(1, len(true_pairs))
        rep["union"]["true_pairs"] = len(true_pairs)
        rep["union"]["true_found"] = int(is_true.sum())
        rep["union"]["candidate_recall"] = float(is_true.sum() / n_true)
        rep["union"]["pair_precision"] = float(is_true.mean()) if len(cand) else 0.0
    mask = cand["blk_mask"].to_numpy() if len(cand) else np.array([], np.int64)
    for b, name in enumerate(BLOCKER_ORDER):
        hit = (mask >> b) & 1 == 1
        if not hit.any():
            continue
        only = hit & (cand["blk_count"].to_numpy() == 1)
        d = {"label": BLOCKER_LABELS[name], "candidates": int(hit.sum()), "reduction_ratio": 1 - hit.sum() / total_space,
             "avg_candidates_per_s1": float(hit.sum() / max(1, n1))}
        if is_true is not None:
            d["recall"] = float((hit & is_true).sum() / max(1, len(true_pairs)))
            d["unique_true_found"] = int((only & is_true).sum())
            d["pair_precision"] = float((hit & is_true).sum() / max(1, hit.sum()))
        rep["blockers"][name] = d
    return rep
