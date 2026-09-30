"""Candidate generation entry point (normalise -> index -> multi-pass blocking), usable standalone."""
from __future__ import annotations

from ..blocking.blocking_pipeline import BlockingPipeline, blocking_report
from ..data.data_loader import ERDataset
from ..features.record_store import RecordStore
from ..preprocessing.normalizer import Normalizer


def generate_candidates(ds: ERDataset, cfg: dict):
    store = RecordStore.build(ds, Normalizer(cfg["normalization"]), int(cfg["features"].get("lsa_dims", 128)),
                              int(cfg.get("seed", 42)))
    cand, info = BlockingPipeline(cfg["blocking"]).run(store)
    rep = blocking_report(cand, store, ds.true_pairs() if ds.ground_truth is not None else None)
    rep["timings"] = info["timings"]
    return store, cand, rep
