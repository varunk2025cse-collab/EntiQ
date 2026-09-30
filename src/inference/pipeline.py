"""Shared data preparation: normalise -> index -> block -> features (cached per dataset + config)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..blocking.blocking_pipeline import BlockingPipeline, blocking_report
from ..blocking.pruner import CandidatePruner, cheap_features
from ..data.data_loader import ERDataset
from ..features.pair_features import compute_pair_features
from ..features.record_store import RecordStore
from ..preprocessing.normalizer import Normalizer
from ..utils.config import config_hash, resolve_path
from ..utils.logging import get_logger
from ..utils.timing import StageTimer

log = get_logger("pipeline")
PREP_VERSION = "prep-v8"


@dataclass
class Prepared:
    name: str
    fingerprint: str
    store: RecordStore
    cand: pd.DataFrame            # i, j, target, blocker cols, s1_code, s1_id, other_id, (is_true)
    X: pd.DataFrame               # pair features aligned with cand
    s1_ids: np.ndarray            # S1 ids in S1 order (s1_code indexes this)
    n_true: np.ndarray | None     # true link count per S1 (incl. links blocking missed)
    blocking: dict
    timings: dict
    ground_truth: dict | None = None
    meta: dict = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        return self.cand["is_true"].to_numpy().astype(int)


def prep_key(ds: ERDataset, cfg: dict, pruner=None) -> str:
    return config_hash({"v": PREP_VERSION, "fp": ds.fingerprint(), "norm": cfg["normalization"],
                        "block": cfg["blocking"], "feat": cfg["features"], "seed": cfg.get("seed"),
                        "pruner": getattr(pruner, "tag", None)})


def prepare(ds: ERDataset, cfg: dict, use_cache: bool = True, pruner=None) -> Prepared:
    """`pruner`: a fitted CandidatePruner (inference). With labels and pruning enabled but no pruner given,
    one is fitted here on out-of-fold scores and returned in `prep.meta['pruner']`."""
    key = prep_key(ds, cfg, pruner)
    cache_dir = resolve_path(cfg, "processed_dir")
    path = Path(cache_dir) / f"prep_{ds.name}_{key}.joblib"
    if use_cache and path.exists():
        log.info("loaded cached preparation", extra={"data": {"path": str(path)}})
        return joblib.load(path)

    timer = StageTimer()
    with timer.stage("normalization_and_indexing"):
        store = RecordStore.build(ds, Normalizer(cfg["normalization"]), int(cfg["features"].get("lsa_dims", 128)),
                                  int(cfg.get("seed", 42)))
    with timer.stage("blocking"):
        cand, binfo = BlockingPipeline(cfg["blocking"]).run(store)
    ids = store.ids()
    s1_lo, s1_hi = store.ranges["S1"]
    s1_ids = ids[s1_lo:s1_hi]
    cand["s1_code"] = (cand["i"].to_numpy() - s1_lo).astype(np.int64)
    cand["s1_id"] = ids[cand["i"].to_numpy()]
    cand["other_id"] = ids[cand["j"].to_numpy()]
    n_true = None
    if ds.ground_truth is not None:
        tp = ds.true_pairs()
        cand["is_true"] = [(a, b) in tp for a, b in zip(cand["s1_id"], cand["other_id"])]
        n_true = np.array([len(ds.ground_truth.get(s, [])) for s in s1_ids], dtype=np.int64)
    truth = ds.true_pairs() if ds.ground_truth is not None else None
    pre_pruning = None
    pcfg = cfg["blocking"].get("pruning", {}) or {}
    if pcfg.get("enabled"):
        with timer.stage("pruning"):
            pre_pruning = blocking_report(cand, store, truth)["union"]
            Xc = cheap_features(store, cand)
            if pruner is None and truth is not None:
                pruner = CandidatePruner(float(pcfg.get("target_recall", 0.999)), int(pcfg.get("keep_top", 3)),
                                         int(cfg.get("seed", 42)))
                keep = pruner.fit(Xc, cand["is_true"].to_numpy(), cand["s1_code"].to_numpy())
                pruner.tag = f"{key}-{uuid.uuid4().hex[:8]}"
            elif pruner is not None:
                keep = pruner.keep_mask(Xc)
            else:
                keep = np.ones(len(cand), bool)
                log.warning("pruning enabled but no fitted pruner and no labels: keeping all candidates")
            cand = cand[keep].reset_index(drop=True)
        log.info("candidates pruned", extra={"data": {"before": pre_pruning["candidates"], "after": len(cand)}})
    with timer.stage("features"):
        X = compute_pair_features(store, cand, cfg["features"])
    rep = blocking_report(cand, store, truth)
    rep["timings"] = binfo["timings"]
    if pre_pruning is not None:
        rep["pre_pruning"] = pre_pruning
        rep["pruner"] = getattr(pruner, "report", None)
    prep = Prepared(ds.name, ds.fingerprint(), store, cand, X, s1_ids, n_true, rep, timer.as_dict(),
                    ds.ground_truth, {"prep_key": key, "pruner": pruner})
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(prep, path, compress=3)
    return prep


@dataclass
class View:
    """A subset of S1 entities (e.g. dev or holdout) with locally re-indexed arrays for the metric."""

    pair_idx: np.ndarray
    s1_local: np.ndarray
    n_true: np.ndarray
    j: np.ndarray
    is_s3: np.ndarray
    is_true: np.ndarray
    s1_codes: np.ndarray

    @property
    def n_s1(self) -> int:
        return len(self.n_true)


def make_view(prep: Prepared, s1_codes: np.ndarray) -> View:
    s1_codes = np.sort(np.asarray(s1_codes))
    local = np.full(len(prep.s1_ids), -1, np.int64)
    local[s1_codes] = np.arange(len(s1_codes))
    code = prep.cand["s1_code"].to_numpy()
    pair_idx = np.flatnonzero(local[code] >= 0)
    is_true = prep.cand["is_true"].to_numpy()[pair_idx] if "is_true" in prep.cand else np.zeros(len(pair_idx), bool)
    return View(pair_idx, local[code[pair_idx]], prep.n_true[s1_codes] if prep.n_true is not None else
                np.zeros(len(s1_codes), np.int64), prep.cand["j"].to_numpy()[pair_idx],
                (prep.cand["target"].to_numpy()[pair_idx] == "S3"), is_true.astype(bool), s1_codes)


def candidate_recall(view: View) -> float:
    tot = view.n_true.sum()
    return float(view.is_true.sum() / tot) if tot else 1.0
