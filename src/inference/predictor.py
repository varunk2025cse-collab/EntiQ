"""Resolution run: dataset + registered model -> decisions, submission files, artifacts for the UI."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.data_loader import ERDataset
from ..data.data_loader import dataset_kind
from ..data.data_validator import validate_dataset
from ..evaluation.error_analysis import analyse_errors
from ..evaluation.metrics import evaluate_selection
from ..features.pair_features import FEATURE_DESCRIPTIONS
from ..matching.decision_engine import REASON_RULES, DecisionParams, select_pairs
from ..matching.graph_consistency import graph_report
from ..matching.singleton_detector import entity_evidence
from ..models.calibration import reliability
from ..models.fellegi_sunter import REQUIRED as FS_REQUIRED
from ..utils.config import resolve_path
from ..utils.logging import get_logger, set_run_id
from ..utils.timing import StageTimer
from .pipeline import candidate_recall, make_view, prepare
from .submission_generator import write_submission
from .validator_runner import run_official, validate_submission

log = get_logger("predictor")

STORED_FEATURES = sorted(set(FEATURE_DESCRIPTIONS) | set(FS_REQUIRED) | {
    "name_core_eq", "name_norm_eq", "name_alt_cos", "name_acronym_match", "name_tok_jaccard", "name_freq_s1",
    "name_freq_other", "x_both_strong", "x_both_weak", "num_first_eq", "num_first_conflict", "country_conflict",
    "name_missing_any", "addr_missing_one", "name_lev_sim", "ctx_x_weighted_rank_other", "blk_count"})
RECORD_COLUMNS = ["entity_id", "source", "business_name", "business_address", "country", "name_norm", "name_core",
                  "name_alt", "addr_norm", "addr_core", "postal", "addr_landmark", "country_norm"]


def score_pairs(bundle: dict, X: pd.DataFrame, batch: int = 500_000):
    """Score in row batches so only one batch-sized copy of the feature matrix exists at a time."""
    cols = bundle["feature_columns"]
    raw, votes = [], {k: [] for k in bundle.get("aux", {})}
    for s in range(0, len(X), batch):
        xb = X.iloc[s:s + batch][cols]
        raw.append(bundle["matcher"].predict(xb))
        for k, m in bundle.get("aux", {}).items():
            votes[k].append(m.predict(xb))
    p_raw = np.concatenate(raw) if raw else np.zeros(0)
    return p_raw, bundle["calibrator"].transform(p_raw), {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in votes.items()}


def next_run_id(runs_dir: Path) -> str:
    year = time.strftime("%Y")
    nums = [int(m.group(1)) for p in runs_dir.glob(f"RUN-{year}-*") if (m := re.match(rf"RUN-{year}-(\d+)$", p.name))]
    return f"RUN-{year}-{(max(nums) + 1) if nums else 1:05d}"


def resolve(ds: ERDataset, bundle: dict, cfg: dict, run_id: str | None = None, model_version: str | None = None,
            write_to_output: bool = False, progress=None) -> dict:
    """Run the full cascade on `ds`. Returns the run summary (also written to runs/<run_id>/)."""
    runs_dir = resolve_path(cfg, "runs_dir")
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or next_run_id(runs_dir)
    rdir = runs_dir / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    token = set_run_id(run_id)
    step = progress or (lambda stage, pct: None)
    timer = StageTimer()
    try:
        # the model version dictates the preprocessing it was trained with
        rcfg = {**cfg, "normalization": bundle["normalization"], "blocking": bundle["blocking"],
                "features": bundle["features"]}
        step("validating data", 5)
        with timer.stage("data_validation"):
            dq = validate_dataset(ds).as_dict()
        log.info("data loaded", extra={"data": {k: len(v) for k, v in ds.sources.items()}})
        step("normalising, blocking, computing features", 15)
        with timer.stage("prepare"):
            prep = prepare(ds, rcfg, pruner=bundle.get("pruner"))
        log.info("candidates generated", extra={"data": {"candidates": len(prep.cand)}})
        missing_cols = [c for c in bundle["feature_columns"] if c not in prep.X.columns]
        if missing_cols:
            raise RuntimeError(f"feature mismatch with model: {missing_cols[:5]}")
        step("scoring candidates", 70)
        with timer.stage("inference"):
            p_raw, p, votes = score_pairs(bundle, prep.X)
        step("deciding", 85)
        prm = DecisionParams(**bundle["decision"])
        cand = prep.cand
        n_s1 = len(prep.s1_ids)
        with timer.stage("decision"):
            sel = select_pairs(cand["s1_code"].to_numpy(), cand["j"].to_numpy(), (cand["target"] == "S3").to_numpy(),
                               p, n_s1, prm)
        log.info("decisions", extra={"data": {"model": bundle["spec"]["name"], "threshold": prm.threshold,
                                              "links": int(sel.sum())}})

        # ---------------------------------------------------------------- artifacts
        step("writing outputs", 90)
        with timer.stage("outputs"):
            out_dir = rdir / "output"
            files = write_submission(out_dir, prep.s1_ids, cand, sel)
            test_dir = Path(ds.files.get("source1", "")).parent if ds.files else None
            validation = (validate_submission(files["matching"], files["candidate"], test_dir)
                          if test_dir and test_dir.exists() else {"status": "SKIPPED", "issues": ["no source files on disk"]})
            official = run_official(resolve_path(cfg, "official_validator"), files["matching"], files["candidate"],
                                    test_dir) if test_dir and test_dir.exists() else None
            if write_to_output:
                write_submission(resolve_path(cfg, "output_dir"), prep.s1_ids, cand, sel)

            feats = prep.X[[c for c in STORED_FEATURES if c in prep.X.columns]].reset_index(drop=True)
            cstore = pd.concat([cand[["s1_code", "s1_id", "other_id", "target", "blk_mask", "blk_count"]]
                                .reset_index(drop=True).rename(columns={"blk_count": "n_blockers"}), feats], axis=1)
            cstore["p"] = p.astype(np.float32)
            cstore["p_raw"] = p_raw.astype(np.float32)
            for k, v in votes.items():
                cstore[f"vote_{k}"] = v.astype(np.float32)
            cstore["selected"] = sel
            if "is_true" in cand:
                cstore["is_true"] = cand["is_true"].to_numpy()
            cstore["rank_s1"] = cstore.groupby("s1_code")["p"].rank(ascending=False, method="first").astype(np.int32)
            cstore.to_pickle(rdir / "candidates.pkl")
            rec = prep.store.records[RECORD_COLUMNS].copy()
            rec.to_pickle(rdir / "records.pkl")

            ents = _entity_table(prep, cstore, p, sel, votes, prm)
            ents.to_pickle(rdir / "entities.pkl")

        metrics, err_summary = None, None
        if prep.n_true is not None:
            view = make_view(prep, np.arange(n_s1))
            metrics = evaluate_selection(view.s1_local, sel[view.pair_idx], view.is_true, view.n_true)
            metrics["candidate_recall"] = candidate_recall(view)
            errs = analyse_errors(prep, view, p[view.pair_idx], sel[view.pair_idx], prm.threshold)
            errs["errors"].to_pickle(rdir / "errors.pkl")
            err_summary = errs["summary"]
            metrics["calibration"] = reliability(p, cand["is_true"].to_numpy())

        n_matched = int((ents["decision"] == "MATCH").sum())
        conf = cstore.loc[sel, "p"].to_numpy()
        summary = {
            "run_id": run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "COMPLETED",
            "dataset": ds.name, "dataset_fingerprint": prep.fingerprint, "model_version": model_version,
            "dataset_kind": dataset_kind(ds, resolve_path(cfg, "dataset_dir")),
            "dataset_root": str(Path(ds.files.get("source1", "")).parent.parent.name) if ds.files else None,
            "model_name": bundle["spec"]["name"], "decision": prm.as_dict(),
            "counts": {"s1": n_s1, "s2": len(ds.s2), "s3": len(ds.s3), "candidates": int(len(cand)),
                       "links": int(sel.sum()), "s2_links": int((sel & (cand["target"] == "S2").to_numpy()).sum()),
                       "s3_links": int((sel & (cand["target"] == "S3").to_numpy()).sum()),
                       "matched_entities": n_matched, "singletons": int(n_s1 - n_matched),
                       "high_confidence_links": int((conf >= 0.9).sum()),
                       "review_links": int(((cstore["p"] >= 0.3) & (cstore["p"] < prm.threshold + 0.1)
                                            & (cstore["rank_s1"] == 1)).sum())},
            "blocking": prep.blocking, "graph": graph_report(cand, sel),
            "confidence_histogram": np.histogram(conf, bins=10, range=(0, 1))[0].tolist(),
            "score_histogram": np.histogram(p, bins=20, range=(0, 1))[0].tolist(),
            "country_distribution": ents["country"].replace("", "<missing>").value_counts().to_dict(),
            "country_match_rate": ents.groupby(ents["country"].replace("", "<missing>"))["decision"]
            .apply(lambda d: float((d == "MATCH").mean())).to_dict(),
            "links_per_entity": ents["n_matches"].clip(upper=5).value_counts().sort_index().to_dict(),
            "reason_code_counts": pd.Series([c for cs in ents.loc[ents.decision == "MATCH", "reason_codes"] for c in cs])
            .value_counts().to_dict(),
            "data_quality": dq, "validation": validation, "official_validation": official,
            "metrics": metrics, "metrics_note": ("in-sample: this dataset was used for training" if metrics else None),
            "error_summary": err_summary, "timings": {**prep.timings, "run": timer.as_dict()},
            "files": {k: str(Path(v).relative_to(rdir)) for k, v in files.items()},
        }
        (rdir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
        log.info("run completed", extra={"data": {"matches": n_matched, "singletons": n_s1 - n_matched,
                                                  "runtime_s": timer.total()}})
        step("completed", 100)
        return summary
    finally:
        set_run_id(None)
        del token


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _entity_table(prep, cstore: pd.DataFrame, p: np.ndarray, sel: np.ndarray, votes: dict, prm: DecisionParams):
    n = len(prep.s1_ids)
    code = cstore["s1_code"].to_numpy()
    ev = entity_evidence(code, p, n, votes)
    s1 = prep.store.source_frame("S1").reset_index(drop=True)
    n_s2 = np.bincount(code, weights=(sel & (cstore["target"] == "S2").to_numpy()).astype(float), minlength=n)
    n_s3 = np.bincount(code, weights=(sel & (cstore["target"] == "S3").to_numpy()).astype(float), minlength=n)
    ents = pd.DataFrame({"s1_id": prep.s1_ids, "business_name": s1["business_name"], "business_address": s1["business_address"],
                         "country": s1["country"], "n_s2": n_s2.astype(int), "n_s3": n_s3.astype(int)})
    ents["n_matches"] = ents["n_s2"] + ents["n_s3"]
    ents = pd.concat([ents, ev], axis=1)
    ents["decision"] = np.where(ents["n_matches"] > 0, "MATCH", "NO_MATCH")
    minsel = np.full(n, np.inf)
    np.minimum.at(minsel, code[sel], p[sel])
    ents["confidence"] = np.where(ents["n_matches"] > 0, minsel, 1 - ents["best"]).round(4)
    # reason codes for the top accepted link (vectorised rule evaluation)
    top = cstore[cstore["selected"]].sort_values("p", ascending=False).drop_duplicates("s1_code")
    cols = {c: top[c].to_numpy() for c in top.columns}

    class _R(dict):
        def get(self, k, d=0):
            return super().get(k, np.full(len(top), d))

    r = _R(cols)
    flags = {codeName: np.asarray(rule(r), bool) for codeName, rule in REASON_RULES}
    reasons = [[] for _ in range(n)]
    for pos, sc in enumerate(top["s1_code"].to_numpy()):
        reasons[sc] = [c for c, f in flags.items() if f[pos]]
    for k in range(n):
        if ents.at[k, "decision"] == "NO_MATCH":
            reasons[k] = (["NO_CANDIDATES"] if ents.at[k, "n_candidates"] == 0 else
                          ["LOW_CONFIDENCE_SINGLETON"] if ents.at[k, "best"] < prm.singleton_threshold else
                          ["AMBIGUOUS_CANDIDATES"])
        elif ents.at[k, "gap"] >= 0.3 and "STRONG_SCORE_GAP" not in reasons[k]:
            reasons[k].append("STRONG_SCORE_GAP")
    ents["reason_codes"] = reasons
    if prep.n_true is not None:
        ents["n_true"] = prep.n_true
        tp = np.bincount(code, weights=(sel & cstore["is_true"].to_numpy()).astype(float), minlength=n)
        ents["tp"] = tp.astype(int)
        from ..evaluation.metrics import entity_fbeta

        ents["f05"] = entity_fbeta(tp, ents["n_matches"], ents["n_true"])
    return ents
