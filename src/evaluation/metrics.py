"""Official metric: per-S1 F_beta (beta=0.5) macro-averaged over ALL S1 entities (singletons included).

Closed form per entity:  F = (1+b^2) tp / (b^2 * n_true + n_pred)
  - n_true == 0 and n_pred == 0  -> 1.0  (correct singleton)
  - n_true == 0 and n_pred  > 0  -> 0.0  (false merge on a singleton)
  - n_true  > 0 and tp == 0      -> 0.0
This equals (1+b^2)PR/(b^2 P + R) whenever P and R are defined.
"""
from __future__ import annotations

import numpy as np

BETA = 0.5


def entity_fbeta(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray, beta: float = BETA) -> np.ndarray:
    b2 = beta * beta
    tp, n_pred, n_true = (np.asarray(x, float) for x in (tp, n_pred, n_true))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(tp > 0, (1 + b2) * tp / (b2 * n_true + n_pred), 0.0)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, f)
    return f


def summarize(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray, beta: float = BETA) -> dict:
    """Macro F0.5 plus the diagnostic metrics tracked for every experiment."""
    f = entity_fbeta(tp, n_pred, n_true, beta)
    f1 = entity_fbeta(tp, n_pred, n_true, 1.0)
    tp, n_pred, n_true = (np.asarray(x, float) for x in (tp, n_pred, n_true))
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_e = np.where(n_pred > 0, tp / n_pred, np.where(n_true == 0, 1.0, 0.0))
        rec_e = np.where(n_true > 0, tp / n_true, np.where(n_pred == 0, 1.0, 0.0))
    single = n_true == 0
    matched = ~single
    TP, FP, FN = tp.sum(), (n_pred - tp).sum(), (n_true - tp).sum()
    return {
        "f05": float(f.mean()) if len(f) else 0.0,
        "f1": float(f1.mean()) if len(f1) else 0.0,
        "precision": float(prec_e.mean()) if len(f) else 0.0,  # macro (entity-level)
        "recall": float(rec_e.mean()) if len(f) else 0.0,
        "pair_precision": float(TP / max(1.0, TP + FP)),
        "pair_recall": float(TP / max(1.0, TP + FN)),
        "true_matches": int(TP), "false_matches": int(FP), "missed_matches": int(FN),
        "entities": int(len(f)),
        "singletons": int(single.sum()),
        "singleton_accuracy": float((n_pred[single] == 0).mean()) if single.any() else 1.0,
        "f05_singletons": float(f[single].mean()) if single.any() else 1.0,
        "f05_matched_entities": float(f[matched].mean()) if matched.any() else 1.0,
        "predicted_singletons": int((n_pred == 0).sum()),
    }


def evaluate_selection(s1_code: np.ndarray, selected: np.ndarray, is_true: np.ndarray, n_true: np.ndarray) -> dict:
    """Metric from candidate-level arrays. n_true counts ALL true matches (incl. ones blocking missed)."""
    n = len(n_true)
    tp = np.bincount(s1_code, weights=(selected & is_true).astype(float), minlength=n)
    npred = np.bincount(s1_code, weights=selected.astype(float), minlength=n)
    return summarize(tp, npred, n_true)


def evaluate_predictions(pred: dict[str, list[str]], gt: dict[str, list[str]], s1_ids) -> dict:
    """Metric from id lists (used on submission-format outputs)."""
    tp, npred, ntrue = [], [], []
    for s in s1_ids:
        p, t = set(pred.get(s, [])), set(gt.get(s, []))
        tp.append(len(p & t))
        npred.append(len(p))
        ntrue.append(len(t))
    return summarize(np.array(tp), np.array(npred), np.array(ntrue))
