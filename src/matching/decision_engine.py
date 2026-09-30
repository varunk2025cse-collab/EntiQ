"""Final decision engine: candidate scores -> per-S1 links (0, 1 or many), with reason codes.

Decision logic (all parameters are tuned on out-of-fold validation scores for macro F0.5):
  1. exclusive_assignment (graph layer, optional): an S2/S3 record may only link to the S1 entity that
     scores it highest (enabled only if validation shows gain).
  2. max_per_source: cap links per S1 per source (None = unlimited multi-match).
  3. threshold: pair-level acceptance on the (calibrated) score.
  4. singleton gate: if the entity's best score < singleton_threshold -> NO MATCH.
  5. score-gap gate: if best accepted score leads the best rejected competitor by < min_gap,
     the entity is ambiguous -> NO MATCH (precision-first).
  mode = "expected_f": instead of 3-5, per entity choose the top-k set maximising expected F0.5
     under the calibrated probabilities (k = 0 means NO MATCH, valued at P(no true link)).
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..evaluation.metrics import BETA, evaluate_selection


@dataclass
class DecisionParams:
    mode: str = "threshold"
    threshold: float = 0.5
    singleton_threshold: float = 0.5
    min_gap: float = 0.0
    max_per_source: int | None = None
    exclusive_assignment: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _group_rank(key: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Dense rank (1 = best) of score within each key group."""
    order = np.lexsort((-score, key))
    ks = key[order]
    start = np.r_[True, ks[1:] != ks[:-1]]
    idx = np.arange(len(ks))
    grp_start = np.maximum.accumulate(np.where(start, idx, 0))
    rank = np.empty(len(ks), np.int64)
    rank[order] = idx - grp_start + 1
    return rank


def _group_max(key: np.ndarray, val: np.ndarray, n: int, fill: float = -1.0) -> np.ndarray:
    out = np.full(n, fill)
    np.maximum.at(out, key, val)
    return out


def select_pairs(s1_code: np.ndarray, j: np.ndarray, is_s3: np.ndarray, p: np.ndarray, n_s1: int,
                 prm: DecisionParams) -> np.ndarray:
    """Vectorised decision over all candidate pairs. Returns a boolean selection mask."""
    if len(p) == 0:
        return np.zeros(0, bool)
    ok = np.ones(len(p), bool)
    if prm.exclusive_assignment:
        best_j = _group_max(j, p, int(j.max()) + 1)
        ok &= p >= best_j[j] - 1e-12
    if prm.max_per_source:
        key = s1_code * 2 + is_s3.astype(np.int64)
        ok &= _group_rank(key, np.where(ok, p, -1.0)) <= prm.max_per_source

    if prm.mode == "expected_f":
        return ok & _expected_f_select(s1_code, np.where(ok, p, 0.0), n_s1)

    sel = ok & (p >= prm.threshold)
    best = _group_max(s1_code, np.where(ok, p, -1.0), n_s1)
    sel &= best[s1_code] >= prm.singleton_threshold
    if prm.min_gap > 0:
        best_rej = _group_max(s1_code, np.where(sel, -1.0, p), n_s1)
        best_acc = _group_max(s1_code, np.where(sel, p, -1.0), n_s1)
        ambiguous = (best_acc >= 0) & (best_rej >= 0) & (best_acc - best_rej < prm.min_gap)
        sel &= ~ambiguous[s1_code]
    return sel


def _expected_f_select(s1_code: np.ndarray, p: np.ndarray, n_s1: int, beta: float = BETA) -> np.ndarray:
    b2 = beta * beta
    order = np.lexsort((-p, s1_code))
    c, ps = s1_code[order], p[order]
    start = np.r_[True, c[1:] != c[:-1]]
    idx = np.arange(len(c))
    gstart = np.maximum.accumulate(np.where(start, idx, 0))
    k = idx - gstart + 1                                   # size of the top-k set ending here
    csum = np.cumsum(ps)
    before = np.where(gstart > 0, csum[gstart - 1], 0.0)
    topk = csum - before                                   # sum of top-k probabilities
    total = np.bincount(c, weights=ps, minlength=n_s1)     # expected number of true links
    ef = (1 + b2) * topk / (k + b2 * total[c])             # approx. E[F_beta | predict top-k]
    logq = np.bincount(c, weights=np.log(np.clip(1 - ps, 1e-12, 1)), minlength=n_s1)
    p_none = np.exp(logq)                                  # P(no true link) = value of predicting none
    best_ef = np.full(n_s1, -1.0)
    np.maximum.at(best_ef, c, ef)
    hit = ef >= best_ef[c] - 1e-12
    use = best_ef > p_none                                 # otherwise predicting NO MATCH is worth more
    first_best_k = np.full(n_s1, np.iinfo(np.int64).max)   # smallest k attaining the maximum
    np.minimum.at(first_best_k, c[hit], k[hit])
    chosen = use[c] & (k <= first_best_k[c])
    out = np.zeros(len(p), bool)
    out[order] = chosen
    return out


# ----------------------------------------------------------------------------- tuning
def tune_decision(s1_code: np.ndarray, j: np.ndarray, is_s3: np.ndarray, p: np.ndarray, is_true: np.ndarray,
                  n_true: np.ndarray, grid: dict) -> tuple[DecisionParams, pd.DataFrame]:
    """Grid search of decision parameters for macro F0.5 (secondary: precision)."""
    n_s1 = len(n_true)
    rows = []
    for ex, mps in itertools.product(grid.get("exclusive_assignment", [False]), grid.get("max_per_source", [None])):
        for t in grid.get("threshold", [0.5]):
            for sd in grid.get("singleton_delta", [0.0]):
                for g in grid.get("min_gap", [0.0]):
                    prm = DecisionParams("threshold", t, min(0.999, t + sd), g, mps, ex)
                    m = evaluate_selection(s1_code, select_pairs(s1_code, j, is_s3, p, n_s1, prm), is_true, n_true)
                    rows.append({**prm.as_dict(), **m})
        prm = DecisionParams("expected_f", 0.0, 0.0, 0.0, mps, ex)
        m = evaluate_selection(s1_code, select_pairs(s1_code, j, is_s3, p, n_s1, prm), is_true, n_true)
        rows.append({**prm.as_dict(), **m})
    table = pd.DataFrame(rows).sort_values(["f05", "precision"], ascending=False).reset_index(drop=True)
    b = table.iloc[0]
    mps = b["max_per_source"]
    best = DecisionParams(b["mode"], float(b["threshold"]), float(b["singleton_threshold"]), float(b["min_gap"]),
                          None if mps is None or (isinstance(mps, float) and np.isnan(mps)) else int(mps),
                          bool(b["exclusive_assignment"]))
    return best, table


# ----------------------------------------------------------------------------- structured results
REASON_RULES = [
    ("NAME_EXACT", lambda r: r.get("name_core_eq", 0) >= 1),
    # element-wise operators only: rules run on scalars (explanations) and on arrays (whole runs)
    ("NAME_HIGH_SIMILARITY", lambda r: (r.get("name_core_eq", 0) < 1) & (r.get("x_name_best", 0) >= 0.85)),
    ("TRADE_NAME_MATCH", lambda r: r.get("name_alt_cos", 0) - r.get("name_char_cos", 0) > 0.2),
    ("ACRONYM_MATCH", lambda r: r.get("name_acronym_match", 0) >= 1),
    ("ADDRESS_HIGH_SIMILARITY", lambda r: r.get("x_addr_best", 0) >= 0.75),
    ("POSTAL_MATCH", lambda r: r.get("postal_eq", 0) >= 1),
    ("STREET_NUMBER_MATCH", lambda r: r.get("num_first_eq", 0) >= 1),
    ("COUNTRY_MATCH", lambda r: r.get("country_eq", 0) >= 1),
    ("SEMANTIC_SUPPORT", lambda r: r.get("combo_lsa_cos", 0) >= 0.8),
    ("MULTI_FIELD_AGREEMENT", lambda r: r.get("x_both_strong", 0) >= 1),
    ("STRONG_SCORE_GAP", lambda r: r.get("ctx_margin_s1", 0) >= 0.2),
    ("MUTUAL_BEST", lambda r: r.get("ctx_mutual_best", 0) >= 1),
    ("NUMBER_CONFLICT", lambda r: r.get("num_first_conflict", 0) >= 1),
    ("POSTAL_CONFLICT", lambda r: r.get("postal_conflict", 0) >= 1),
    ("COUNTRY_CONFLICT", lambda r: r.get("country_conflict", 0) >= 1),
    ("COMMON_NAME", lambda r: r.get("name_freq_s1", 0) >= np.log1p(4)),
]


def reason_codes(feat_row: dict) -> list[str]:
    return [code for code, rule in REASON_RULES if rule(feat_row)]


def entity_result(s1_id: str, rows: pd.DataFrame, prm: DecisionParams) -> dict:
    """Structured decision for one S1 entity. `rows`: its candidates with columns
    other_id, target, p, selected and feature columns."""
    sel = rows[rows["selected"]]
    best = float(rows["p"].max()) if len(rows) else 0.0
    ps = np.sort(rows["p"].to_numpy())[::-1]
    second = float(ps[1]) if len(ps) > 1 else 0.0
    codes: list[str] = []
    if len(sel):
        top = sel.sort_values("p", ascending=False).iloc[0].to_dict()
        codes = reason_codes(top)
        if best - second >= 0.3 and "STRONG_SCORE_GAP" not in codes:
            codes.append("STRONG_SCORE_GAP")
        conf = float(sel["p"].min())
    else:
        codes = ["NO_CANDIDATES"] if not len(rows) else (
            ["LOW_CONFIDENCE_SINGLETON"] if best < prm.singleton_threshold else ["AMBIGUOUS_CANDIDATES"])
        conf = float(1 - best)
    return {
        "s1_id": s1_id,
        "matched_s2_ids": sel.loc[sel.target == "S2", "other_id"].tolist(),
        "matched_s3_ids": sel.loc[sel.target == "S3", "other_id"].tolist(),
        "decision": "MATCH" if len(sel) else "NO_MATCH",
        "confidence": round(conf, 4),
        "best_score": round(best, 4),
        "second_score": round(second, 4),
        "score_gap": round(best - second, 4),
        "n_candidates": int(len(rows)),
        "reason_codes": codes,
    }
