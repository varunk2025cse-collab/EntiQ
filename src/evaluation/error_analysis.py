"""Automatic error analysis: categorise every false positive / false negative on validation data."""
from __future__ import annotations

import numpy as np
import pandas as pd

FP_RULES = [  # (category, predicate on a feature row) — first match = primary category
    ("singleton_false_merge", lambda r: r["n_true"] == 0),
    ("common_business_name", lambda r: r["name_freq_s1"] >= np.log1p(2) or r["name_freq_other"] >= np.log1p(3)),
    ("name_collision", lambda r: r["x_name_best"] >= 0.9 and r["x_addr_best"] < 0.5),
    ("address_collision", lambda r: r["x_addr_best"] >= 0.8 and r["x_name_best"] < 0.6),
    ("over_normalization", lambda r: r["name_core_eq"] >= 1 and r["name_norm_eq"] < 1),
    ("semantic_false_positive", lambda r: r["combo_lsa_cos"] >= 0.85 and r["x_name_best"] < 0.7),
    ("ambiguous_candidate", lambda r: abs(r["ctx_margin_s1"]) < 0.05),
    ("threshold_issue", lambda r: r["p"] - r["threshold"] < 0.1),
    ("weak_evidence", lambda r: True),
]
FN_RULES = [
    ("lost_to_competitor", lambda r: r["ctx_x_weighted_rank_s1"] > 1 and r["p"] < r["threshold"]),
    ("threshold_too_strict", lambda r: r["p"] >= r["threshold"] - 0.15),
    ("trade_name", lambda r: r["name_alt_cos"] - r["name_char_cos"] > 0.2),
    ("abbreviation", lambda r: r["name_acronym_match"] >= 1 or (r["name_tok_jaccard"] < 0.5 and r["name_char_cos"] >= 0.5)),
    ("transliteration", lambda r: r["name_phon_cos"] - r["name_char_cos"] > 0.08),
    ("typo", lambda r: r["name_core_eq"] < 1 and r["name_lev_sim"] >= 0.8),
    ("missing_field", lambda r: r["addr_missing_one"] >= 1 or r["name_missing_any"] >= 1),
    ("address_variation", lambda r: r["x_addr_best"] < 0.5 and r["x_name_best"] >= 0.8),
    ("semantic_failure", lambda r: r["combo_lsa_cos"] < 0.5),
    ("rejected_by_decision_gate", lambda r: True),
]
KEY_FEATURES = ["x_name_best", "x_addr_best", "name_char_cos", "addr_char_cos", "combo_lsa_cos", "postal_eq",
                "num_first_eq", "country_eq", "ctx_margin_s1", "ctx_x_weighted_rank_s1"]


def _categorise(row: dict, rules) -> tuple[str, list[str]]:
    tags = [c for c, f in rules[:-1] if _safe(f, row)]
    return (tags[0] if tags else rules[-1][0]), tags


def _safe(f, row) -> bool:
    try:
        return bool(f(row))
    except Exception:
        return False


def analyse_errors(prep, view, p: np.ndarray, selected: np.ndarray, threshold: float, limit: int | None = None) -> dict:
    """`view` restricts to validation entities; p/selected are aligned with view.pair_idx."""
    cand = prep.cand.iloc[view.pair_idx]
    X = prep.X.iloc[view.pair_idx]
    rec = prep.store.records
    n_true_map = dict(zip(prep.s1_ids[view.s1_codes], view.n_true))
    base = pd.concat([cand[["s1_id", "other_id", "target", "i", "j"]].reset_index(drop=True),
                      X.reset_index(drop=True)], axis=1)
    base["p"] = p
    base["selected"] = selected
    base["is_true"] = view.is_true
    base["threshold"] = threshold
    base["n_true"] = base["s1_id"].map(n_true_map)
    rows = []
    for r in base[base.selected & ~base.is_true].to_dict("records"):
        cat, tags = _categorise(r, FP_RULES)
        rows.append(_row("FALSE_POSITIVE", r, cat, tags, rec))
    for r in base[~base.selected & base.is_true].to_dict("records"):
        cat, tags = _categorise(r, FN_RULES)
        rows.append(_row("FALSE_NEGATIVE", r, cat, tags, rec))
    # true links never retrieved by blocking
    if prep.ground_truth is not None:
        found = set(zip(base.loc[base.is_true, "s1_id"], base.loc[base.is_true, "other_id"]))
        id_row = prep.store.id_to_row()
        for s1 in prep.s1_ids[view.s1_codes]:
            for m in prep.ground_truth.get(s1, []):
                if (s1, m) not in found and m in id_row:
                    i, j = id_row[s1], id_row[m]
                    rows.append({"error_type": "FALSE_NEGATIVE", "category": "blocking_failure", "tags": ["blocking_failure"],
                                 "s1_id": s1, "other_id": m, "p": None, "threshold": threshold, "ground_truth": "same business",
                                 "prediction": "never retrieved", "reason_codes": [], **_texts(rec, i, j)})
    errs = pd.DataFrame(rows)
    summary = {"false_positives": int((errs.error_type == "FALSE_POSITIVE").sum()) if len(errs) else 0,
               "false_negatives": int((errs.error_type == "FALSE_NEGATIVE").sum()) if len(errs) else 0,
               "by_category": errs.groupby(["error_type", "category"]).size().rename("count").reset_index()
               .to_dict(orient="records") if len(errs) else []}
    if limit is not None and len(errs):
        errs = errs.head(limit)
    return {"summary": summary, "errors": errs}


def _texts(rec: pd.DataFrame, i: int, j: int) -> dict:
    a, b = rec.iloc[i], rec.iloc[j]
    return {"s1_name": a.business_name, "s1_address": a.business_address, "s1_country": a.country,
            "other_name": b.business_name, "other_address": b.business_address, "other_country": b.country}


def _row(kind: str, r: dict, cat: str, tags: list[str], rec: pd.DataFrame) -> dict:
    from ..matching.decision_engine import reason_codes

    out = {"error_type": kind, "category": cat, "tags": tags, "s1_id": r["s1_id"], "other_id": r["other_id"],
           "p": round(float(r["p"]), 4), "threshold": round(float(r["threshold"]), 4),
           "ground_truth": "same business" if kind == "FALSE_NEGATIVE" else "different businesses",
           "prediction": "match" if kind == "FALSE_POSITIVE" else "no match",
           "reason_codes": reason_codes({k: v for k, v in r.items() if isinstance(v, (int, float))})}
    out.update({k: round(float(r[k]), 4) for k in KEY_FEATURES if k in r})
    out.update(_texts(rec, int(r["i"]), int(r["j"])))
    return out
