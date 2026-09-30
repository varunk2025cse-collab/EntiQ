"""Human-readable explanations for one (S1, candidate) pair of a resolution run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.pair_features import FEATURE_DESCRIPTIONS
from ..features.similarity import jaro_winkler, levenshtein
from ..matching.decision_engine import reason_codes
from ..models.fellegi_sunter import COMPARISONS, _levels

EVIDENCE = [  # (feature, label, kind) shown as bars in the UI
    ("x_name_best", "Name similarity", "sim"),
    ("x_addr_best", "Address similarity", "sim"),
    ("name_idf_jaccard", "Rare-name-token overlap", "sim"),
    ("combo_lsa_cos", "Semantic similarity", "sim"),
    ("country_eq", "Country match", "flag"),
    ("postal_eq", "Postal code match", "flag"),
    ("num_first_eq", "Street number match", "flag"),
    ("ctx_margin_s1", "Lead over next candidate", "gap"),
]


def token_diff(a: str, b: str) -> dict:
    ta, tb = a.split(), b.split()
    sa, sb = set(ta), set(tb)
    return {"left": [{"t": t, "shared": t in sb} for t in ta], "right": [{"t": t, "shared": t in sa} for t in tb],
            "shared": sorted(sa & sb), "left_only": sorted(sa - sb), "right_only": sorted(sb - sa)}


def explain_pair(records: pd.DataFrame, cand_row: pd.Series, fs_model=None, decision: dict | None = None) -> dict:
    """`records` indexed by entity_id; `cand_row` a row of the run's candidates table."""
    a = records.loc[cand_row["s1_id"]]
    b = records.loc[cand_row["other_id"]]
    feats = {k: float(v) for k, v in cand_row.items() if isinstance(v, (int, float, np.floating, np.integer))
             and not isinstance(v, bool)}
    codes = reason_codes(feats)
    evidence = []
    for f, label, kind in EVIDENCE:
        if f in feats:
            evidence.append({"feature": f, "label": label, "kind": kind, "value": round(feats[f], 4),
                             "description": FEATURE_DESCRIPTIONS.get(f, label)})
    votes = {"final (calibrated)": round(float(cand_row["p"]), 4), "primary model (raw)": round(float(cand_row["p_raw"]), 4)}
    for k in cand_row.index:
        if k.startswith("vote_"):
            votes[k[5:].replace("_", " ")] = round(float(cand_row[k]), 4)
    fs = None
    if fs_model is not None:
        try:
            one = pd.DataFrame([feats])
            lv = _levels(one)
            fs = [{"field": f, "level": COMPARISONS[f][2][lv[f][0]], "weight_bits": round(float(fs_model.weights[f][lv[f][0]]), 2)}
                  for f in lv]
        except Exception:
            fs = None
    na, nb = str(a["name_core"]), str(b["name_core"])
    verdict = "MATCH" if bool(cand_row["selected"]) else "NO MATCH"
    return {
        "s1": _rec(a, cand_row["s1_id"]), "candidate": _rec(b, cand_row["other_id"]),
        "target": cand_row["target"],
        "name_tokens": token_diff(na, nb),
        "address_tokens": token_diff(str(a["addr_core"]), str(b["addr_core"])),
        "string_metrics": {"name_levenshtein": levenshtein(na, nb), "name_jaro_winkler": round(jaro_winkler(na, nb), 4)},
        "country_agreement": str(a["country_norm"]) == str(b["country_norm"]) and str(a["country_norm"]) != "",
        "evidence": evidence, "features": {k: round(v, 4) for k, v in feats.items()},
        "model_votes": votes, "fellegi_sunter": fs, "reason_codes": codes,
        "decision": verdict, "confidence": round(float(cand_row["p"]), 4), "rank": int(cand_row.get("rank_s1", 0)),
        "decision_params": decision, "summary": _sentence(verdict, feats, codes, float(cand_row["p"])),
        "ground_truth": (bool(cand_row["is_true"]) if "is_true" in cand_row else None),
    }


def _rec(r: pd.Series, eid: str) -> dict:
    return {"entity_id": eid, "business_name": r["business_name"], "business_address": r["business_address"],
            "country": r["country"], "name_normalized": r["name_norm"], "name_core": r["name_core"],
            "address_normalized": r["addr_norm"], "postal": r["postal"], "landmark": r["addr_landmark"]}


def _sentence(verdict: str, f: dict, codes: list[str], p: float) -> str:
    name, addr = f.get("x_name_best", 0), f.get("x_addr_best", 0)
    parts = [f"Names are {_level(name)} similar ({name:.0%})", f"addresses are {_level(addr)} similar ({addr:.0%})"]
    if f.get("postal_conflict"):
        parts.append("postal codes disagree")
    elif f.get("postal_eq"):
        parts.append("postal codes agree")
    if f.get("country_conflict"):
        parts.append("countries differ")
    if "COMMON_NAME" in codes:
        parts.append("the name is shared by several businesses, so the address carries the decision")
    return f"{verdict} at {p:.0%} confidence: " + ", ".join(parts) + "."


def _level(x: float) -> str:
    return "nearly identically" if x >= 0.95 else "highly" if x >= 0.8 else "moderately" if x >= 0.5 else "weakly"
