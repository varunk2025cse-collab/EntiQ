"""Blocking experiments: compare rule sets on candidate recall vs. volume, choose from evidence."""
from __future__ import annotations

import copy
import time

import pandas as pd

from ..features.record_store import RecordStore
from .blocking_pipeline import BLOCKER_ORDER, BlockingPipeline, blocking_report

RULE_SETS = {
    "A_names_only": ["exact_name", "name_word_tfidf", "name_char_tfidf"],
    "B_names_address": ["exact_name", "name_word_tfidf", "name_char_tfidf", "address_tfidf", "postal_token"],
    "C_lexical_full": ["exact_name", "acronym", "name_word_tfidf", "name_char_tfidf", "name_char_global",
                       "address_tfidf", "name_address_char", "postal_token"],
    "D_full_with_semantic": BLOCKER_ORDER,
    "E_full_no_country_partition": BLOCKER_ORDER,
}


def _cfg_for(base: dict, enabled: list[str], partition: bool, k_scale: float = 1.0) -> dict:
    cfg = copy.deepcopy(base)
    cfg["partition_by_country"] = partition
    for name, p in cfg["blockers"].items():
        p["enabled"] = name in enabled
        if "k" in p:
            p["k"] = max(1, int(round(p["k"] * k_scale)))
    return cfg


def run_blocking_experiments(store: RecordStore, true_pairs: set, base_cfg: dict, k_scales=(0.5, 1.0, 1.5)) -> tuple[pd.DataFrame, dict]:
    rows, reports = [], {}
    configs = []
    for name, enabled in RULE_SETS.items():
        part = not name.startswith("E_") and base_cfg.get("partition_by_country", True)
        configs.append((name, enabled, part, 1.0))
    for ks in k_scales:  # candidate-depth sweep of the full rule set
        if ks != 1.0:
            configs.append((f"D_full_k_x{ks}", BLOCKER_ORDER, base_cfg.get("partition_by_country", True), ks))
    for name, enabled, part, ks in configs:
        cfg = _cfg_for(base_cfg, enabled, part, ks)
        t0 = time.perf_counter()
        cand, _ = BlockingPipeline(cfg).run(store)
        secs = time.perf_counter() - t0
        rep = blocking_report(cand, store, true_pairs)
        u = rep["union"]
        rows.append({"rule_set": name, "blockers": "+".join(enabled), "partition_by_country": part, "k_scale": ks,
                     "candidate_recall": round(u.get("candidate_recall", 0), 5), "candidates": u["candidates"],
                     "avg_candidates_per_s1": round(u["avg_candidates_per_s1"], 2),
                     "reduction_ratio": round(u["reduction_ratio"], 6), "true_found": u.get("true_found"),
                     "s1_without_candidates": u["s1_without_candidates"], "seconds": round(secs, 2)})
        reports[name] = {"config": cfg, "report": rep}
    # leave-one-out: marginal value of each blocker within the full set
    full = reports["D_full_with_semantic"]["report"]["union"].get("true_found", 0)
    loo = []
    for b in BLOCKER_ORDER:
        cfg = _cfg_for(base_cfg, [x for x in BLOCKER_ORDER if x != b], base_cfg.get("partition_by_country", True))
        cand, _ = BlockingPipeline(cfg).run(store)
        u = blocking_report(cand, store, true_pairs)["union"]
        loo.append({"removed": b, "true_lost": int(full - u.get("true_found", 0)), "candidates_saved":
                    int(reports["D_full_with_semantic"]["report"]["union"]["candidates"] - u["candidates"])})
    table = pd.DataFrame(rows)
    return table, {"reports": reports, "leave_one_out": loo}


def choose_rule_set(table: pd.DataFrame, tolerance: float = 0.001) -> pd.Series:
    """Highest recall; within `tolerance` of it, the smallest candidate volume."""
    best = table["candidate_recall"].max()
    ok = table[table["candidate_recall"] >= best - tolerance]
    return ok.sort_values(["candidates", "seconds"]).iloc[0]
