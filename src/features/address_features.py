"""Address / country / missingness features for candidate pairs."""
from __future__ import annotations

import numpy as np

from .record_store import RecordStore
from .similarity import dense_rowwise_dot, levenshtein_pairs, rowwise_dot, set_overlap


def address_features(store: RecordStore, i: np.ndarray, j: np.ndarray, lev_max_len: int = 64) -> dict[str, np.ndarray]:
    r = store.records
    m = store.mats
    f: dict[str, np.ndarray] = {}
    core = r["addr_core"].to_numpy()
    norm = r["addr_norm"].to_numpy()
    f["addr_norm_eq"] = ((norm[i] == norm[j]) & (norm[i] != "")).astype(np.float32)
    f["addr_core_eq"] = ((core[i] == core[j]) & (core[i] != "")).astype(np.float32)
    f["addr_word_cos"] = rowwise_dot(m["addr_word"], m["addr_word"], i, j)
    f["addr_char_cos"] = rowwise_dot(m["addr_char"], m["addr_char"], i, j)
    f["addr_phon_cos"] = rowwise_dot(m["addr_phon"], m["addr_phon"], i, j)
    s = set_overlap(m["addr_bin"], m["addr_bin"], i, j)
    f["addr_tok_jaccard"], f["addr_tok_overlap"] = s["jaccard"], s["overlap"]
    f["addr_tok_shared"] = s["inter"]
    f["addr_ntok_diff"] = np.abs(s["na"] - s["nb"])
    w = set_overlap(m["addr_bin"], m["addr_bin"], i, j, weights=store.idf["addr_bin"])
    f["addr_idf_jaccard"], f["addr_idf_overlap"] = w["jaccard"], w["overlap"]
    f["addr_idf_shared"] = w["inter"]
    codes, _, lens = store.codes("addr_core", lev_max_len)
    _, f["addr_lev_sim"] = levenshtein_pairs(codes, lens, i, j)

    full_len = r["addr_core"].str.len().to_numpy().astype(np.float32)
    la, lb = full_len[i], full_len[j]
    f["addr_len_diff"] = np.abs(la - lb)
    f["addr_len_ratio"] = np.where(np.maximum(la, lb) > 0, np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1), 0)

    # numeric tokens (house numbers, suite, cross, sector ...)
    n = set_overlap(m["addr_num"], m["addr_num"], i, j)
    f["num_shared"], f["num_jaccard"] = n["inter"], n["jaccard"]
    both_nums = (n["na"] > 0) & (n["nb"] > 0)
    f["num_conflict"] = (both_nums & (n["inter"] == 0)).astype(np.float32)
    f["num_all_agree"] = (both_nums & (n["inter"] == np.maximum(n["na"], n["nb"]))).astype(np.float32)
    first = r["addr_nums"].str.split(" ").str[0].fillna("").to_numpy()
    f["num_first_eq"] = ((first[i] == first[j]) & (first[i] != "")).astype(np.float32)
    f["num_first_conflict"] = ((first[i] != first[j]) & (first[i] != "") & (first[j] != "")).astype(np.float32)

    postal = r["postal"].to_numpy()
    pa, pb = postal[i], postal[j]
    f["postal_eq"] = ((pa == pb) & (pa != "")).astype(np.float32)
    f["postal_conflict"] = ((pa != pb) & (pa != "") & (pb != "")).astype(np.float32)
    f["postal_prefix3_eq"] = ((np.char.ljust(pa.astype(str), 3).astype("U3") == np.char.ljust(pb.astype(str), 3).astype("U3"))
                              & (pa != "") & (pb != "")).astype(np.float32)
    f["postal_missing_any"] = ((pa == "") | (pb == "")).astype(np.float32)
    lm = r["addr_landmark"].to_numpy()
    f["landmark_any"] = ((lm[i] != "") | (lm[j] != "")).astype(np.float32)

    f["combo_lsa_cos"] = dense_rowwise_dot(store.dense["lsa_combo"], store.dense["lsa_combo"], i, j)
    return f


def country_missing_features(store: RecordStore, i: np.ndarray, j: np.ndarray) -> dict[str, np.ndarray]:
    r = store.records
    c = r["country_norm"].to_numpy()
    raw = r["country"].to_numpy()
    f: dict[str, np.ndarray] = {}
    f["country_raw_eq"] = ((raw[i] == raw[j]) & (raw[i] != "")).astype(np.float32)
    f["country_eq"] = ((c[i] == c[j]) & (c[i] != "")).astype(np.float32)
    f["country_conflict"] = ((c[i] != c[j]) & (c[i] != "") & (c[j] != "")).astype(np.float32)
    f["country_missing_any"] = ((c[i] == "") | (c[j] == "")).astype(np.float32)
    nm, am = r["name_missing"].to_numpy(), r["addr_missing"].to_numpy()
    f["name_missing_any"] = ((nm[i] + nm[j]) > 0).astype(np.float32)
    f["addr_missing_s1"] = am[i].astype(np.float32)
    f["addr_missing_other"] = am[j].astype(np.float32)
    f["addr_missing_one"] = ((am[i] + am[j]) == 1).astype(np.float32)
    f["addr_missing_both"] = ((am[i] + am[j]) == 2).astype(np.float32)
    return f
