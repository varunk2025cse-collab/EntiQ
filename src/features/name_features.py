"""Name similarity features for candidate pairs."""
from __future__ import annotations

import numpy as np

from .record_store import RecordStore
from .similarity import dense_rowwise_dot, levenshtein_pairs, prefix_suffix_pairs, rowwise_dot, set_overlap


def _eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a == b) & (a != "")).astype(np.float32)


def name_features(store: RecordStore, i: np.ndarray, j: np.ndarray, lev_max_len: int = 48) -> dict[str, np.ndarray]:
    r = store.records
    m = store.mats
    f: dict[str, np.ndarray] = {}
    raw = r["business_name"].str.lower().str.strip().to_numpy()
    for col, key in (("name_norm", "name_norm_eq"), ("name_core", "name_core_eq"), ("name_phon", "name_phon_eq"),
                     ("name_sorted", "name_sorted_eq")):
        v = r[col].to_numpy()
        f[key] = _eq(v[i], v[j])
    f["name_raw_eq"] = _eq(raw[i], raw[j])

    f["name_word_cos"] = rowwise_dot(m["name_word"], m["name_word"], i, j)
    f["name_char_cos"] = rowwise_dot(m["name_char"], m["name_char"], i, j)
    f["name_phon_cos"] = rowwise_dot(m["name_phon"], m["name_phon"], i, j)
    alt1 = rowwise_dot(m["name_alt_char"], m["name_char"], i, j)
    alt2 = rowwise_dot(m["name_char"], m["name_alt_char"], i, j)
    alt3 = rowwise_dot(m["name_alt_char"], m["name_alt_char"], i, j)
    f["name_alt_cos"] = np.maximum.reduce([alt1, alt2, alt3, f["name_char_cos"]])
    f["name_lsa_cos"] = dense_rowwise_dot(store.dense["lsa_name"], store.dense["lsa_name"], i, j)

    s = set_overlap(m["name_bin"], m["name_bin"], i, j)
    f["name_tok_jaccard"], f["name_tok_overlap"] = s["jaccard"], s["overlap"]
    f["name_tok_cov_s1"], f["name_tok_cov_other"] = s["cov_a"], s["cov_b"]
    f["name_tok_shared"] = s["inter"]
    f["name_ntok_s1"], f["name_ntok_other"] = s["na"], s["nb"]
    f["name_ntok_diff"] = np.abs(s["na"] - s["nb"])
    w = set_overlap(m["name_bin"], m["name_bin"], i, j, weights=store.idf["name_bin"])
    f["name_idf_jaccard"], f["name_idf_overlap"] = w["jaccard"], w["overlap"]
    f["name_idf_shared"] = w["inter"]  # rare-token overlap mass
    f["name_idf_unshared"] = (w["na"] + w["nb"] - 2 * w["inter"])

    codes, rcodes, lens = store.codes("name_core", lev_max_len)
    f["name_lev_dist"], f["name_lev_sim"] = levenshtein_pairs(codes, lens, i, j)
    scodes, _, slens = store.codes("name_sorted", lev_max_len)
    _, f["name_sorted_lev_sim"] = levenshtein_pairs(scodes, slens, i, j)
    f["name_prefix_sim"], f["name_suffix_sim"] = prefix_suffix_pairs(codes, rcodes, lens, i, j)

    full_len = r["name_core"].str.len().to_numpy().astype(np.float32)
    la, lb = full_len[i], full_len[j]
    f["name_len_diff"] = np.abs(la - lb)
    f["name_len_ratio"] = np.where(np.maximum(la, lb) > 0, np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1), 0)

    legal = r["name_legal"].to_numpy()
    la_, lb_ = legal[i], legal[j]
    both = (la_ != "") & (lb_ != "")
    f["name_legal_eq"] = (both & (la_ == lb_)).astype(np.float32)
    f["name_legal_conflict"] = (both & (la_ != lb_)).astype(np.float32)

    acr = r["name_acronym"].to_numpy()
    nospace = r["name_core"].str.replace(" ", "", regex=False).to_numpy()
    f["name_acronym_match"] = (((acr[i] != "") & (acr[i] == nospace[j])) | ((acr[j] != "") & (acr[j] == nospace[i]))).astype(np.float32)

    first = r["name_core"].str.split(" ").str[0].fillna("").to_numpy()
    f["name_first_tok_eq"] = _eq(first[i], first[j])
    nf = store.name_freq
    f["name_freq_s1"] = np.log1p(nf[i])
    f["name_freq_other"] = np.log1p(nf[j])
    return f
