"""Pair feature engine: base similarities + cross + retrieval provenance + group context.

Feature definitions live here (FEATURE_GROUPS) so experiments can ablate groups by name and the
explanation layer can describe every column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..blocking.blocking_pipeline import BLOCKER_ORDER
from ..utils.logging import get_logger
from .address_features import address_features, country_missing_features
from .name_features import name_features
from .record_store import RecordStore
from .similarity import dense_rowwise_dot, rowwise_dot

log = get_logger("features")

SEMANTIC_FEATURES = ["name_lsa_cos", "combo_lsa_cos", "bs_semantic_lsa"]
CONTEXT_PREFIXES = ("ctx_",)


def _cross(f: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    name_s = np.maximum.reduce([f["name_char_cos"], f["name_alt_cos"], f["name_lev_sim"], f["name_sorted_lev_sim"]])
    addr_s = np.maximum(f["addr_char_cos"], f["addr_word_cos"])
    out = {
        "x_name_best": name_s,
        "x_addr_best": addr_s,
        "x_name_x_addr": name_s * addr_s,
        "x_max": np.maximum(name_s, addr_s),
        "x_min": np.minimum(name_s, addr_s),
        "x_weighted": 0.6 * name_s + 0.4 * addr_s,
        "x_name_strong_addr_weak": ((name_s >= 0.85) & (addr_s < 0.4)).astype(np.float32),
        "x_addr_strong_name_weak": ((addr_s >= 0.8) & (name_s < 0.5)).astype(np.float32),
        "x_both_strong": ((name_s >= 0.85) & (addr_s >= 0.7)).astype(np.float32),
        "x_both_weak": ((name_s < 0.5) & (addr_s < 0.4)).astype(np.float32),
    }
    return {k: v.astype(np.float32) for k, v in out.items()}


def _context(df: pd.DataFrame) -> pd.DataFrame:
    """Competition among candidates: how does this pair rank for its S1 entity and for its S2/S3 record?"""
    out = pd.DataFrame(index=df.index)
    gi = df.groupby(["i", "target"], sort=False, observed=True)
    gj = df.groupby("j", sort=False)
    for col in ("x_weighted", "name_char_cos", "addr_char_cos"):
        s = df[col]
        mx_i = gi[col].transform("max")
        out[f"ctx_{col}_rank_s1"] = gi[col].rank(ascending=False, method="min").astype(np.float32)
        out[f"ctx_{col}_gap_s1"] = (s - mx_i).astype(np.float32)
        mx_j = gj[col].transform("max")
        out[f"ctx_{col}_rank_other"] = gj[col].rank(ascending=False, method="min").astype(np.float32)
        out[f"ctx_{col}_gap_other"] = (s - mx_j).astype(np.float32)
    # margin over the best *other* candidate (positive => this pair clearly dominates)
    s = df["x_weighted"]
    mx = gi["x_weighted"].transform("max")
    # second-best per group without Python lambdas: sort, then take the 2nd row of each group
    srt = df[["i", "target", "x_weighted"]].sort_values(["i", "target", "x_weighted"], ascending=[True, True, False])
    second = srt[srt.groupby(["i", "target"], sort=False, observed=True).cumcount() == 1].set_index(["i", "target"])["x_weighted"]
    top2 = pd.MultiIndex.from_frame(df[["i", "target"]]).map(second.to_dict().get)
    top2 = pd.Series(top2, index=df.index).astype(float).fillna(0.0)
    out["ctx_margin_s1"] = np.where(s >= mx, s - top2, s - mx).astype(np.float32)
    out["ctx_n_cand_s1"] = gi["x_weighted"].transform("size").astype(np.float32)
    out["ctx_n_cand_other"] = gj["x_weighted"].transform("size").astype(np.float32)
    strong = (s >= 0.75).astype(np.float32)
    out["ctx_n_strong_s1"] = strong.groupby([df["i"], df["target"]], observed=True).transform("sum").astype(np.float32)
    out["ctx_mutual_best"] = ((out["ctx_x_weighted_rank_s1"] == 1) & (out["ctx_x_weighted_rank_other"] == 1)).astype(np.float32)
    return out


def _siblings(store: RecordStore, df: pd.DataFrame) -> pd.DataFrame:
    """Multi-match support: similarity of each candidate to its S1's best *other* candidate (any source).

    Rank features alone punish the 2nd true record of a one-to-many entity ("lost to competitor").
    A candidate that is a near-duplicate of the leading candidate is likely the same business listed
    twice, not a competitor; a dissimilar runner-up is a genuine competitor.
    """
    s = df["x_weighted"].to_numpy()
    i = df["i"].to_numpy()
    j = df["j"].to_numpy()
    order = np.lexsort((-s, i))
    io, jo = i[order], j[order]
    start = np.r_[True, io[1:] != io[:-1]] if len(io) else np.array([], bool)
    first_pos = np.maximum.accumulate(np.where(start, np.arange(len(io)), 0)) if len(io) else np.array([], int)
    top1 = jo[first_pos]
    has2 = np.r_[~start[1:], False] if len(io) else np.array([], bool)
    second_idx = np.minimum(first_pos + 1, max(len(io) - 1, 0))
    top2 = np.where(has2[first_pos], jo[second_idx], -1)
    s_sorted = s[order]
    sib = np.where(jo == top1, top2, top1)                 # best other candidate
    sib_score = np.where(jo == top1, np.where(has2[first_pos], s_sorted[second_idx], 0.0), s_sorted[first_pos])
    out = np.zeros((len(io), 4), np.float32)
    ok = sib >= 0
    if ok.any():
        m = store.mats
        out[ok, 0] = rowwise_dot(m["name_char"], m["name_char"], jo[ok], sib[ok])
        out[ok, 1] = rowwise_dot(m["addr_char"], m["addr_char"], jo[ok], sib[ok])
        out[ok, 2] = dense_rowwise_dot(store.dense["lsa_combo"], store.dense["lsa_combo"], jo[ok], sib[ok])
    out[:, 3] = sib_score
    res = np.zeros_like(out)
    res[order] = out
    frame = pd.DataFrame(res, index=df.index, columns=["ctx_sib_name_cos", "ctx_sib_addr_cos", "ctx_sib_combo_cos",
                                                       "ctx_sib_score"])
    frame["ctx_sib_dup"] = ((frame["ctx_sib_name_cos"] >= 0.8) & (frame["ctx_sib_addr_cos"] >= 0.6)).astype(np.float32)
    return frame


def compute_pair_features(store: RecordStore, cand: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Return a float32 feature frame aligned with `cand` (same index/order)."""
    cfg = cfg or {}
    bs = int(cfg.get("batch_size", 200_000))
    lev = int(cfg.get("levenshtein_max_len", 48))
    workers = int(cfg.get("n_workers", 4))
    # warm per-record caches once (encodings) so worker threads only read shared state
    store.codes("name_core", lev), store.codes("name_sorted", lev), store.codes("addr_core", max(lev, 64))

    def batch(s: int) -> dict[str, np.ndarray]:
        c = cand.iloc[s:s + bs]
        i, j = c["i"].to_numpy(), c["j"].to_numpy()
        f = name_features(store, i, j, lev)
        f.update(address_features(store, i, j, max(lev, 64)))
        f.update(country_missing_features(store, i, j))
        f.update(_cross(f))
        f["is_s3"] = (c["target"].to_numpy() == "S3").astype(np.float32)
        for n in BLOCKER_ORDER:
            f[f"bs_{n}"] = c[f"bs_{n}"].to_numpy().astype(np.float32)
        f["blk_count"] = c["blk_count"].to_numpy().astype(np.float32)
        return f

    starts = list(range(0, len(cand), bs))
    if not starts:
        return pd.DataFrame()
    # Batches are written straight into ONE preallocated float32 matrix as they finish, so peak memory is
    # a single copy of the feature matrix (plus in-flight batches), not batches + concatenation copies.
    out: np.ndarray | None = None
    columns: list[str] = []

    reserve = 40  # room for context + sibling columns, filled in place below (no concat copy)

    def put(s: int, f: dict[str, np.ndarray]) -> None:
        nonlocal out, columns
        if out is None:
            columns = list(f)
            out = np.empty((len(cand), len(columns) + reserve), dtype=np.float32)
        n = len(next(iter(f.values())))
        for k, col in enumerate(columns):
            out[s:s + n, k] = f[col]

    if workers > 1 and len(starts) > 1:
        # numpy / scipy kernels release the GIL, so batches run concurrently in threads
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for s, f in zip(starts, ex.map(batch, starts)):
                put(s, f)
                del f
    else:
        for s in starts:
            put(s, batch(s))
    nb = len(columns)
    pick = [columns.index(c) for c in ("x_weighted", "name_char_cos", "addr_char_cos")]
    base = cand[["i", "j", "target"]].copy()
    for c, k in zip(("x_weighted", "name_char_cos", "addr_char_cos"), pick):
        base[c] = out[:, k]
    for extra in (_context(base), _siblings(store, base)):
        for c in extra.columns:
            if len(columns) - nb >= reserve:
                raise RuntimeError("increase `reserve` in compute_pair_features")
            out[:, len(columns)] = extra[c].to_numpy(np.float32)
            columns.append(c)
        del extra
    # a column-slice view of the single matrix: no copy, one block
    X = pd.DataFrame(out[:, :len(columns)], index=cand.index, columns=columns, copy=False)
    log.info("features computed", extra={"data": {"pairs": len(X), "features": X.shape[1]}})
    return X


def select_features(columns, drop_groups: list[str] | None = None) -> list[str]:
    """Feature subset for ablations: groups 'semantic', 'context', 'blocking'."""
    drop_groups = drop_groups or []
    cols = list(columns)
    if "semantic" in drop_groups:
        cols = [c for c in cols if c not in SEMANTIC_FEATURES]
    if "context" in drop_groups:
        cols = [c for c in cols if not c.startswith(CONTEXT_PREFIXES)]
    if "sibling" in drop_groups:
        cols = [c for c in cols if not c.startswith("ctx_sib_")]
    if "blocking" in drop_groups:
        cols = [c for c in cols if not (c.startswith("bs_") or c == "blk_count")]
    return cols


FEATURE_DESCRIPTIONS = {
    "name_char_cos": "Character n-gram TF-IDF cosine of core names",
    "name_word_cos": "Word TF-IDF cosine of core names",
    "name_alt_cos": "Best cosine incl. trade/DBA names",
    "name_phon_cos": "Transliteration-folded name similarity",
    "name_lev_sim": "Normalised edit similarity of core names",
    "name_sorted_lev_sim": "Edit similarity after token sorting (word-order invariant)",
    "name_idf_jaccard": "Rare-token-weighted name token overlap",
    "name_lsa_cos": "Semantic (LSA) name embedding similarity",
    "addr_char_cos": "Character n-gram cosine of addresses",
    "addr_word_cos": "Word TF-IDF cosine of addresses",
    "addr_idf_jaccard": "Rare-token-weighted address overlap",
    "num_conflict": "Both addresses have numbers but share none",
    "postal_eq": "Identical postal code",
    "postal_conflict": "Different postal codes",
    "country_eq": "Same country",
    "combo_lsa_cos": "Semantic (LSA) similarity of name+address",
    "ctx_margin_s1": "Lead over the next-best candidate of the same S1",
    "ctx_mutual_best": "Best candidate in both directions",
    "x_weighted": "Weighted name/address evidence",
}
