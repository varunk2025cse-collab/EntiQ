"""First-stage candidate pruner (cascade stage 1): cheap features -> small model -> drop hopeless pairs.

Why: blocking is tuned for recall, so most candidates are easy negatives. Scoring them with the full
~110-feature engine (edit distances, context ranks, ...) dominates runtime. The pruner uses only
retrieval scores and a few sparse cosines (near-free) and keeps a pair if
    pruner_score >= tau   OR   the pair is in the S1's top-`keep_top` for its source.
tau is chosen on OUT-OF-FOLD scores (grouped by S1) as the largest value that still retains
`target_recall` of the true links that blocking found. The surviving set is what the matcher scores,
i.e. exactly what candidate_pairs.tsv reports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features.record_store import RecordStore
from ..features.similarity import dense_rowwise_dot, rowwise_dot
from .blocking_pipeline import BLOCKER_ORDER


def cheap_features(store: RecordStore, cand: pd.DataFrame) -> pd.DataFrame:
    i, j = cand["i"].to_numpy(), cand["j"].to_numpy()
    m = store.mats
    c = store.records["country_norm"].to_numpy()
    postal = store.records["postal"].to_numpy()
    f = {f"bs_{n}": cand[f"bs_{n}"].to_numpy() for n in BLOCKER_ORDER}
    f["blk_count"] = cand["blk_count"].to_numpy().astype(np.float32)
    f["name_char"] = rowwise_dot(m["name_char"], m["name_char"], i, j)
    f["name_word"] = rowwise_dot(m["name_word"], m["name_word"], i, j)
    f["addr_char"] = rowwise_dot(m["addr_char"], m["addr_char"], i, j)
    f["combo_lsa"] = dense_rowwise_dot(store.dense["lsa_combo"], store.dense["lsa_combo"], i, j)
    f["country_eq"] = (c[i] == c[j]).astype(np.float32)
    f["postal_eq"] = ((postal[i] == postal[j]) & (postal[i] != "")).astype(np.float32)
    X = pd.DataFrame(f, index=cand.index).astype(np.float32)
    # rank of the pair among its S1's candidates (per source) on the strongest cheap signal
    s = 0.6 * X["name_char"] + 0.4 * X["addr_char"]
    X["rank_s1"] = s.groupby([cand["i"], cand["target"]], observed=True).rank(ascending=False, method="first").astype(np.float32)
    X["log_rank_s1"] = np.log1p(X["rank_s1"]).astype(np.float32)  # linear-model friendly
    return X


class CandidatePruner:
    def __init__(self, target_recall: float = 0.999, keep_top: int = 3, seed: int = 42, max_iter: int = 80) -> None:
        self.target_recall = target_recall
        self.keep_top = keep_top
        self.seed = seed
        self.max_iter = max_iter
        self.model = None
        self.tau = 0.0
        self.report: dict = {}

    def _new(self):
        # Logistic regression, deliberately: tau lives in the extreme low tail (~1e-5), where boosted-tree
        # probabilities are quantised leaf floors that shift with early-stopping splits and seeds (we
        # measured train/test kept fractions diverging from 42% to 100%). LR scores are smooth and scale
        # consistently between out-of-fold and inference, and cost almost nothing on millions of pairs.
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, folds: int = 3) -> np.ndarray:
        """Grouped K-fold: OOF scores choose tau; the K fold models are KEPT and averaged at inference.

        Scoring new data with the same fold models keeps test scores on the scale tau was calibrated on
        (a model refitted on all data has a different probability floor, which at tau ~ 1e-5 silently
        disables pruning). Returns the OOF keep-mask for the training data.
        """
        from sklearn.model_selection import GroupKFold

        y = np.asarray(y).astype(int)
        Xn = X.to_numpy()
        uniq = np.unique(groups)
        perm = dict(zip(uniq, np.random.RandomState(self.seed).permutation(len(uniq))))
        g = np.vectorize(perm.get)(groups) if len(groups) else groups
        oof = np.zeros(len(y))
        self.models = []
        for tr, va in GroupKFold(n_splits=max(2, min(folds, len(uniq)))).split(Xn, y, g):
            m = self._new().fit(Xn[tr], y[tr])
            oof[va] = m.predict_proba(Xn[va])[:, 1]
            self.models.append(m)
        top = X["rank_s1"].to_numpy() <= self.keep_top
        pos = y == 1
        best_tau = 0.0
        for tau in np.unique(np.round(np.quantile(oof[~top], np.linspace(0, 0.995, 200)), 6)) if (~top).any() else []:
            keep = top | (oof >= tau)
            rec = (keep & pos).sum() / max(1, pos.sum())
            if rec >= self.target_recall:
                best_tau = float(tau)
        self.tau = best_tau
        keep = top | (oof >= self.tau)
        self.report = {"tau": self.tau, "keep_top": self.keep_top, "pairs_in": int(len(y)), "pairs_kept": int(keep.sum()),
                       "kept_fraction": float(keep.mean()) if len(y) else 1.0,
                       "true_retained_oof": float((keep & pos).sum() / max(1, pos.sum())),
                       "true_lost_oof": int((~keep & pos).sum())}
        return keep

    def score(self, X: pd.DataFrame) -> np.ndarray:
        Xn = X.to_numpy()
        return np.mean([m.predict_proba(Xn)[:, 1] for m in self.models], axis=0)

    def keep_mask(self, X: pd.DataFrame) -> np.ndarray:
        return (X["rank_s1"].to_numpy() <= self.keep_top) | (self.score(X) >= self.tau)
