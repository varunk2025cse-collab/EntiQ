"""Entity-level evidence used to decide MATCH vs NO MATCH (singleton) and to explain it."""
from __future__ import annotations

import numpy as np
import pandas as pd


def entity_evidence(s1_code: np.ndarray, p: np.ndarray, n_s1: int, votes: dict[str, np.ndarray] | None = None,
                    strong: float = 0.75) -> pd.DataFrame:
    """Per-S1: best / second / third score, best-second gap, #strong candidates, #candidates,
    and agreement of auxiliary models on the best candidate."""
    order = np.lexsort((-p, s1_code))
    c, ps = s1_code[order], p[order]
    start = np.r_[True, c[1:] != c[:-1]] if len(c) else np.array([], bool)
    idx = np.arange(len(c))
    rank = idx - np.maximum.accumulate(np.where(start, idx, 0)) + 1 if len(c) else idx
    out = pd.DataFrame({"best": np.zeros(n_s1), "second": np.zeros(n_s1), "third": np.zeros(n_s1)})
    for r, col in ((1, "best"), (2, "second"), (3, "third")):
        m = rank == r
        out.loc[c[m], col] = ps[m]
    out["gap"] = out["best"] - out["second"]
    out["n_candidates"] = np.bincount(s1_code, minlength=n_s1)
    out["n_strong"] = np.bincount(s1_code, weights=(p >= strong).astype(float), minlength=n_s1).astype(int)
    if votes:
        best_pair = np.full(n_s1, -1)
        first = rank == 1
        best_pair[c[first]] = order[first]
        agree = np.zeros(n_s1)
        for v in votes.values():
            ok = best_pair >= 0
            agree[ok] += (v[best_pair[ok]] >= 0.5)
        out["model_agreement"] = agree / max(1, len(votes))
    return out
