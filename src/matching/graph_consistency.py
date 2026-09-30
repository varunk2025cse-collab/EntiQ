"""Entity-graph diagnostics over accepted links (S1 -- S2/S3 bipartite graph).

The *enforcing* variant (exclusive assignment) lives in the decision engine and is only switched on
when validation shows it helps. This module reports graph facts for analysis and the UI.
"""
from __future__ import annotations

import pandas as pd


def graph_report(cand: pd.DataFrame, selected) -> dict:
    sel = cand[selected]
    claimed = sel.groupby("other_id")["s1_id"].nunique()
    per_s1 = sel.groupby("s1_id").agg(n=("other_id", "size"), s2=("target", lambda t: (t == "S2").sum()),
                                      s3=("target", lambda t: (t == "S3").sum()))
    return {
        "linked_records": int(len(claimed)),
        "records_claimed_by_multiple_s1": int((claimed > 1).sum()),
        "s1_with_multiple_links": int((per_s1["n"] > 1).sum()) if len(per_s1) else 0,
        "s1_linked_to_both_sources": int(((per_s1["s2"] > 0) & (per_s1["s3"] > 0)).sum()) if len(per_s1) else 0,
        "max_links_per_s1": int(per_s1["n"].max()) if len(per_s1) else 0,
    }
