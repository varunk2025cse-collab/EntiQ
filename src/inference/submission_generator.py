"""Write matching_results.tsv and candidate_pairs.tsv in the exact competition format."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _write(path: Path, header: tuple[str, str], s1_ids, lists: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{header[0]}\t{header[1]}\n")
        for s in s1_ids:
            seen, ordered = set(), []
            for x in lists.get(s, []):
                if x not in seen:
                    seen.add(x)
                    ordered.append(x)
            fh.write(f"{s}\t{','.join(ordered)}\n")


def write_submission(out_dir: str | Path, s1_ids, cand: pd.DataFrame, selected_mask) -> dict[str, str]:
    """cand needs columns s1_id, other_id; selected_mask aligned with cand."""
    out = Path(out_dir)
    cand_lists = cand.groupby("s1_id", sort=False)["other_id"].apply(list).to_dict()
    sel = cand[selected_mask]
    match_lists = sel.groupby("s1_id", sort=False)["other_id"].apply(list).to_dict()
    # matches are by construction a subset of candidates; assert it anyway (pipeline-bug guard)
    for s, ms in match_lists.items():
        missing = set(ms) - set(cand_lists.get(s, []))
        if missing:
            raise AssertionError(f"{s}: matched ids not in candidate set: {sorted(missing)[:5]}")
    mp, cp = out / "matching_results.tsv", out / "candidate_pairs.tsv"
    _write(mp, ("source1_entity_id", "matched_entity_ids"), s1_ids, match_lists)
    _write(cp, ("source1_entity_id", "candidate_entity_ids"), s1_ids, cand_lists)
    return {"matching": str(mp), "candidate": str(cp)}
