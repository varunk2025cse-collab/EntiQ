"""Submission validation: built-in re-implementation of every documented rule, plus the official
utils/validate_submission.py when it is available (its verdict is authoritative)."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def _read(path: Path, id_col: str, list_col: str, issues: list[str]) -> dict[str, list[str]] | None:
    if not path.exists():
        issues.append(f"{path.name}: file not found")
        return None
    rows: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        if header != [id_col, list_col]:
            issues.append(f"{path.name}: header must be '{id_col}<TAB>{list_col}', got {header}")
        for n, line in enumerate(fh, start=2):
            line = line.rstrip("\r\n")
            parts = line.split("\t")
            if len(parts) != 2:
                issues.append(f"{path.name}:{n}: expected 2 tab-separated columns, got {len(parts)}")
                continue
            s1, lst = parts
            if s1 in rows:
                issues.append(f"{path.name}:{n}: duplicate source1_entity_id {s1}")
            ids = [x for x in lst.split(",")] if lst else []
            if any(x == "" or x != x.strip() for x in ids):
                issues.append(f"{path.name}:{n}: empty or whitespace-padded id in list")
            if len(ids) != len(set(ids)):
                issues.append(f"{path.name}:{n}: duplicate ids within list for {s1}")
            rows[s1] = ids
    return rows


def _ids(path: Path) -> set[str]:
    with open(path, encoding="utf-8", newline="") as fh:
        r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(r)
        k = header.index("entity_id")
        return {row[k].strip() for row in r if row}


def validate_submission(matching: str | Path, candidate: str | Path, test_dir: str | Path) -> dict:
    issues: list[str] = []
    warnings: list[str] = []
    test_dir = Path(test_dir)
    split = test_dir.name
    s1 = _ids(test_dir / f"{split}_source1.tsv")
    other = _ids(test_dir / f"{split}_source2.tsv") | _ids(test_dir / f"{split}_source3.tsv")
    m = _read(Path(matching), "source1_entity_id", "matched_entity_ids", issues)
    c = _read(Path(candidate), "source1_entity_id", "candidate_entity_ids", issues)
    for name, rows in (("matching_results.tsv", m), ("candidate_pairs.tsv", c)):
        if rows is None:
            continue
        missing = s1 - set(rows)
        extra = set(rows) - s1
        if missing:
            issues.append(f"{name}: {len(missing)} Source 1 entities missing (e.g. {sorted(missing)[:3]})")
        if extra:
            issues.append(f"{name}: {len(extra)} rows for unknown Source 1 ids (e.g. {sorted(extra)[:3]})")
        bad = {x for ids in rows.values() for x in ids if x not in other}
        if bad:
            issues.append(f"{name}: {len(bad)} ids not present in test S2/S3 (e.g. {sorted(bad)[:3]})")
        s1_refs = {x for ids in rows.values() for x in ids if x.startswith("S1-")}
        if s1_refs:
            issues.append(f"{name}: lists reference Source 1 ids (self-matches) e.g. {sorted(s1_refs)[:3]}")
    if m is not None and c is not None:
        not_cand = sum(len(set(v) - set(c.get(k, []))) for k, v in m.items())
        if not_cand:
            warnings.append(f"{not_cand} matched ids never appear in candidate_pairs.tsv (pipeline bug signal)")
    result = {"status": "PASS" if not issues else "FAIL", "issues": issues, "warnings": warnings,
              "matching_rows": len(m or {}), "candidate_rows": len(c or {}),
              "matched_links": sum(len(v) for v in (m or {}).values()),
              "candidate_links": sum(len(v) for v in (c or {}).values())}
    return result


def run_official(script: str | Path, matching: str | Path, candidate: str | Path, test_dir: str | Path) -> dict | None:
    script = Path(script)
    if not script.exists():
        return None
    proc = subprocess.run([sys.executable, str(script), "--matching", str(matching), "--candidate", str(candidate),
                           "--test-dir", str(test_dir)], capture_output=True, text=True, cwd=script.parent.parent)
    return {"exit_code": proc.returncode, "status": "PASS" if proc.returncode == 0 else "FAIL",
            "stdout": proc.stdout[-5000:], "stderr": proc.stderr[-2000:]}
