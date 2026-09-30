"""Integrity audit of the competition dataset (streamed in chunks; nothing is assumed).

  python scripts/audit_dataset.py [--dataset-dir dataset] [--out experiments/data_audit.json]

Checks: existence/size, raw line counts vs. the published release table, header/columns, malformed rows,
UTF-8 validity, U+FFFD, blank values, duplicate ids, id prefixes, duplicate records, country distribution,
script mix, train/test overlap, ground-truth coverage & integrity, match multiplicity, singletons.
Memory: every file is read in 500K-row chunks; ids/records are kept only as 64-bit hashes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.data_loader import parse_id_list  # noqa: E402

# Published with the v1.0-dataset release README. They equal the files' TOTAL line counts (header included).
PUBLISHED_LINES = {"train_source1.tsv": 2_206_822, "train_source2.tsv": 5_034_617, "train_source3.tsv": 5_285_604,
                   "train_ground_truth.tsv": 2_206_822, "test_source1.tsv": 1_732_545, "test_source2.tsv": 4_887_274,
                   "test_source3.tsv": 5_082_317}
SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]
CHUNK = 500_000
READ = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=csv.QUOTE_NONE, encoding="utf-8",
            encoding_errors="replace", chunksize=CHUNK)


def hash_str(s: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(s, index=False).to_numpy()


def physical_scan(path: Path, n_cols: int) -> dict:
    bad_fields = bad_utf8 = fffd = crlf = lines = 0
    examples = []
    with open(path, "rb") as fh:
        header = fh.readline()
        for raw in fh:
            lines += 1
            if raw.endswith(b"\r\n"):
                crlf += 1
            try:
                s = raw.decode("utf-8")
            except UnicodeDecodeError:
                bad_utf8 += 1
                s = raw.decode("utf-8", errors="replace")
            if "�" in s:
                fffd += 1
            if s.rstrip("\r\n").count("\t") != n_cols - 1:
                bad_fields += 1
                if len(examples) < 3:
                    examples.append(s[:160])
    return {"total_lines_incl_header": lines + 1, "data_rows": lines, "malformed_rows": bad_fields, "malformed_examples": examples,
            "invalid_utf8_rows": bad_utf8, "rows_with_U+FFFD": fffd, "crlf_rows": crlf,
            "header": header.decode("utf-8", "replace").rstrip("\r\n").split("\t")}


def audit_source(path: Path, prefix: str) -> tuple[dict, np.ndarray, np.ndarray]:
    rep = {"file": path.name, "bytes": path.stat().st_size, **physical_scan(path, 4)}
    rep["published_lines"] = PUBLISHED_LINES.get(path.name)
    rep["matches_published"] = rep["published_lines"] == rep["total_lines_incl_header"]
    rep["columns_ok"] = rep["header"] == SOURCE_COLS
    id_h, rec_h = [], []
    n = bad_prefix = 0
    blanks = Counter()
    countries = Counter()
    name_len = addr_len = 0
    scripts = Counter()
    examples = None
    for ch in pd.read_csv(path, **READ):
        n += len(ch)
        ch = ch.apply(lambda c: c.str.strip())
        if examples is None:
            examples = ch.sample(min(5, len(ch)), random_state=0).to_dict("records")
        id_h.append(hash_str(ch["entity_id"]))
        rec_h.append(pd.util.hash_pandas_object(ch[["business_name", "business_address", "country"]], index=False).to_numpy())
        bad_prefix += int((~ch["entity_id"].str.startswith(prefix)).sum())
        for c in SOURCE_COLS:
            blanks[c] += int((ch[c] == "").sum())
        countries.update(ch["country"].replace("", "<blank>"))
        name_len += int(ch["business_name"].str.len().sum())
        addr_len += int(ch["business_address"].str.len().sum())
        nm = ch["business_name"]
        scripts["devanagari"] += int(nm.str.contains("[ऀ-ॿ]", regex=True).sum())
        scripts["latin_accented"] += int(nm.str.contains("[À-ɏ]", regex=True).sum())
        scripts["other_non_ascii"] += int(nm.str.contains(r"[^\x00-\x7FÀ-ɏऀ-ॿ]", regex=True).sum())
    ids = np.concatenate(id_h) if id_h else np.array([], np.uint64)
    recs = np.concatenate(rec_h) if rec_h else np.array([], np.uint64)
    u_ids, u_recs = np.unique(ids), np.unique(recs)
    rep.update({"rows_loaded": n, "duplicate_entity_ids": int(len(ids) - len(u_ids)), "bad_prefix_ids": bad_prefix,
                **{f"blank_{c}": v for c, v in blanks.items()},
                "duplicate_records_name_address_country": int(len(recs) - len(u_recs)),
                "country_distribution": dict(countries.most_common()),
                "name_length_mean": round(name_len / max(1, n), 2), "address_length_mean": round(addr_len / max(1, n), 2),
                "name_script_counts": dict(scripts), "examples": examples})
    return rep, u_ids, u_recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "data_audit.json"))
    a = ap.parse_args()
    base = Path(a.dataset_dir)
    t0 = time.time()
    report: dict = {"dataset_dir": str(base), "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": {},
                    "note": "published_lines are the release README's counts; they include the header line"}
    ids: dict[str, np.ndarray] = {}
    recs: dict[str, np.ndarray] = {}
    for split in ("train", "test"):
        for k, prefix in (("source1", "S1-"), ("source2", "S2-"), ("source3", "S3-")):
            p = base / split / f"{split}_{k}.tsv"
            if not p.exists():
                report["files"][p.name] = {"exists": False}
                print(f"{p.name}: MISSING", flush=True)
                continue
            rep, ids[f"{split}_{k}"], recs[f"{split}_{k}"] = audit_source(p, prefix)
            rep["exists"] = True
            report["files"][p.name] = rep
            print(f"{p.name:24s} data_rows={rep['rows_loaded']:>10,} lines={rep['total_lines_incl_header']:>10,} "
                  f"published={rep['published_lines']:>10,} match={rep['matches_published']} malformed={rep['malformed_rows']} "
                  f"dup_ids={rep['duplicate_entity_ids']} countries={rep['country_distribution']}", flush=True)
    gp = base / "train" / "train_ground_truth.tsv"
    if gp.exists():
        rep = {"file": gp.name, "bytes": gp.stat().st_size, **physical_scan(gp, 2)}
        rep["published_lines"] = PUBLISHED_LINES[gp.name]
        rep["matches_published"] = rep["published_lines"] == rep["total_lines_incl_header"]
        rep["columns_ok"] = rep["header"] == GT_COLS
        s1_h, linked_h, mult = [], [], Counter()
        n_rows = dup_in_list = bad_pref = s2_links = s3_links = s1_w2 = s1_w3 = singles = 0
        for ch in pd.read_csv(gp, **READ):
            n_rows += len(ch)
            s1_h.append(hash_str(ch["source1_entity_id"].str.strip()))
            lists = ch["matched_entity_ids"].map(parse_id_list)
            lens = lists.map(len)
            singles += int((lens == 0).sum())
            mult.update(lens.clip(upper=10).tolist())
            dup_in_list += int((lens != lists.map(lambda l: len(set(l)))).sum())
            n2 = lists.map(lambda l: sum(x.startswith("S2-") for x in l))
            n3 = lists.map(lambda l: sum(x.startswith("S3-") for x in l))
            s2_links += int(n2.sum()); s3_links += int(n3.sum())
            s1_w2 += int((n2 > 0).sum()); s1_w3 += int((n3 > 0).sum())
            flat = pd.Series([x for l in lists for x in l], dtype=object)
            bad_pref += int((~flat.str.startswith(("S2-", "S3-"))).sum()) if len(flat) else 0
            linked_h.append(hash_str(flat) if len(flat) else np.array([], np.uint64))
        s1g = np.concatenate(s1_h)
        lk = np.concatenate(linked_h)
        u_lk, lk_counts = np.unique(lk, return_counts=True)
        s1 = ids.get("train_source1", np.array([], np.uint64))
        s23 = np.union1d(ids.get("train_source2", np.array([], np.uint64)), ids.get("train_source3", np.array([], np.uint64)))
        s2_n, s3_n = len(ids.get("train_source2", [])), len(ids.get("train_source3", []))
        rep.update({
            "rows_loaded": n_rows, "duplicate_source1_ids": int(len(s1g) - len(np.unique(s1g))),
            "gt_s1_ids_not_in_source1": int((~np.isin(s1g, s1)).sum()), "source1_ids_without_gt_row": int((~np.isin(s1, s1g)).sum()),
            "links_total": int(len(lk)), "linked_records_unique": int(len(u_lk)),
            "linked_ids_missing_from_s2_s3": int((~np.isin(u_lk, s23)).sum()), "linked_ids_bad_prefix": bad_pref,
            "rows_with_duplicate_ids_in_list": dup_in_list, "records_linked_to_multiple_s1": int((lk_counts > 1).sum()),
            "singletons": singles, "singleton_rate": round(singles / max(1, n_rows), 5),
            "match_multiplicity_capped_at_10": {str(k): int(v) for k, v in sorted(mult.items())},
            "mean_links_per_s1": round(len(lk) / max(1, n_rows), 4),
            "s2_links": s2_links, "s3_links": s3_links, "s1_with_s2_link": s1_w2, "s1_with_s3_link": s1_w3,
            "s2_records_never_linked": int(s2_n - np.isin(ids.get("train_source2", []), u_lk).sum()),
            "s3_records_never_linked": int(s3_n - np.isin(ids.get("train_source3", []), u_lk).sum()),
        })
        rep["exists"] = True
        report["files"][gp.name] = rep
        print(f"{gp.name:24s} data_rows={n_rows:>10,} singletons={rep['singleton_rate']:.2%} links={len(lk):,}", flush=True)
    report["train_test_overlap"] = {k: {"shared_entity_ids": int(len(np.intersect1d(ids[f"train_{k}"], ids[f"test_{k}"]))),
                                        "distinct_test_records_identical_to_a_train_record": int(len(np.intersect1d(recs[f"train_{k}"], recs[f"test_{k}"])))}
                                    for k in ("source1", "source2", "source3") if f"train_{k}" in ids and f"test_{k}" in ids}
    report["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"train_test_overlap": report["train_test_overlap"], "seconds": report["seconds"]}, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
