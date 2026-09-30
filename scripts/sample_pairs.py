"""Extract a random sample of labelled S1 entities with their true S2/S3 records (streamed, memory-light).

  python scripts/sample_pairs.py --n 20000 --out data/real_pair_sample.pkl
Used for EDA only (what do true matches look like? which shard keys keep them together?).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data.data_loader import parse_id_list  # noqa: E402

READ = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=csv.QUOTE_NONE, chunksize=500_000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "real_pair_sample.pkl"))
    a = ap.parse_args()
    d = Path(a.dataset_dir) / "train"
    gt = pd.read_csv(d / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    gt = gt.sample(a.n, random_state=a.seed)
    gt["m"] = gt["matched_entity_ids"].map(parse_id_list)
    need = set(gt["source1_entity_id"]) | {x for l in gt["m"] for x in l}
    recs = []
    for k in ("source1", "source2", "source3"):
        for ch in pd.read_csv(d / f"train_{k}.tsv", **READ):
            recs.append(ch[ch["entity_id"].isin(need)])
    rec = pd.concat(recs).set_index("entity_id")
    pd.to_pickle({"gt": gt[["source1_entity_id", "m"]], "records": rec}, a.out)
    print(f"{len(gt)} S1 entities, {len(rec)} records -> {a.out}")


if __name__ == "__main__":
    main()
