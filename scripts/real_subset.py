"""Build a labelled REAL-DATA subset for experiments/training from whole shards (never synthetic).

  python scripts/real_subset.py --target-s1 50000 [--shards data/shards] [--out data/real_subset] [--seed 42]

Whole shards keep realistic competition (every S2/S3 record of a region competes for its S1 entities).
Shards are drawn at random (seeded) per country in proportion to each country's share of training S1.
Ground truth keeps FULL match lists, so links routed outside the subset are counted as misses (honest).
The folder carries SHARD.json {"source": "competition"} so the pipeline labels it real competition data.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

READ = dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=str(ROOT / "data" / "shards"))
    ap.add_argument("--out", default=str(ROOT / "data" / "real_subset"))
    ap.add_argument("--target-s1", type=int, default=50_000)
    ap.add_argument("--max-shard-s1", type=int, default=25_000, help="skip shards larger than this (memory)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rep = json.loads((Path(a.shards) / "shard_report.json").read_text())
    train = {k.split("/", 1)[1]: v for k, v in rep["shards"].items() if k.startswith("train/") and not k.endswith("_overflow")}
    by_country = defaultdict(list)
    for key, sz in train.items():
        if 200 <= sz.get("source1", 0) <= a.max_shard_s1:
            by_country[key.split("__")[0]].append((key, sz["source1"]))
    total_s1 = {c: sum(s for _, s in v) for c, v in by_country.items()}
    grand = sum(total_s1.values())
    rng = random.Random(a.seed)
    chosen = []
    for c, lst in sorted(by_country.items()):
        quota = a.target_s1 * total_s1[c] / grand
        rng.shuffle(lst)
        got = 0
        for key, n in lst:
            if got >= quota:
                break
            chosen.append(key)
            got += n
    out = Path(a.out) / "train"
    out.mkdir(parents=True, exist_ok=True)
    frames = defaultdict(list)
    for key in chosen:
        d = Path(a.shards) / "train" / key / "train"
        for k in ("source1", "source2", "source3", "ground_truth"):
            p = d / f"train_{k}.tsv"
            if p.exists():
                frames[k].append(pd.read_csv(p, **READ))
    for k, fs in frames.items():
        df = pd.concat(fs, ignore_index=True)
        df = df.drop_duplicates("source1_entity_id" if k == "ground_truth" else "entity_id")
        df.to_csv(out / f"train_{k}.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    meta = {"source": "competition", "kind": "real labelled subset", "shards": chosen, "seed": a.seed,
            "counts": {k: int(sum(len(f) for f in fs)) for k, fs in frames.items()}}
    (Path(a.out) / "SHARD.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("counts", "seed")}, indent=1), f"\n{len(chosen)} shards -> {a.out}")


if __name__ == "__main__":
    main()
