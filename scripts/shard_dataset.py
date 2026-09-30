"""Shard the competition dataset by (country, region) and measure the routing recall ceiling.

  python scripts/shard_dataset.py [--dataset-dir dataset] [--out data/shards]

Writes data/shards/<split>/<country>__<region>/<split>/<split>_source{1,2,3}.tsv (+ ground truth for train,
with FULL match lists so shard-level metrics still count links routed elsewhere as misses), a SHARD.json
marker per shard, the fitted router (router.joblib) and shard_report.json with:
  * routing recall on all training links (the recall ceiling imposed by sharding)
  * shard sizes, route kinds (address region / name-routed / overflow)
Label-free: the router never sees ground truth; labels are used only to MEASURE routing recall.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.data_loader import parse_id_list  # noqa: E402
from src.data.sharding import OVERFLOW, RegionRouter  # noqa: E402

READ = dict(sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=csv.QUOTE_NONE, chunksize=250_000)
COLS = ["entity_id", "business_name", "business_address", "country"]


def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)[:40] or "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    ap.add_argument("--out", default=str(ROOT / "data" / "shards"))
    ap.add_argument("--alias-sample", type=float, default=0.35)
    a = ap.parse_args()
    base, out = Path(a.dataset_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = {(sp, k): base / sp / f"{sp}_{k}.tsv" for sp in ("train", "test") for k in ("source1", "source2", "source3")}

    # 1) canonical regions + name index from ALL Source 1 records (train + test), label-free
    s1 = pd.concat([pd.read_csv(files[(sp, "source1")], sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                                usecols=["business_name", "business_address", "country"]) for sp in ("train", "test")], ignore_index=True)
    router = RegionRouter().fit_canonical(s1)
    print(f"canonical regions: { {c: len(v) for c, v in router.canon.items()} }  ({time.time() - t0:.0f}s)", flush=True)

    # 2) aliases from co-occurrence over a sample of every source
    def chunks():
        for (sp, k), p in files.items():
            for ch in pd.read_csv(p, usecols=["business_address", "country"], **READ):
                yield ch.sample(frac=a.alias_sample, random_state=1) if k != "source1" else ch
    router.learn_aliases(chunks, rounds=2)
    router.fit_name_index(s1)
    del s1
    joblib.dump(router, out / "router.joblib")
    print(f"aliases: {router.summary()}  ({time.time() - t0:.0f}s)", flush=True)

    # 3) route every record and write shards
    route_of: dict[str, dict[str, tuple]] = {"train": {}}
    sizes = defaultdict(Counter)
    kinds = Counter()
    for (sp, k), p in files.items():
        for ch in pd.read_csv(p, **READ):
            groups = defaultdict(list)
            for row in ch.itertuples(index=False):
                ctry = row.country.strip().lower()
                routes = router.routes(row.business_name, row.business_address, ctry)
                kind = "overflow" if routes == [OVERFLOW] else ("region" if router.region(row.business_address, ctry) else "name")
                kinds[f"{sp}:{k}:{kind}"] += 1
                keys = tuple(f"{slug(ctry)}__{slug(r)}" for r in routes)
                if sp == "train":
                    route_of["train"][row.entity_id] = keys
                for key in keys:
                    groups[key].append(row)
            for key, rows in groups.items():
                d = out / sp / key / sp
                d.mkdir(parents=True, exist_ok=True)
                f = d / f"{sp}_{k}.tsv"
                new = not f.exists()
                pd.DataFrame(rows, columns=COLS).to_csv(f, sep="\t", index=False, header=new, mode="a", quoting=csv.QUOTE_NONE, escapechar="\\")
                sizes[f"{sp}/{key}"][k] += len(rows)
        print(f"routed {p.name}  ({time.time() - t0:.0f}s)", flush=True)

    # 4) ground truth per train shard (full lists) + routing recall ceiling
    gt = pd.read_csv(base / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    ro = route_of["train"]
    total = kept = 0
    lost_kind = Counter()
    per_shard_gt = defaultdict(list)
    for s1_id, lst in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        sa = ro.get(s1_id, ())
        for key in sa:
            per_shard_gt[key].append((s1_id, lst))
        for m in parse_id_list(lst):
            total += 1
            sb = ro.get(m, ())
            if set(sa) & set(sb):
                kept += 1
            else:
                lost_kind["overflow" if any(k.endswith(OVERFLOW) for k in sb) else "different_shard"] += 1
    for key, rows in per_shard_gt.items():
        d = out / "train" / key / "train"
        if d.exists():
            pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(d / "train_ground_truth.tsv", sep="\t", index=False)
    for sp in ("train", "test"):
        for key_dir in (out / sp).iterdir() if (out / sp).exists() else []:
            (key_dir / "SHARD.json").write_text(json.dumps({"source": "competition", "split": sp, "shard": key_dir.name,
                                                            "note": "subset of the official files routed by src/data/sharding.py"}))
    report = {"routing_recall_train_links": kept / max(1, total), "train_links": total, "links_lost": total - kept,
              "lost_by_kind": dict(lost_kind), "route_kinds": dict(kinds), "router": router.summary(),
              "shards": {k: dict(v) for k, v in sorted(sizes.items())}, "seconds": round(time.time() - t0, 1)}
    (out / "shard_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("routing_recall_train_links", "train_links", "links_lost", "lost_by_kind", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
