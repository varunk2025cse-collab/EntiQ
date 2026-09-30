"""Scalability benchmark of the production inference path on synthetic data.

For each size N (S1 records; S2/S3 are generated at similar scale) it measures, with a model
trained once on a small labelled split:
  normalise+index -> multi-pass blocking -> cascade pruning -> pair features -> scoring -> decisions
plus candidate volumes and peak memory.

  python scripts/benchmark.py --sizes 10000 50000 100000
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data.data_loader import load_split  # noqa: E402
from src.data.synthetic import Generator  # noqa: E402
from src.inference.pipeline import prepare  # noqa: E402
from src.inference.predictor import score_pairs  # noqa: E402
from src.matching.decision_engine import DecisionParams, select_pairs  # noqa: E402
from src.models.trainer import build_specs, train_final  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.timing import peak_rss_mb, rss_mb  # noqa: E402


def write_split(d: Path, n: int, split: str, seed: int, countries: dict) -> None:
    g = Generator(seed)
    s1, s2, s3, gt = g.make_split(n, countries, split)
    (d / split).mkdir(parents=True, exist_ok=True)
    for nm, df in (("source1", s1), ("source2", s2), ("source3", s3)):
        df.to_csv(d / split / f"{split}_{nm}.tsv", sep="\t", index=False)
    gt.to_csv(d / split / f"{split}_ground_truth.tsv", sep="\t", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 50_000, 100_000])
    ap.add_argument("--train-size", type=int, default=3000)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "benchmark.json"))
    args = ap.parse_args()
    cfg = load_config()
    cfg = {**cfg, "model": {**cfg["model"], "cv_folds": 3}}
    countries = {"US": 0.4, "India": 0.45, "France": 0.15}
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg["paths"] = {**cfg["paths"], "processed_dir": str(tmp / "cache"), "validation_dir": str(tmp / "val")}
        write_split(tmp / "train_data", args.train_size, "train", 5, {"US": 0.5, "India": 0.5})
        t = time.perf_counter()
        bundle, _ = train_final(prepare(load_split(tmp / "train_data", "train"), cfg, use_cache=False),
                                build_specs(cfg)["hgb"], cfg)
        train_s = round(time.perf_counter() - t, 1)
        print(f"reference model (hgb) trained on {args.train_size} S1 entities in {train_s}s")
        for n in args.sizes:
            d = tmp / f"n{n}"
            write_split(d, n, "test", 11 + n, countries)
            ds = load_split(d, "test")
            gc.collect()
            base = rss_mb()
            t0 = time.perf_counter()
            prep = prepare(ds, {**cfg, "normalization": bundle["normalization"], "blocking": bundle["blocking"],
                                "features": bundle["features"]}, use_cache=False, pruner=bundle["pruner"])
            t1 = time.perf_counter()
            _, p, _ = score_pairs({**bundle, "aux": {}}, prep.X)
            t2 = time.perf_counter()
            c = prep.cand
            sel = select_pairs(c["s1_code"].to_numpy(), c["j"].to_numpy(), (c["target"] == "S3").to_numpy(), p,
                               len(prep.s1_ids), DecisionParams(**bundle["decision"]))
            t3 = time.perf_counter()
            st = prep.timings["stages"]
            u, pre = prep.blocking["union"], prep.blocking.get("pre_pruning", {})
            row = {"s1": len(ds.s1), "s2": len(ds.s2), "s3": len(ds.s3),
                   "normalize_index_s": st["normalization_and_indexing"]["seconds"], "blocking_s": st["blocking"]["seconds"],
                   "pruning_s": st.get("pruning", {}).get("seconds"), "features_s": st["features"]["seconds"],
                   "inference_s": round(t2 - t1, 2), "decision_s": round(t3 - t2, 3), "total_s": round(t3 - t0, 1),
                   "candidates_blocked": pre.get("candidates"), "candidates_scored": u["candidates"],
                   "candidate_recall": round(u.get("candidate_recall", float("nan")), 5),
                   "links": int(sel.sum()), "entities_per_second": int(len(ds.s1) / (t3 - t0)),
                   "rss_delta_mb": round(rss_mb() - base, 0), "peak_rss_mb": round(peak_rss_mb(), 0)}
            print(json.dumps(row), flush=True)
            results.append(row)
            del prep, p, sel, ds
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"train_seconds_reference_model": train_s, "rows": results}, indent=2))
    print(f"-> {args.out}")
    _ = np


if __name__ == "__main__":
    main()
