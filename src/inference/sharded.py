"""Full-scale resolution over geographic shards (scripts/shard_dataset.py) with bounded memory.

For each shard, Source 1 is processed in chunks (<= chunk_s1 entities) against the shard's complete
S2/S3 pool, through the same prepare -> score -> decide path as a normal run and the same trained bundle.
Per-chunk outputs are reduced immediately to what the submission and UI need, then merged:
  * every Source 1 entity of the split gets exactly one row (entities never retrieved -> empty lists)
  * global exclusivity: an S2/S3 record may be kept for at most one S1 (its highest score). The training
    labels show every S2/S3 record belongs to <= 1 S1, and name-routed records can appear in >1 shard.
  * candidate_pairs.tsv = the union of what the model actually scored.
"""
from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.data_loader import ERDataset, load_split
from ..matching.decision_engine import DecisionParams, select_pairs
from ..utils.logging import get_logger
from ..utils.timing import peak_rss_mb
from .pipeline import prepare
from .predictor import STORED_FEATURES, score_pairs

log = get_logger("sharded")


def shard_dirs(root: str | Path, split: str) -> list[Path]:
    base = Path(root) / split
    if not base.exists():
        return []
    dirs = [d for d in base.iterdir() if (d / split / f"{split}_source1.tsv").exists()]
    return sorted(dirs, key=lambda d: (d / split / f"{split}_source1.tsv").stat().st_size, reverse=True)


def resolve_sharded(shards_root: str | Path, split: str, bundle: dict, cfg: dict, out_dir: str | Path,
                    chunk_s1: int = 40_000, keep_top: int = 3, progress=None, only: list[str] | None = None) -> dict:
    """Run every shard; write per-shard compact results to out_dir/parts/*.pkl; return a summary."""
    out = Path(out_dir)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    rcfg = {**cfg, "normalization": bundle["normalization"], "blocking": bundle["blocking"], "features": bundle["features"]}
    prm = DecisionParams(**bundle["decision"])
    dirs = shard_dirs(shards_root, split)
    if only:
        dirs = [d for d in dirs if d.name in set(only)]
    t0 = time.perf_counter()
    stats = []
    for k, sd in enumerate(dirs):
        part = out / "parts" / f"{sd.name}.pkl"
        if part.exists():  # resumable: finished shards are skipped
            stats.append(json.loads((out / "parts" / f"{sd.name}.json").read_text()))
            continue
        ds_full = load_split(sd, split)
        s1 = ds_full.s1
        chunks = [s1.iloc[i:i + chunk_s1] for i in range(0, len(s1), chunk_s1)] or [s1]
        pieces, n_cand, t_sh = [], 0, time.perf_counter()
        for ci, s1c in enumerate(chunks):
            ds = ERDataset(ds_full.name, s1c.reset_index(drop=True), ds_full.s2, ds_full.s3, None, ds_full.files)
            prep = prepare(ds, rcfg, use_cache=False, pruner=bundle.get("pruner"))
            p_raw, p, votes = score_pairs(bundle, prep.X)
            c = prep.cand
            sel = select_pairs(c["s1_code"].to_numpy(), c["j"].to_numpy(), (c["target"] == "S3").to_numpy(), p, len(prep.s1_ids), prm)
            feats = prep.X[[f for f in STORED_FEATURES if f in prep.X.columns]].reset_index(drop=True)
            df = pd.concat([c[["s1_id", "other_id", "target", "blk_mask", "blk_count"]].reset_index(drop=True).rename(columns={"blk_count": "n_blockers"}), feats], axis=1)
            df["p"], df["p_raw"], df["selected"] = p.astype(np.float32), p_raw.astype(np.float32), sel
            for kk, v in votes.items():
                df[f"vote_{kk}"] = v.astype(np.float32)
            df["rank_s1"] = df.groupby("s1_id")["p"].rank(ascending=False, method="first").astype(np.int32)
            n_cand += len(df)
            # keep: every scored pair's ids (candidate file) + full rows for top-k and selected (UI / merge)
            pieces.append((df[["s1_id", "other_id", "target", "p", "selected"]],
                           df[(df["rank_s1"] <= keep_top) | df["selected"]]))
            del prep, c, p, p_raw, votes, df, feats
            gc.collect()
        slim = pd.concat([x[0] for x in pieces], ignore_index=True)
        rich = pd.concat([x[1] for x in pieces], ignore_index=True)
        pd.to_pickle({"slim": slim, "rich": rich, "s1_ids": s1["entity_id"].tolist()}, part)
        st = {"shard": sd.name, "s1": len(s1), "s2": len(ds_full.s2), "s3": len(ds_full.s3), "chunks": len(chunks),
              "candidates": n_cand, "links": int(slim["selected"].sum()), "seconds": round(time.perf_counter() - t_sh, 1),
              "peak_rss_mb": round(peak_rss_mb(), 0)}
        (out / "parts" / f"{sd.name}.json").write_text(json.dumps(st))
        stats.append(st)
        log.info("shard done", extra={"data": {**st, "done": f"{k + 1}/{len(dirs)}", "elapsed_s": round(time.perf_counter() - t0)}})
        if progress:
            progress(sd.name, k + 1, len(dirs))
        del ds_full, slim, rich, pieces
        gc.collect()
    return {"shards": stats, "seconds": round(time.perf_counter() - t0, 1)}


def merge_shards(out_dir: str | Path, s1_all: list[str], exclusive: bool = True) -> dict:
    """Merge per-shard parts into submission files + compact run artifacts."""
    out = Path(out_dir)
    parts = sorted((out / "parts").glob("*.pkl"))
    slim = pd.concat([pd.read_pickle(p)["slim"] for p in parts], ignore_index=True)
    # an S1 routed to several shards appears several times: keep each (s1, other) once, best score
    slim = slim.sort_values("p", ascending=False).drop_duplicates(["s1_id", "other_id"])
    sel = slim[slim["selected"]]
    if exclusive:  # each S2/S3 record belongs to at most one S1 (verified on training labels)
        sel = sel.drop_duplicates("other_id", keep="first")
    match_lists = sel.groupby("s1_id")["other_id"].apply(list).to_dict()
    cand_lists = slim.groupby("s1_id")["other_id"].apply(list).to_dict()
    for name, header, lists in (("matching_results.tsv", "matched_entity_ids", match_lists), ("candidate_pairs.tsv", "candidate_entity_ids", cand_lists)):
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"source1_entity_id\t{header}\n")
            for s in s1_all:
                fh.write(f"{s}\t{','.join(dict.fromkeys(lists.get(s, [])))}\n")
    rich = pd.concat([pd.read_pickle(p)["rich"] for p in parts], ignore_index=True)
    rich = rich.sort_values("p", ascending=False).drop_duplicates(["s1_id", "other_id"])
    kept = set(zip(sel["s1_id"], sel["other_id"]))
    rich["selected"] = [(a, b) in kept for a, b in zip(rich["s1_id"], rich["other_id"])]
    return {"s1": len(s1_all), "s1_with_candidates": len(cand_lists), "links": int(len(sel)),
            "candidates": int(len(slim)), "matched_entities": len(match_lists), "rich": rich}


def read_ids(path: str | Path) -> list[str]:
    ids = []
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, usecols=["entity_id"], chunksize=500_000):
        ids.extend(ch["entity_id"].str.strip().tolist())
    return ids
