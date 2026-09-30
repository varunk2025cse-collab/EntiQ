"""Read-side queries over resolution-run artifacts (entities, candidates, matches, analytics, errors)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.blocking.blocking_pipeline import BLOCKER_LABELS, BLOCKER_ORDER
from src.explainability.explainer import explain_pair

from ..core.settings import settings
from ..database.db import Dataset, ResolutionRun, SessionLocal
from .jobs import registry


# ----------------------------------------------------------------------------- run discovery
def sync_cli_runs() -> None:
    """Register runs produced by the CLI (runs/<RUN-ID>/summary.json) so the UI can explore them."""
    runs_dir = settings.path("runs_dir")
    if not runs_dir.exists():
        return
    with SessionLocal() as db:
        if db.get(Dataset, "cli") is None:
            db.add(Dataset(id="cli", name="CLI runs", kind="cli", root=str(runs_dir), splits=[],
                           description="Runs started from the command line"))
        known = {r.id for r in db.query(ResolutionRun.id).all()}
        for d in sorted(runs_dir.glob("RUN-*")):
            s = d / "summary.json"
            if d.name in known or not s.exists():
                continue
            summ = json.loads(s.read_text())
            db.add(ResolutionRun(id=d.name, dataset_id="cli", split=summ.get("dataset", ""),
                                 model_version=summ.get("model_version"), status="COMPLETED", stage="completed",
                                 progress=100, runtime_s=(summ.get("timings", {}).get("run", {}) or {}).get("total_seconds"),
                                 summary={k: summ.get(k) for k in ("run_id", "dataset", "dataset_kind", "model_name", "model_version",
                                                                    "decision", "counts", "metrics", "validation",
                                                                    "timings", "metrics_note")}))
        db.commit()


def latest_run_id() -> str | None:
    sync_cli_runs()
    with SessionLocal() as db:
        r = db.query(ResolutionRun).filter(ResolutionRun.status == "COMPLETED").order_by(ResolutionRun.created.desc(),
                                                                                         ResolutionRun.id.desc()).first()
        return r.id if r else None


def run_dataset_kind(run_id: str | None) -> str | None:
    """Dataset identity recorded in the run summary ('synthetic' | 'competition' | 'custom' | 'unknown')."""
    if not run_id:
        return None
    p = settings.path("runs_dir") / run_id / "summary.json"
    try:
        return json.loads(p.read_text()).get("dataset_kind") or "unknown"
    except Exception:
        return "unknown"


def run_row(run_id: str) -> dict | None:
    with SessionLocal() as db:
        r = db.get(ResolutionRun, run_id)
        if r is None:
            return None
        return {"id": r.id, "dataset_id": r.dataset_id, "split": r.split, "model_version": r.model_version,
                "status": r.status, "stage": r.stage, "progress": r.progress,
                "created": r.created.isoformat() if r.created else None,
                "finished": r.finished.isoformat() if r.finished else None, "runtime_s": r.runtime_s,
                "error": r.error, "summary": r.summary}


def list_runs() -> list[dict]:
    sync_cli_runs()
    with SessionLocal() as db:
        ids = [r.id for r in db.query(ResolutionRun).order_by(ResolutionRun.created.desc(), ResolutionRun.id.desc()).all()]
    return [run_row(i) for i in ids]


# ----------------------------------------------------------------------------- artifacts
class Artifacts:
    def __init__(self, run_id: str) -> None:
        d = settings.path("runs_dir") / run_id
        if not (d / "summary.json").exists():
            raise FileNotFoundError(run_id)
        self.dir = d
        self.summary = json.loads((d / "summary.json").read_text())
        self.cand = pd.read_pickle(d / "candidates.pkl")
        self.rec = pd.read_pickle(d / "records.pkl").set_index("entity_id", drop=False)
        self.ents = pd.read_pickle(d / "entities.pkl")
        self.ents_idx = self.ents.set_index("s1_id", drop=False)
        self.errors = pd.read_pickle(d / "errors.pkl") if (d / "errors.pkl").exists() else None
        self.by_s1 = {k: v for k, v in self.cand.groupby("s1_id", sort=False).indices.items()}
        self.by_other = {k: v for k, v in self.cand.groupby("other_id", sort=False).indices.items()}
        self.has_truth = "is_true" in self.cand

    @property
    def fs_model(self):
        try:
            b = registry().load(self.summary.get("model_version"))
            return b["aux"]["fellegi_sunter"].model
        except Exception:
            return None


@lru_cache(maxsize=4)
def artifacts(run_id: str) -> Artifacts:
    return Artifacts(run_id)


def resolve_run(run_id: str | None) -> str:
    rid = run_id or latest_run_id()
    if rid is None:
        raise LookupError("no completed resolution run yet")
    return rid


# ----------------------------------------------------------------------------- helpers
def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_clean(x) for x in v]
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def blockers_of(mask: int) -> list[dict]:
    return [{"key": n, "label": BLOCKER_LABELS[n]} for b, n in enumerate(BLOCKER_ORDER) if (int(mask) >> b) & 1]


def _record(a: Artifacts, eid: str) -> dict:
    r = a.rec.loc[eid]
    return {"entity_id": eid, "source": r["source"], "business_name": r["business_name"],
            "business_address": r["business_address"], "country": r["country"], "name_normalized": r["name_norm"],
            "name_core": r["name_core"], "address_normalized": r["addr_norm"], "postal": r["postal"]}


# ----------------------------------------------------------------------------- queries
def search_entities(run_id: str | None, q: str = "", country: str = "", decision: str = "", page: int = 1,
                    size: int = 25, sort: str = "s1_id", truth: str = "") -> dict:
    rid = resolve_run(run_id)
    a = artifacts(rid)
    e = a.ents
    if q:
        ql = q.lower().strip()
        m = (e["s1_id"].str.lower().str.contains(ql, regex=False) | e["business_name"].str.lower().str.contains(ql, regex=False)
             | e["business_address"].str.lower().str.contains(ql, regex=False))
        # also find S1 entities linked to a searched S2/S3 id
        if ql.startswith(("s2-", "s3-")):
            hit = a.cand[a.cand["other_id"].str.lower() == ql]["s1_id"]
            m |= e["s1_id"].isin(hit)
        e = e[m]
    if country:
        e = e[e["country"].str.lower() == country.lower()]
    if decision == "MATCH":
        e = e[e["decision"] == "MATCH"]
    elif decision == "NO_MATCH":
        e = e[e["decision"] == "NO_MATCH"]
    elif decision == "REVIEW":
        e = e[(e["best"] >= 0.3) & (e["gap"] < 0.15)]
    elif decision == "MULTI":
        e = e[e["n_matches"] > 1]
    if truth and "f05" in e:
        e = e[e["f05"] < 1] if truth == "wrong" else e[e["f05"] >= 1]
    asc = not sort.startswith("-")
    key = sort.lstrip("-")
    if key in e.columns:
        e = e.sort_values(key, ascending=asc)
    total = len(e)
    page_df = e.iloc[(page - 1) * size: page * size]
    cols = ["s1_id", "business_name", "business_address", "country", "n_s2", "n_s3", "n_matches", "best", "second",
            "gap", "n_candidates", "decision", "confidence", "reason_codes"] + [c for c in ("n_true", "f05") if c in e]
    return {"run_id": rid, "total": total, "page": page, "size": size, "items": _records(page_df[cols])}


def entity_detail(run_id: str | None, eid: str) -> dict:
    rid = resolve_run(run_id)
    a = artifacts(rid)
    if eid not in a.rec.index:
        raise KeyError(eid)
    if not eid.startswith("S1-"):
        idx = a.by_other.get(eid, [])
        rows = a.cand.iloc[idx].sort_values("p", ascending=False)
        return {"run_id": rid, "record": _record(a, eid), "kind": "linked-record",
                "linked_to": [_record(a, s) | {"p": _clean(p), "selected": bool(sel)}
                              for s, p, sel in zip(rows["s1_id"], rows["p"], rows["selected"])]}
    e = a.ents_idx.loc[eid]
    idx = a.by_s1.get(eid, [])
    rows = a.cand.iloc[idx].sort_values("rank_s1")
    return {"run_id": rid, "kind": "s1", "record": _record(a, eid), "decision": a.summary.get("decision"),
            "entity": {k: _clean(v) for k, v in e.to_dict().items()},
            "matches": _cands(a, rows[rows["selected"]]),
            "candidates": _cands(a, rows.head(60)), "n_candidates": int(len(rows)),
            "ground_truth": (sorted(rows.loc[rows["is_true"], "other_id"].tolist()) if a.has_truth else None)}


def _cands(a: Artifacts, rows: pd.DataFrame) -> list[dict]:
    out = []
    for r in rows.itertuples(index=False):
        d = _record(a, r.other_id)
        d.update({"target": r.target, "p": _clean(r.p), "p_raw": _clean(r.p_raw), "selected": bool(r.selected),
                  "rank": int(r.rank_s1), "blockers": blockers_of(r.blk_mask), "n_blockers": int(r.n_blockers),
                  "name_similarity": _clean(r.x_name_best), "address_similarity": _clean(r.x_addr_best),
                  "is_true": (bool(r.is_true) if a.has_truth else None),
                  "votes": {k[5:]: _clean(getattr(r, k)) for k in rows.columns if k.startswith("vote_")},
                  "match_id": f"{r.s1_id}~{r.other_id}"})
        out.append(d)
    return out


def entity_candidates(run_id: str | None, eid: str) -> dict:
    rid = resolve_run(run_id)
    a = artifacts(rid)
    rows = a.cand.iloc[a.by_s1.get(eid, [])].sort_values("rank_s1")
    if eid not in a.rec.index:
        raise KeyError(eid)
    return {"run_id": rid, "record": _record(a, eid), "candidates": _cands(a, rows),
            "blocker_catalog": [{"key": n, "label": BLOCKER_LABELS[n]} for n in BLOCKER_ORDER]}


def match_detail(run_id: str | None, match_id: str, explain: bool = True) -> dict:
    rid = resolve_run(run_id)
    a = artifacts(rid)
    s1, _, other = match_id.partition("~")
    idx = [k for k in a.by_s1.get(s1, []) if a.cand.at[k, "other_id"] == other]
    if not idx:
        raise KeyError(match_id)
    row = a.cand.iloc[idx[0]]
    out = {"run_id": rid, "match_id": match_id, "s1": _record(a, s1), "candidate": _record(a, other),
           "p": _clean(row["p"]), "selected": bool(row["selected"]), "blockers": blockers_of(row["blk_mask"])}
    if explain:
        ex = explain_pair(a.rec, row, a.fs_model, a.summary.get("decision"))
        out["explanation"] = json.loads(json.dumps(ex, default=_clean))
    return out


def analytics(run_id: str | None) -> dict:
    rid = resolve_run(run_id)
    a = artifacts(rid)
    s = a.summary
    s.setdefault("dataset_kind", run_dataset_kind(rid))
    c = a.cand
    sel = c["selected"].to_numpy()
    hist_bins = np.linspace(0, 1, 11)
    conf_hist = np.histogram(c.loc[sel, "p"], bins=hist_bins)[0].tolist()
    top = c[c["rank_s1"] == 1]
    out = {
        "run_id": rid, "summary": {k: s.get(k) for k in ("counts", "decision", "model_name", "model_version", "dataset", "dataset_kind", "created",
                                                         "metrics", "metrics_note", "graph", "validation")},
        "confidence_histogram": {"bins": [f"{hist_bins[k]:.1f}-{hist_bins[k + 1]:.1f}" for k in range(10)], "counts": conf_hist},
        "best_score_histogram": np.histogram(top["p"], bins=hist_bins)[0].tolist(),
        "candidates_per_entity": np.histogram(a.ents["n_candidates"], bins=[0, 1, 5, 10, 20, 40, 80, 160, 10**9])[0].tolist(),
        "candidates_per_entity_bins": ["0", "1-4", "5-9", "10-19", "20-39", "40-79", "80-159", "160+"],
        "source_distribution": {"S2": s["counts"]["s2_links"], "S3": s["counts"]["s3_links"]},
        "country_distribution": s.get("country_distribution"), "country_match_rate": s.get("country_match_rate"),
        "links_per_entity": s.get("links_per_entity"), "reason_codes": s.get("reason_code_counts"),
        "blocking": s.get("blocking"), "timings": s.get("timings"),
        "gap_vs_decision": _gap_table(a.ents),
    }
    if a.errors is not None and len(a.errors):
        out["error_categories"] = a.errors.groupby(["error_type", "category"]).size().rename("count").reset_index().to_dict("records")
    return json.loads(json.dumps(out, default=_clean))


def _gap_table(e: pd.DataFrame) -> list[dict]:
    bins = [-0.01, 0.05, 0.15, 0.3, 0.5, 1.01]
    labels = ["<0.05", "0.05-0.15", "0.15-0.3", "0.3-0.5", ">=0.5"]
    g = pd.cut(e["gap"], bins, labels=labels)
    t = e.groupby(g, observed=False)["decision"].agg(total="size", matched=lambda d: int((d == "MATCH").sum()))
    return [{"gap": k, **{kk: int(vv) for kk, vv in v.items()}} for k, v in t.to_dict("index").items()]


def errors(run_id: str | None, etype: str = "", category: str = "", page: int = 1, size: int = 50) -> dict:
    rid = resolve_run(run_id)
    a = artifacts(rid)
    if a.errors is None:
        return {"run_id": rid, "available": False, "total": 0, "items": [], "categories": []}
    e = a.errors
    cats = e.groupby(["error_type", "category"]).size().rename("count").reset_index().to_dict("records")
    if etype:
        e = e[e["error_type"] == etype]
    if category:
        e = e[e["category"] == category]
    total = len(e)
    items = e.iloc[(page - 1) * size: page * size].copy()
    items["match_id"] = items["s1_id"] + "~" + items["other_id"]
    return {"run_id": rid, "available": True, "total": total, "categories": cats, "items": _records(items),
            "note": a.summary.get("metrics_note")}


def result_files(run_id: str) -> dict[str, Path]:
    d = settings.path("runs_dir") / run_id / "output"
    return {"matching_results.tsv": d / "matching_results.tsv", "candidate_pairs.tsv": d / "candidate_pairs.tsv"}
