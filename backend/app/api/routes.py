"""REST API."""
from __future__ import annotations

import io
import json
import os
import platform
import shutil
import time
import zipfile

import re

import yaml
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi import Path as FPath
from fastapi.responses import FileResponse, StreamingResponse

from src.models.base_models import available_models
from src.utils.config import selected_path
from src.utils.timing import rss_mb

from ..core.settings import settings
from ..database.db import SessionLocal, TrainingJob, engine
from ..schemas.models import ID_PATTERN, PromoteRequest, RunRequest, TrainRequest
from ..services import datasets as dsvc
from ..services import explore, jobs

router = APIRouter(prefix="/api")
STARTED = time.time()
RunQ = Query(default=None, pattern=r"^RUN-\d{4}-\d{5}$")
EntityPath = r"^S[123]-[A-Za-z0-9_\-]{1,40}$"


def _nf(e: Exception):
    raise HTTPException(status_code=404, detail=str(e) or "not found")


# ----------------------------------------------------------------------------- health
@router.get("/health")
def health():
    db_ok = True
    try:
        with engine.connect() as c:
            c.exec_driver_sql("SELECT 1")
    except Exception:
        db_ok = False
    reg = jobs.registry()
    du = shutil.disk_usage(str(settings.path("runs_dir").parent))
    return {"status": "ok" if db_ok else "degraded", "database": "ok" if db_ok else "error",
            "database_backend": engine.dialect.name, "uptime_s": round(time.time() - STARTED),
            "workspace": settings.workspace or "default", "demo_mode": settings.demo_mode,
            "production_model": reg.production(), "model_versions": len(reg.list()),
            "latest_run": explore.latest_run_id(), "python": platform.python_version(),
            "platform": platform.platform(), "cpu_count": os.cpu_count(), "process_rss_mb": round(rss_mb(), 1),
            "disk_free_gb": round(du.free / 2**30, 1), "available_model_families": available_models()}


# ----------------------------------------------------------------------------- datasets
@router.get("/datasets")
def datasets():
    return dsvc.list_datasets()


@router.post("/datasets/upload")
async def upload(name: str = Form(default="Uploaded dataset", max_length=120), files: list[UploadFile] = File(...)):
    if not files or len(files) > 8:
        raise HTTPException(400, "upload 3-8 .tsv files")
    limit = settings.max_upload_mb * 2**20
    payload = {}
    for f in files:
        data = await f.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"{f.filename}: larger than {settings.max_upload_mb} MB")
        payload[f.filename or "unnamed"] = data
    res = dsvc.save_upload(name, payload)
    if not res["ok"]:
        raise HTTPException(422, {"issues": res["issues"]})
    return res


@router.get("/datasets/{dataset_id}/profile")
def dataset_profile(dataset_id: str, refresh: bool = False):
    if not re.match(ID_PATTERN, dataset_id):
        raise HTTPException(400, "bad dataset id")
    try:
        return dsvc.profile(dataset_id, refresh)
    except KeyError as e:
        _nf(e)


# ----------------------------------------------------------------------------- runs
@router.post("/resolution/run", status_code=202)
def run(req: RunRequest):
    try:
        rid = jobs.start_run(req.dataset_id, req.split, req.model_version)
    except KeyError as e:
        _nf(e)
    except (ValueError, LookupError) as e:
        raise HTTPException(409, str(e))
    return {"run_id": rid, "status": "QUEUED"}


@router.get("/resolution/runs")
def runs():
    return explore.list_runs()


@router.get("/resolution/{run_id}/status")
def run_status(run_id: str):
    r = explore.run_row(run_id)
    if r is None:
        _nf(KeyError(run_id))
    return {k: r[k] for k in ("id", "status", "stage", "progress", "created", "finished", "runtime_s", "error")}


@router.get("/resolution/{run_id}/summary")
def run_summary(run_id: str):
    p = settings.path("runs_dir") / run_id / "summary.json"
    if not p.exists() or ".." in run_id:
        _nf(KeyError(run_id))
    return json.loads(p.read_text())


# ----------------------------------------------------------------------------- entities / matches
@router.get("/entities")
def entities(run_id: str | None = RunQ, q: str = Query(default="", max_length=200), country: str = Query(default="", max_length=60),
             decision: str = Query(default="", pattern=r"^(|MATCH|NO_MATCH|REVIEW|MULTI)$"),
             truth: str = Query(default="", pattern=r"^(|wrong|right)$"),
             sort: str = Query(default="s1_id", pattern=r"^-?[a-z0-9_]{1,30}$"),
             page: int = Query(default=1, ge=1, le=100000), size: int = Query(default=25, ge=1, le=200)):
    try:
        return explore.search_entities(run_id, q, country, decision, page, size, sort, truth)
    except LookupError as e:
        raise HTTPException(409, str(e))


@router.get("/entities/{entity_id}")
def entity(entity_id: str = FPath(pattern=EntityPath), run_id: str | None = RunQ):
    try:
        return explore.entity_detail(run_id, entity_id)
    except KeyError as e:
        _nf(e)
    except LookupError as e:
        raise HTTPException(409, str(e))


@router.get("/entities/{entity_id}/matches")
def entity_matches(entity_id: str = FPath(pattern=EntityPath), run_id: str | None = RunQ):
    try:
        d = explore.entity_detail(run_id, entity_id)
        return {"run_id": d["run_id"], "entity_id": entity_id, "matches": d.get("matches", d.get("linked_to", []))}
    except KeyError as e:
        _nf(e)


@router.get("/entities/{entity_id}/candidates")
def entity_candidates(entity_id: str = FPath(pattern=EntityPath), run_id: str | None = RunQ):
    try:
        return explore.entity_candidates(run_id, entity_id)
    except KeyError as e:
        _nf(e)


MatchPath = r"^S1-[A-Za-z0-9_\-]{1,40}~S[23]-[A-Za-z0-9_\-]{1,40}$"


@router.get("/matches/{match_id}")
def match(match_id: str = FPath(pattern=MatchPath), run_id: str | None = RunQ):
    try:
        return explore.match_detail(run_id, match_id, explain=False)
    except KeyError as e:
        _nf(e)


@router.get("/matches/{match_id}/explanation")
def match_explanation(match_id: str = FPath(pattern=MatchPath), run_id: str | None = RunQ):
    try:
        return explore.match_detail(run_id, match_id, explain=True)
    except KeyError as e:
        _nf(e)


# ----------------------------------------------------------------------------- analytics / errors
@router.get("/analytics/{run_id}")
def analytics(run_id: str):
    try:
        return explore.analytics(None if run_id == "latest" else run_id)
    except (FileNotFoundError, KeyError) as e:
        _nf(e)
    except LookupError as e:
        raise HTTPException(409, str(e))


@router.get("/errors")
def errors(run_id: str | None = RunQ, type: str = Query(default="", pattern=r"^(|FALSE_POSITIVE|FALSE_NEGATIVE)$"),
           category: str = Query(default="", max_length=60), page: int = Query(default=1, ge=1), size: int = Query(default=50, ge=1, le=200)):
    try:
        return explore.errors(run_id, type, category, page, size)
    except LookupError as e:
        raise HTTPException(409, str(e))


# ----------------------------------------------------------------------------- models / experiments
@router.get("/models")
def models():
    reg = jobs.registry()
    out = []
    for v in reversed(reg.list()):
        try:
            m = reg.meta(v["version"])
        except Exception:
            m = {}
        out.append({**v, "decision": m.get("decision"), "cv_metrics": m.get("cv_metrics"), "spec": m.get("spec"),
                    "feature_count": m.get("feature_count"), "feature_importance": m.get("feature_importance"),
                    "fs_weights": m.get("fs_weights"), "train_seconds": m.get("train_seconds"),
                    "blocking_report": m.get("blocking_report")})
    return {"production": reg.production(), "versions": out, "available_families": available_models()}


@router.post("/models/train", status_code=202)
def train(req: TrainRequest):
    return {"job_id": jobs.start_training(req.model, req.promote), "status": "QUEUED"}


@router.get("/models/jobs")
def train_jobs():
    with SessionLocal() as db:
        return [{"id": j.id, "model": j.model, "status": j.status, "stage": j.stage, "version": j.version,
                 "error": j.error, "created": j.created.isoformat() if j.created else None}
                for j in db.query(TrainingJob).order_by(TrainingJob.created.desc()).limit(20)]


@router.post("/models/promote")
def promote(req: PromoteRequest):
    try:
        jobs.registry().promote(req.version)
    except KeyError as e:
        _nf(e)
    return {"production": req.version}


def _tracker():
    from src.evaluation.experiment_tracker import ExperimentTracker

    return ExperimentTracker(settings.path("experiments_dir"))


@router.get("/experiments")
def experiments():
    df = _tracker().results()
    exp_dir = settings.path("experiments_dir")
    blocking = json.loads((exp_dir / "blocking_experiments.json").read_text()) if (exp_dir / "blocking_experiments.json").exists() else None
    profile = json.loads((exp_dir / "data_profile.json").read_text()) if (exp_dir / "data_profile.json").exists() else None
    selp = selected_path()
    selected = yaml.safe_load(selp.read_text()) if selp.exists() else None
    return {"results": json.loads(df.to_json(orient="records")), "blocking": blocking, "selected": selected,
            "data_profile": profile}


@router.get("/experiments/{exp_id}")
def experiment(exp_id: str = FPath(pattern=r"^exp_\d{3,5}$")):
    e = _tracker().get(exp_id)
    if e is None:
        _nf(KeyError(exp_id))
    return json.loads(json.dumps(e, default=str))


# ----------------------------------------------------------------------------- results / export
@router.get("/results/{run_id}")
def results(run_id: str = FPath(pattern=r"^RUN-\d{4}-\d{5}$")):
    files = explore.result_files(run_id)
    if not all(p.exists() for p in files.values()):
        _nf(KeyError(run_id))
    s = json.loads((settings.path("runs_dir") / run_id / "summary.json").read_text())
    preview = files["matching_results.tsv"].read_text(encoding="utf-8").splitlines()[:12]
    return {"run_id": run_id, "files": {k: {"bytes": p.stat().st_size} for k, p in files.items()},
            "validation": s.get("validation"), "official_validation": s.get("official_validation"),
            "preview": preview, "counts": s.get("counts")}


@router.get("/results/{run_id}/download")
def download(run_id: str = FPath(pattern=r"^RUN-\d{4}-\d{5}$"),
             file: str = Query(default="bundle", pattern=r"^(bundle|matching_results\.tsv|candidate_pairs\.tsv|diagnostics\.json|evaluation\.json)$")):
    rdir = settings.path("runs_dir") / run_id
    files = explore.result_files(run_id)
    if file in files:
        if not files[file].exists():
            _nf(KeyError(file))
        return FileResponse(files[file], media_type="text/tab-separated-values", filename=file)
    s = json.loads((rdir / "summary.json").read_text())
    if file == "diagnostics.json":
        return _json_download(s, f"{run_id}_diagnostics.json")
    if file == "evaluation.json":
        ev = {"run_id": run_id, "metrics": s.get("metrics"), "metrics_note": s.get("metrics_note"),
              "error_summary": s.get("error_summary"), "blocking": s.get("blocking"), "decision": s.get("decision"),
              "model": s.get("model_version")}
        return _json_download(ev, f"{run_id}_evaluation.json")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, p in files.items():
            z.write(p, f"output/{name}")
        z.writestr("diagnostics.json", json.dumps(s, indent=2, default=str))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{run_id}_results.zip"'})


def _json_download(obj, name: str):
    data = json.dumps(obj, indent=2, default=str).encode()
    return StreamingResponse(io.BytesIO(data), media_type="application/json",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})
