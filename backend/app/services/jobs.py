"""Background execution of resolution runs and model training (never blocks HTTP requests)."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from src.data.data_loader import load_split
from src.inference.predictor import next_run_id, resolve
from src.models.registry import ModelRegistry
from src.utils.logging import get_logger

from ..core.settings import settings
from ..database.db import Match, Metric, ResolutionRun, SessionLocal, TrainingJob, now
from .datasets import get_dataset

log = get_logger("jobs")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="entiq-job")  # ML jobs are CPU-heavy: serialise
_id_lock = threading.Lock()


def registry() -> ModelRegistry:
    return ModelRegistry(settings.path("models_dir"))


def start_run(dataset_id: str, split: str, model_version: str | None) -> str:
    ds_row = get_dataset(dataset_id)
    if ds_row is None:
        raise KeyError(f"dataset {dataset_id} not found")
    if split not in (ds_row.splits or []):
        raise ValueError(f"dataset {dataset_id} has no '{split}' split")
    reg = registry()
    version = model_version or reg.production()
    if version is None:
        raise LookupError("no trained model is registered yet — train one first (Model Laboratory or `python -m src.cli train`)")
    with _id_lock, SessionLocal() as db:
        runs_dir = settings.path("runs_dir")
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_id = next_run_id(runs_dir)
        (runs_dir / run_id).mkdir(parents=True, exist_ok=True)  # reserve the id
        db.add(ResolutionRun(id=run_id, dataset_id=dataset_id, split=split, model_version=version, status="QUEUED"))
        db.commit()
    _executor.submit(_execute_run, run_id, ds_row.root, split, version)
    return run_id


def _set(run_id: str, **kw) -> None:
    with SessionLocal() as db:
        r = db.get(ResolutionRun, run_id)
        for k, v in kw.items():
            setattr(r, k, v)
        db.commit()


def _execute_run(run_id: str, root: str, split: str, version: str) -> None:
    t0 = time.perf_counter()
    try:
        _set(run_id, status="RUNNING", stage="loading data", progress=2)
        ds = load_split(root, split)
        bundle = registry().load(version)
        summary = resolve(ds, bundle, settings.cfg, run_id=run_id, model_version=version,
                          progress=lambda stage, pct: _set(run_id, stage=stage, progress=int(pct)))
        _store_results(run_id, summary)
        _set(run_id, status="COMPLETED", stage="completed", progress=100, finished=now(),
             runtime_s=round(time.perf_counter() - t0, 2), summary=_light(summary))
    except Exception as e:  # surface the failure to the UI, keep the server alive
        log.error("run failed", extra={"data": {"run_id": run_id, "error": str(e)}})
        _set(run_id, status="FAILED", error=f"{e}\n{traceback.format_exc()[-3000:]}", finished=now(),
             runtime_s=round(time.perf_counter() - t0, 2))


def _light(summary: dict) -> dict:
    keep = ("run_id", "dataset", "dataset_kind", "model_name", "model_version", "decision", "counts", "metrics", "validation",
            "timings", "dataset_fingerprint", "metrics_note")
    return {k: summary.get(k) for k in keep}


def _store_results(run_id: str, summary: dict) -> None:
    import pandas as pd

    cand = pd.read_pickle(settings.path("runs_dir") / run_id / "candidates.pkl")
    sel = cand[cand["selected"]]
    with SessionLocal() as db:
        db.bulk_save_objects([Match(run_id=run_id, s1_id=r.s1_id, other_id=r.other_id, target=r.target,
                                    confidence=float(r.p), correct=(bool(r.is_true) if "is_true" in sel else None))
                              for r in sel.itertuples()])
        flat = {f"count_{k}": v for k, v in summary["counts"].items()}
        for k, v in (summary.get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                flat[k] = v
        db.bulk_save_objects([Metric(run_id=run_id, name=k, value=float(v)) for k, v in flat.items()])
        db.commit()


# ----------------------------------------------------------------------------- training
def start_training(model: str | None, promote: bool) -> str:
    job_id = f"TRAIN-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    with SessionLocal() as db:
        db.add(TrainingJob(id=job_id, model=model or settings.cfg["model"]["primary"], status="QUEUED"))
        db.commit()
    _executor.submit(_execute_training, job_id, model, promote)
    return job_id


def _execute_training(job_id: str, model: str | None, promote: bool) -> None:
    from src.inference.pipeline import prepare
    from src.models.trainer import build_specs, train_final
    from src.utils.config import resolve_path

    def upd(**kw):
        with SessionLocal() as db:
            j = db.get(TrainingJob, job_id)
            for k, v in kw.items():
                setattr(j, k, v)
            db.commit()

    try:
        cfg = settings.cfg
        upd(status="RUNNING", stage="preparing training data")
        ds = load_split(resolve_path(cfg, "dataset_dir"), "train")
        prep = prepare(ds, cfg)
        name = model or cfg["model"]["primary"]
        upd(stage=f"cross-validating and fitting {name}")
        bundle, meta = train_final(prep, build_specs(cfg)[name], cfg)
        meta["blocking_report"] = prep.blocking["union"]
        version = registry().register(bundle, meta, promote=promote)  # new version; never overwrites
        upd(status="COMPLETED", stage="registered", version=version, finished=now())
    except Exception as e:
        upd(status="FAILED", error=f"{e}\n{traceback.format_exc()[-3000:]}", finished=now())
