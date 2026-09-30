"""Experiment registry: experiments/exp_NNN/ folders + experiments/results.csv (append-only)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

RESULT_COLUMNS = ["experiment_id", "timestamp", "suite", "name", "seed", "dataset", "dataset_kind", "dataset_fingerprint",
                  "normalization", "blocking_version", "feature_set", "model", "threshold", "decision_mode",
                  "cv_f05", "cv_precision", "cv_recall", "cv_f1", "cv_singleton_accuracy",
                  "holdout_f05", "holdout_precision", "holdout_recall", "holdout_f1", "holdout_singleton_accuracy",
                  "holdout_hard_f05", "candidate_recall", "candidates", "runtime_s", "peak_rss_mb", "notes"]


class ExperimentTracker:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.csv = self.root / "results.csv"

    def _next_id(self) -> str:
        nums = [int(p.name.split("_")[1]) for p in self.root.glob("exp_*") if p.name.split("_")[1].isdigit()]
        return f"exp_{(max(nums) + 1) if nums else 1:03d}"

    def log(self, record: dict, artifacts: dict | None = None) -> str:
        exp_id = self._next_id()
        d = self.root / exp_id
        d.mkdir(parents=True)
        record = {"experiment_id": exp_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        (d / "experiment.json").write_text(json.dumps(record, indent=2, default=str))
        for name, obj in (artifacts or {}).items():
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(d / f"{name}.csv", index=False)
            else:
                (d / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))
        row = {c: record.get(c, "") for c in RESULT_COLUMNS}
        df = pd.DataFrame([row])
        if self.csv.exists():  # reconcile schema so older files gain new columns instead of misaligning
            df = pd.concat([pd.read_csv(self.csv), df], ignore_index=True).reindex(columns=RESULT_COLUMNS)
        df.to_csv(self.csv, index=False)
        return exp_id

    def results(self) -> pd.DataFrame:
        return pd.read_csv(self.csv) if self.csv.exists() else pd.DataFrame(columns=RESULT_COLUMNS)

    def get(self, exp_id: str) -> dict | None:
        d = self.root / exp_id
        if not (d / "experiment.json").exists():
            return None
        out = json.loads((d / "experiment.json").read_text())
        out["artifacts"] = {}
        for p in d.iterdir():
            if p.name == "experiment.json":
                continue
            if p.suffix == ".json":
                out["artifacts"][p.stem] = json.loads(p.read_text())
            elif p.suffix == ".csv":
                # to_json maps NaN -> null (valid JSON), unlike to_dict
                out["artifacts"][p.stem] = json.loads(pd.read_csv(p).head(500).to_json(orient="records"))
        return out
