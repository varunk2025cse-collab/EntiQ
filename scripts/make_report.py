"""Fill the results section of Documentation_template.md from experiment artefacts (no hand-typed numbers)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path, selected_path  # noqa: E402

START, END = "<!-- AUTO:RESULTS:START -->", "<!-- AUTO:RESULTS:END -->"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def build() -> str:
    cfg = load_config()
    exp = resolve_path(cfg, "experiments_dir")
    parts = []
    prof = exp / "data_profile.json"
    if prof.exists():
        p = json.loads(prof.read_text())
        rows = []
        for split, d in p.items():
            for s, v in d["profile"]["sources"].items():
                rows.append({"split": split, "source": s, "records": v["records"], "missing name": v["missing_name_rate"],
                             "missing address": v["missing_address_rate"], "countries": ", ".join(f"{k}:{c}" for k, c in v["country_distribution"].items())})
        parts.append("### 9.1 Data profile\n\n" + md_table(pd.DataFrame(rows)))
        g = p.get("train", {}).get("profile", {}).get("ground_truth")
        if g:
            parts.append(f"Training ground truth: {g['true_pairs']} links, singleton rate {g['singleton_rate']:.2%}, "
                         f"multiplicity {g['match_multiplicity']}, records linked to more than one S1: "
                         f"{g['records_linked_to_multiple_s1']}, country agreement of true pairs {g['true_pair_country_agreement']}.")
    bj = exp / "blocking_experiments.json"
    if bj.exists():
        b = json.loads(bj.read_text())
        t = pd.DataFrame(b["table"])[["rule_set", "candidate_recall", "candidates", "avg_candidates_per_s1", "reduction_ratio", "seconds"]]
        parts.append("### 9.2 Blocking experiment\n\n" + md_table(t))
        parts.append("Leave-one-out (true links lost when a blocker is removed):\n\n" + md_table(pd.DataFrame(b["leave_one_out"])))
    rc = exp / "results.csv"
    if rc.exists():
        r = pd.read_csv(rc)
        cols = ["experiment_id", "suite", "name", "normalization", "cv_f05", "cv_precision", "cv_recall", "holdout_f05",
                "holdout_precision", "holdout_recall", "holdout_singleton_accuracy", "holdout_hard_f05", "threshold", "decision_mode", "runtime_s"]
        parts.append("### 9.3 Normalisation and model experiments (DEV 5-fold CV and single-use HOLDOUT)\n\n" + md_table(r[[c for c in cols if c in r]]))
    sp = selected_path()
    if sp.exists():
        sel = yaml.safe_load(sp.read_text())
        parts.append("### 9.4 Selected configuration (from validation evidence)\n\n```yaml\n" + yaml.safe_dump(sel.get("overrides", {}), sort_keys=False) + "```")
    reg = resolve_path(cfg, "models_dir") / "registry.json"
    if reg.exists():
        rj = json.loads(reg.read_text())
        prod = rj.get("production")
        if prod:
            meta = json.loads((resolve_path(cfg, "models_dir") / prod / "meta.json").read_text())
            parts.append(f"### 9.5 Final model `{prod}`\n\nModel `{meta['model_name']}`, decision `{json.dumps(meta['decision'])}`, "
                         f"full-train out-of-fold F0.5 {meta['cv_f05']:.4f} (precision {meta['cv_metrics']['precision']:.4f}, "
                         f"recall {meta['cv_metrics']['recall']:.4f}, singleton accuracy {meta['cv_metrics']['singleton_accuracy']:.4f}).")
    bm = exp / "benchmark.json"
    if bm.exists():
        b = json.loads(bm.read_text())
        rows = b["rows"] if isinstance(b, dict) else b
        parts.append("### 9.6 Scalability benchmark (production inference path on synthetic data)\n\n" + md_table(pd.DataFrame(rows)))
    return "\n\n".join(parts) if parts else "_No experiment artefacts found yet._"


def main() -> None:
    doc = ROOT / "Documentation_template.md"
    text = doc.read_text(encoding="utf-8")
    new = re.sub(re.escape(START) + ".*?" + re.escape(END), START + "\n" + build().replace("\\", "\\\\") + "\n" + END, text, flags=re.S)
    doc.write_text(new, encoding="utf-8")
    print("Documentation_template.md results section updated")


if __name__ == "__main__":
    main()
