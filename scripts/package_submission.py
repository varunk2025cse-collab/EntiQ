"""Build <team>_submission.zip in the structure required by the challenge.

  <team>_submission.zip
  ├── output/{matching_results.tsv, candidate_pairs.tsv}
  ├── code/business_entity_resolution/{src/, configs/, backend/, frontend/, tests/, scripts/, README.md, requirements.txt}
  └── Documentation_template.md

Refuses to package if the output files are missing or fail validation.
  python scripts/package_submission.py --team TeamName
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.validator_runner import validate_submission  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

CODE_DIRS = ["src", "configs", "backend", "frontend", "tests", "scripts"]
CODE_FILES = ["README.md", "requirements.txt"]
SKIP = {"__pycache__", ".pytest_cache"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--out", default=str(ROOT))
    ap.add_argument("--dataset-dir", help="competition dataset root (defaults to paths.dataset_dir)")
    a = ap.parse_args()
    cfg = load_config(overrides={"paths.dataset_dir": a.dataset_dir} if a.dataset_dir else None)
    out = resolve_path(cfg, "output_dir")
    m, c = out / "matching_results.tsv", out / "candidate_pairs.tsv"
    test_dir = resolve_path(cfg, "dataset_dir") / "test"
    v = validate_submission(m, c, test_dir)
    if v["status"] != "PASS":
        print("REFUSING TO PACKAGE — submission validation failed:")
        for i in v["issues"]:
            print("  -", i)
        sys.exit(1)
    doc = ROOT / "Documentation_template.md"
    if not doc.exists():
        print("Documentation_template.md missing")
        sys.exit(1)
    zp = Path(a.out) / f"{a.team}_submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(m, "output/matching_results.tsv")
        z.write(c, "output/candidate_pairs.tsv")
        base = "code/business_entity_resolution"
        for d in CODE_DIRS:
            for p in (ROOT / d).rglob("*"):
                if p.is_file() and not (set(p.parts) & SKIP) and p.suffix != ".pyc":
                    z.write(p, f"{base}/{p.relative_to(ROOT).as_posix()}")
        for f in CODE_FILES:
            z.write(ROOT / f, f"{base}/{f}")
        z.write(doc, "Documentation_template.md")
    print(f"PASS -> {zp} ({zp.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
