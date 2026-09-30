# EntiQ — Business Identity Intelligence

End-to-end business entity resolution for the three-source challenge. For every Source 1 entity it finds **all**
matching Source 2 / Source 3 records (zero, one or many) and produces `matching_results.tsv` and
`candidate_pairs.tsv`. It also ships an explainable web application for inspecting every decision.

```
RAW TSV ─► data quality ─► normalisation ─► 9-pass blocking (union) ─► ~110 pair features
       ─► matcher (model family chosen on validation) ─► isotonic calibration
       ─► decision engine (threshold · singleton gate · score gap · multi-match · optional exclusivity)
       ─► matching_results.tsv + candidate_pairs.tsv ─► validator
```

Every stage is measured, and every design choice (normalisation mode, blocking rule set, model family, ensembling,
hard-negative mining, semantic cascade, decision policy) is **selected on held-out data** and recorded in
`experiments/` and `configs/selected.yaml`.

## Requirements

* Python 3.11 (tested on 3.11.9, Windows 11, 8 logical cores, 16 GB RAM; no GPU needed)
* `pip install -r requirements.txt`
* No internet access, external data or pretrained model downloads are used, as the fair-play rules require.

## Reproduce the competition outputs (data → blocking → matching → output)

1. Put the official files under `dataset/` exactly as shipped:
   `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`, `dataset/test/test_source{1,2,3}.tsv`.
   (Or point elsewhere with `--dataset-dir PATH`.) Optionally copy `student_resource/utils/validate_submission.py`
   to `student_resource/utils/` so the official validator also runs.
2. Run everything:

```bash
python -m src.cli all
```

This runs the steps below in order. Each can also be run alone:

| step | command | produces |
|---|---|---|
| profile + validate data | `python -m src.cli profile` | `experiments/data_profile.json` |
| blocking experiment | `python -m src.cli blocking` | `experiments/blocking_experiments.{csv,json}`, blocking choice in `configs/selected.yaml` |
| normalisation / pruning / sibling-feature A/B tests + model suite | `python -m src.cli experiment --suite standard` | `experiments/exp_NNN/`, `experiments/results.csv`, choices in `configs/selected.yaml` |
| hyper-parameter search (cv macro F0.5) | `python -m src.cli tune --models hgb,lr --trials 12` | best params + evidence in `configs/selected.yaml` |
| final training (all train data) | `python -m src.cli train` | `models/entity-model-vN/` (versioned, never overwritten) |
| test resolution + validation | `python -m src.cli predict` | `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `runs/RUN-YYYY-NNNNN/` |
| methodology tables | `python scripts/make_report.py` | fills the results section of `Documentation_template.md` |
| package | `python scripts/package_submission.py --team NAME` | `NAME_submission.zip` (refuses if validation fails) |

`predict` exits non-zero and prints the exact issues if the submission fails any rule.

Seeds are fixed (`seed: 42`). All parameters live in `configs/config.yaml`; override any of them with
`--set model.cv_folds=10`.

## Web application

```bash
python -m backend.app.main            # http://127.0.0.1:8000
```

Pages: Dashboard · Datasets (profiling, validated upload) · Resolution Runs (background jobs, live progress) ·
Entity Explorer · Match Investigation (side-by-side token diff, evidence, model votes, Fellegi-Sunter weights,
reason codes) · Candidate Explorer (which blocking passes retrieved each candidate) · Analytics · Model Laboratory
(versions, promotion, training jobs) · Experiments (every selection with its evidence) · Error Analysis ·
Export Center (validated TSV downloads, diagnostics) · System Health.

The UI never shows hard-coded numbers. Test data has no labels, so model quality is reported from **held-out
training entities** and labelled as such.

**Try it without the competition data:** a synthetic generator reproduces the documented noise patterns.
It is for development only and is never used to train the competition model.

```bash
python -m src.cli synth                                             # -> data/synthetic
python -m src.cli --workspace data/demo_workspace --dataset-dir data/synthetic all
python -m backend.app.main --workspace data/demo_workspace          # UI shows a "synthetic data" banner
```

Database: SQLite by default. Set `DATABASE_URL=postgresql+psycopg://user:pw@host/db` for PostgreSQL. The database
holds metadata (datasets, runs, metrics, accepted matches); large intermediates stay in run artifacts.

## Tests & benchmarks

```bash
python -m pytest -q                                   # unit + integration + API + end-to-end (synthetic)
python scripts/benchmark.py --sizes 10000 50000 100000
```

## Layout

```
configs/            config.yaml (all parameters) · selected.yaml (validation-driven choices + evidence)
src/data/           loader (tab-separated), validator, profiler, synthetic generator
src/preprocessing/  normalizer (conservative / aggressive; keeps original + derived forms)
src/blocking/       exact / acronym / postal keys, TF-IDF top-k (word & char), dense LSA retrieval, pipeline, experiments
src/features/       record store (cached representations), vectorised similarity kernels, name/address/pair features
src/models/         LR / RF / ET / HistGB (+XGBoost/LightGBM if installed), Fellegi-Sunter, semantic cascade,
                    stacking, calibration, hard-negative mining, trainer, versioned registry
src/matching/       decision engine, singleton evidence, graph diagnostics, candidate generator
src/evaluation/     official macro F0.5, error analysis, experiment tracker
src/inference/      cached preparation, resolution runs, submission writer, validator
src/explainability/ pair explanations (token diff, evidence, votes, FS contributions, plain-language summary)
backend/app/        FastAPI (api/, services/, database/, schemas/, core/)
frontend/           zero-build SPA (ES modules + hand-built SVG charts), served by the backend
tests/ scripts/
```
