"""Integration + end-to-end: data -> blocking -> features -> model -> decision -> TSV -> validator."""
import numpy as np
import pandas as pd

from src.blocking.blocking_pipeline import BLOCKER_ORDER, BlockingPipeline, blocking_report
from src.data.data_loader import load_split, read_tsv
from src.data.data_validator import validate_dataset
from src.features.pair_features import compute_pair_features
from src.features.record_store import RecordStore
from src.inference.validator_runner import validate_submission
from src.preprocessing.normalizer import Normalizer


def test_loader_uses_tabs_and_keeps_commas(synth_dir):
    df = read_tsv(synth_dir / "train" / "train_source1.tsv")
    assert list(df.columns) == ["entity_id", "business_name", "business_address", "country"]
    assert df["business_address"].str.contains(",").any()


def test_validation_reports_unseen_country_without_failing(synth_dir):
    ds = load_split(synth_dir, "test")
    rep = validate_dataset(ds, known_countries={"us", "india"})
    assert rep.ok and any("france" in w for w in rep.warnings)


def test_blocking_each_blocker_can_be_disabled(synth_dir, small_cfg):
    ds = load_split(synth_dir, "train")
    store = RecordStore.build(ds, Normalizer(small_cfg["normalization"]), 32)
    full, _ = BlockingPipeline(small_cfg["blocking"]).run(store)
    rep = blocking_report(full, store, ds.true_pairs())
    assert rep["union"]["candidate_recall"] > 0.95
    assert rep["union"]["reduction_ratio"] > 0.8
    cfg = {**small_cfg["blocking"], "blockers": {k: {**v, "enabled": k == "exact_name"} for k, v in small_cfg["blocking"]["blockers"].items()}}
    only, _ = BlockingPipeline(cfg).run(store)
    assert 0 < len(only) < len(full)
    assert set(np.unique(only["blk_mask"])) == {1 << BLOCKER_ORDER.index("exact_name")}


def test_features_numeric_and_aligned(synth_dir, small_cfg):
    ds = load_split(synth_dir, "train")
    store = RecordStore.build(ds, Normalizer(), 32)
    cand, _ = BlockingPipeline(small_cfg["blocking"]).run(store)
    X = compute_pair_features(store, cand.head(3000), small_cfg["features"])
    assert len(X) == 3000 and X.dtypes.eq(np.float32).all() and not X.isna().any().any()


def test_end_to_end_submission_passes_validator(synth_dir, small_cfg, workspace):
    from src.inference.pipeline import prepare
    from src.inference.predictor import resolve
    from src.models.registry import ModelRegistry
    from src.models.trainer import build_specs, evaluate_spec, make_split, train_final
    from src.utils.config import resolve_path

    train = load_split(synth_dir, "train")
    prep = prepare(train, small_cfg)
    split = make_split(prep, small_cfg)
    assert not set(split["dev"]) & set(split["holdout"])
    res = evaluate_spec(prep, build_specs(small_cfg)["hgb"], small_cfg, split)
    assert 0.5 < res["holdout_f05"] <= 1.0
    bundle, meta = train_final(prep, build_specs(small_cfg)["hgb"], small_cfg)
    version = ModelRegistry(resolve_path(small_cfg, "models_dir")).register(bundle, meta)
    assert version.startswith("entity-model-v")

    test = load_split(synth_dir, "test")
    summary = resolve(test, bundle, small_cfg, model_version=version)
    assert summary["validation"]["status"] == "PASS", summary["validation"]
    out = resolve_path(small_cfg, "runs_dir") / summary["run_id"] / "output"
    m = pd.read_csv(out / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert list(m.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert len(m) == len(test.s1) and m["source1_entity_id"].is_unique
    # France (unseen in training) is resolved like any other country
    fr = set(test.s1.loc[test.s1.country == "France", "entity_id"])
    assert fr and fr <= set(m["source1_entity_id"])
    # hidden synthetic labels: sanity check the model is far better than chance on the unseen split
    hidden = read_tsv(synth_dir / "_hidden_test_ground_truth.tsv")
    from src.data.data_loader import parse_id_list
    from src.evaluation.metrics import evaluate_predictions

    gt = {r.source1_entity_id: parse_id_list(r.matched_entity_ids) for r in hidden.itertuples()}
    pred = {r.source1_entity_id: parse_id_list(r.matched_entity_ids) for r in m.itertuples()}
    assert evaluate_predictions(pred, gt, list(m.source1_entity_id))["f05"] > 0.85


def test_pruner_transfers_from_train_to_unseen_split(synth_dir, small_cfg):
    """Regression: tau is fitted on OOF scores; inference must stay on the same score scale
    (a boosted-tree pruner once kept ~100% of test candidates while keeping ~40% on train)."""
    from src.data.data_loader import parse_id_list
    from src.inference.pipeline import prepare

    tr = prepare(load_split(synth_dir, "train"), small_cfg, use_cache=False)
    pr = tr.meta["pruner"]
    te = prepare(load_split(synth_dir, "test"), small_cfg, use_cache=False, pruner=pr)
    kept_train = pr.report["kept_fraction"]
    kept_test = len(te.cand) / te.blocking["pre_pruning"]["candidates"]
    assert kept_test < 0.6 and abs(kept_test - kept_train) < 0.25, (kept_train, kept_test)
    hidden = read_tsv(synth_dir / "_hidden_test_ground_truth.tsv")
    tp = {(r.source1_entity_id, m) for r in hidden.itertuples() for m in parse_id_list(r.matched_entity_ids)}
    found = sum((a, b) in tp for a, b in zip(te.cand.s1_id, te.cand.other_id))
    assert found / len(tp) > 0.97


def test_validator_catches_rule_violations(synth_dir, tmp_path):
    s1 = read_tsv(synth_dir / "test" / "test_source1.tsv")["entity_id"].tolist()
    s2 = read_tsv(synth_dir / "test" / "test_source2.tsv")["entity_id"].tolist()
    m, c = tmp_path / "m.tsv", tmp_path / "c.tsv"
    lines = ["source1_entity_id\tmatched_entity_ids"] + [f"{s}\t" for s in s1[1:]] + [f"{s1[1]}\t{s2[0]},{s2[0]},S2-NOPE"]
    m.write_text("\n".join(lines) + "\n")
    c.write_text("source1_entity_id\tcandidate_entity_ids\n" + "".join(f"{s}\t\n" for s in s1))
    r = validate_submission(m, c, synth_dir / "test")
    joined = " ".join(r["issues"])
    assert r["status"] == "FAIL"
    assert "missing" in joined and "duplicate" in joined and "not present" in joined
