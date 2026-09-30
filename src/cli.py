"""EntiQ command line.

  python -m src.cli profile             # data validation + profiling report (train & test)
  python -m src.cli blocking            # blocking rule-set experiment -> selects blocking config
  python -m src.cli experiment          # normalisation A/B + model suite on DEV/HOLDOUT -> selects model
  python -m src.cli tune                # randomized hyper-parameter search (scored by cv macro F0.5)
  python -m src.cli train               # final model on all training data -> models/entity-model-vN
  python -m src.cli predict             # test resolution -> output/*.tsv + validator
  python -m src.cli all                 # everything above in order
  python -m src.cli synth --out DIR     # synthetic development data (never used for the competition model)

Global options: --config PATH, --set key.path=value (repeatable), --dataset-dir DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from .utils.config import PROJECT_ROOT, load_config, parse_override, resolve_path, save_selection
from .utils.logging import get_logger, setup_logging

log = get_logger("cli")


def _cfg(args) -> dict:
    ov = dict(parse_override(s) for s in (args.set or []))
    if args.dataset_dir:
        ov["paths.dataset_dir"] = args.dataset_dir
    return load_config(args.config, ov)


def _load(cfg, split):
    from .data.data_loader import load_split

    return load_split(resolve_path(cfg, "dataset_dir"), split)


def _print_table(df: pd.DataFrame, cols=None) -> None:
    with pd.option_context("display.max_columns", 50, "display.width", 200):
        print((df[cols] if cols else df).to_string(index=False))


# ----------------------------------------------------------------------------- commands
def cmd_profile(args, cfg) -> dict:
    from .data.data_validator import validate_dataset
    from .data.profiler import profile_dataset

    out = {}
    train = _load(cfg, "train")
    known = set(pd.concat([train.s1, train.s2, train.s3])["country"].str.lower())
    for split in ("train", "test"):
        try:
            ds = train if split == "train" else _load(cfg, split)
        except FileNotFoundError as e:
            print(f"[{split}] {e}")
            continue
        out[split] = {"validation": validate_dataset(ds, known if split == "test" else None).as_dict(),
                      "profile": profile_dataset(ds)}
    rep = resolve_path(cfg, "experiments_dir") / "data_profile.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(out, indent=2, default=str))
    for split, d in out.items():
        print(f"\n== {split} ==  validation: {'OK' if d['validation']['ok'] else 'ERRORS'}")
        for e in d["validation"]["errors"] + d["validation"]["warnings"]:
            print("  -", e)
        for s, p in d["profile"]["sources"].items():
            print(f"  {s}: {p['records']} records, missing name {p['missing_name_rate']:.2%}, missing address "
                  f"{p['missing_address_rate']:.2%}, countries {p['country_distribution']}")
        if "ground_truth" in d["profile"]:
            g = d["profile"]["ground_truth"]
            print(f"  ground truth: {g['true_pairs']} links, singletons {g['singleton_rate']:.2%}, multiplicity "
                  f"{g['match_multiplicity']}, records linked to >1 S1: {g['records_linked_to_multiple_s1']}, "
                  f"true-pair country agreement {g['true_pair_country_agreement']}")
    print(f"\nreport -> {rep}")
    return out


def cmd_blocking(args, cfg) -> None:
    from .blocking.blocking_experiment import choose_rule_set, run_blocking_experiments
    from .features.record_store import RecordStore
    from .preprocessing.normalizer import Normalizer

    ds = _load(cfg, "train")
    store = RecordStore.build(ds, Normalizer(cfg["normalization"]), int(cfg["features"].get("lsa_dims", 128)))
    table, extra = run_blocking_experiments(store, ds.true_pairs(), cfg["blocking"])
    exp_dir = resolve_path(cfg, "experiments_dir")
    exp_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(exp_dir / "blocking_experiments.csv", index=False)
    (exp_dir / "blocking_experiments.json").write_text(json.dumps(
        {"table": table.to_dict("records"), "leave_one_out": extra["leave_one_out"],
         "per_blocker": {k: v["report"]["blockers"] for k, v in extra["reports"].items()}}, indent=2, default=str))
    print("\nBLOCKING EXPERIMENT")
    _print_table(table, ["rule_set", "partition_by_country", "k_scale", "candidate_recall", "candidates",
                         "avg_candidates_per_s1", "reduction_ratio", "seconds"])
    print("\nLeave-one-out (true links lost when a blocker is removed):")
    _print_table(pd.DataFrame(extra["leave_one_out"]))
    best = choose_rule_set(table)
    print(f"\nSelected: {best['rule_set']} (recall {best['candidate_recall']:.4%}, {best['candidates']} candidates)")
    if not args.no_apply:
        chosen = extra["reports"][best["rule_set"]]["config"]
        save_selection("blocking", {"blocking": chosen}, {"selected": best.to_dict(), "table": table.to_dict("records")})
        print("-> configs/selected.yaml updated (blocking)")


def cmd_experiment(args, cfg) -> list[dict]:
    from .evaluation.experiment_tracker import ExperimentTracker
    from .inference.pipeline import prepare
    from .models.trainer import SUITES, build_specs, evaluate_spec, make_split, select_best

    ds = _load(cfg, "train")
    tracker = ExperimentTracker(resolve_path(cfg, "experiments_dir"))
    results = []
    # A/B: conservative vs aggressive normalisation with the reference model
    modes = ["conservative", "aggressive"] if not args.skip_normalization else [cfg["normalization"]["mode"]]
    norm_res = {}
    ref = args.reference_model
    for mode in modes:
        c = json.loads(json.dumps(cfg))
        c["normalization"]["mode"] = mode
        prep = prepare(ds, c)
        split = make_split(prep, c)
        r = evaluate_spec(prep, build_specs(c)[ref], c, split, tracker, suite="normalization",
                          notes=f"normalization={mode}")
        norm_res[mode] = r
        results.append(r)
    best_mode = max(norm_res, key=lambda m: (round(norm_res[m]["cv_f05"], 4), norm_res[m]["cv_precision"]))
    cfg["normalization"]["mode"] = best_mode
    if len(modes) > 1:
        save_selection("normalization", {"normalization": {"mode": best_mode}},
                       {m: {"cv_f05": r["cv_f05"], "holdout_f05": r["holdout_f05"], "experiment_id": r.get("experiment_id")}
                        for m, r in norm_res.items()})
    print(f"normalization: {best_mode}  " + "  ".join(f"{m}: cv {r['cv_f05']:.4f} / holdout {r['holdout_f05']:.4f}"
                                                       for m, r in norm_res.items()))
    # A/B: cascade pruning on vs off (kept if it costs <= 0.001 cv F0.5; it buys speed)
    if not args.skip_pruning_ab and cfg["blocking"].get("pruning", {}).get("enabled"):
        c = json.loads(json.dumps(cfg))
        c["blocking"]["pruning"]["enabled"] = False
        p_off = prepare(ds, c)
        r_off = evaluate_spec(p_off, build_specs(c)[ref], c, make_split(p_off, c), tracker, suite="pruning",
                              notes="pruning=off")
        r_on = norm_res[best_mode]
        p_on = prepare(ds, cfg)
        keep = r_on["cv_f05"] >= r_off["cv_f05"] - 0.001
        ev = {"on": {"cv_f05": r_on["cv_f05"], "holdout_f05": r_on["holdout_f05"], "candidates": len(p_on.cand),
                     "candidate_recall": p_on.blocking["union"].get("candidate_recall"), "runtime_s": r_on["runtime_s"],
                     "feature_seconds": p_on.timings["stages"]["features"]["seconds"]},
              "off": {"cv_f05": r_off["cv_f05"], "holdout_f05": r_off["holdout_f05"], "candidates": len(p_off.cand),
                      "candidate_recall": p_off.blocking["union"].get("candidate_recall"), "runtime_s": r_off["runtime_s"],
                      "feature_seconds": p_off.timings["stages"]["features"]["seconds"]}}
        cfg["blocking"]["pruning"]["enabled"] = keep
        save_selection("pruning", {"blocking": {"pruning": {"enabled": keep}}}, ev)
        print(f"pruning: {'ON' if keep else 'OFF'}  " + "  ".join(
            f"{k}: cv {v['cv_f05']:.4f} holdout {v['holdout_f05']:.4f} candidates {v['candidates']} "
            f"recall {v['candidate_recall']:.4f} experiment {v['runtime_s']:.0f}s" for k, v in ev.items()))
        if not keep:
            norm_res[best_mode] = r_off
    # A/B: sibling (multi-match consistency) features on vs off
    if not args.skip_pruning_ab:
        p_ab = prepare(ds, cfg)
        s_ab = make_split(p_ab, cfg)
        sp = build_specs(cfg)
        base_drop = set(cfg["model"].get("drop_groups") or [])
        with_s = evaluate_spec(p_ab, sp["hgb_with_sibling"], cfg, s_ab, tracker, suite="sibling", notes="sibling=on")
        c_off = json.loads(json.dumps(cfg))
        c_off["model"]["drop_groups"] = sorted(base_drop | {"sibling"})
        without = (norm_res[best_mode] if "sibling" in base_drop else
                   evaluate_spec(p_ab, build_specs(c_off)[ref], c_off, s_ab, tracker, suite="sibling", notes="sibling=off"))
        keep_sib = with_s["cv_f05"] > without["cv_f05"] + 0.0005
        drop = sorted((base_drop - {"sibling"}) if keep_sib else (base_drop | {"sibling"}))
        cfg["model"]["drop_groups"] = drop
        # the reference model reused by the suite must reflect the chosen feature set
        norm_res[best_mode] = {**(with_s if keep_sib else without), "name": ref}
        save_selection("sibling_features", {"model": {"drop_groups": drop}},
                       {"on": {"cv_f05": with_s["cv_f05"], "holdout_f05": with_s["holdout_f05"]},
                        "off": {"cv_f05": without["cv_f05"], "holdout_f05": without["holdout_f05"]}, "kept": keep_sib})
        print(f"sibling features: {'ON' if keep_sib else 'OFF'}  on: cv {with_s['cv_f05']:.4f} / holdout {with_s['holdout_f05']:.4f}"
              f"  off: cv {without['cv_f05']:.4f} / holdout {without['holdout_f05']:.4f}")
    prep = prepare(ds, cfg)
    split = make_split(prep, cfg)
    specs = build_specs(cfg)
    names = [n for n in (args.models.split(",") if args.models else SUITES[args.suite]) if n in specs]
    # the reference model on the chosen normalisation was already evaluated -> reuse it
    suite_res = [norm_res[best_mode]] if ref in names else []
    suite_res += [evaluate_spec(prep, specs[n], cfg, split, tracker, suite=args.suite) for n in names if n != ref]
    results += [r for r in suite_res if r is not norm_res.get(best_mode)]
    table = pd.DataFrame([{k: r[k] for k in ("name", "cv_f05", "cv_precision", "cv_recall", "holdout_f05",
                                             "holdout_precision", "holdout_recall", "holdout_singleton_accuracy",
                                             "holdout_hard_f05", "threshold", "decision_mode", "runtime_s",
                                             "experiment_id") if k in r} for r in suite_res])
    print("\nMODEL EXPERIMENT (selection by DEV cross-validated F0.5; HOLDOUT = unbiased estimate)")
    _print_table(table.sort_values("cv_f05", ascending=False).round(4))
    best = select_best(suite_res)
    save_selection("model", {"model": {"primary": best["name"]}},
                   {"selected": best["name"], "cv_f05": best["cv_f05"], "holdout_f05": best["holdout_f05"],
                    "experiment_id": best.get("experiment_id"), "table": table.round(5).to_dict("records")})
    print(f"\nSelected model: {best['name']}  (cv F0.5 {best['cv_f05']:.4f}, holdout F0.5 {best['holdout_f05']:.4f})"
          f"\n-> configs/selected.yaml updated (normalization, model)")
    return results


def cmd_tune(args, cfg) -> None:
    from .evaluation.experiment_tracker import ExperimentTracker
    from .inference.pipeline import prepare
    from .models.trainer import make_split, random_search

    ds = _load(cfg, "train")
    prep = prepare(ds, cfg)
    split = make_split(prep, cfg)
    tracker = ExperimentTracker(resolve_path(cfg, "experiments_dir"))
    for kind in args.models.split(","):
        best, rows = random_search(prep, split, cfg, kind, args.trials, tracker)
        inc = rows[0]["cv_f05"]
        print()
        print(f"HPO {kind}: incumbent cv F0.5 {inc:.4f} -> best {best['cv_f05']:.4f} (trial {best['trial']}) {best['params']}")
        if not best["incumbent"] and best["cv_f05"] > inc:
            save_selection(f"hpo_{kind}", {"model": {"params": {kind: best["params"]}}},
                           {"incumbent_cv_f05": inc, "best_cv_f05": best["cv_f05"], "trials": len(rows)})
            print(f"-> configs/selected.yaml updated (model.params.{kind})")


def cmd_train(args, cfg) -> str:
    from .evaluation.experiment_tracker import ExperimentTracker
    from .inference.pipeline import prepare
    from .models.registry import ModelRegistry
    from .models.trainer import build_specs, train_final

    ds = _load(cfg, "train")
    name = args.model or cfg["model"]["primary"]
    spec = build_specs(cfg)[name]
    prep = prepare(ds, cfg)
    bundle, meta = train_final(prep, spec, cfg)
    res = ExperimentTracker(resolve_path(cfg, "experiments_dir")).results()
    hit = res[res["name"] == name].tail(1) if len(res) else res
    if len(hit):
        meta["holdout_f05"] = float(hit["holdout_f05"].iloc[0])
        meta["experiment_id"] = hit["experiment_id"].iloc[0]
    meta["training_prep_timings"] = prep.timings
    meta["blocking_report"] = prep.blocking["union"]
    version = ModelRegistry(resolve_path(cfg, "models_dir")).register(bundle, meta, promote=not args.no_promote)
    print(f"registered {version}: model={name} cv F0.5={meta['cv_f05']:.4f} decision={meta['decision']}")
    return version


def cmd_predict(args, cfg) -> dict:
    from .inference.predictor import resolve
    from .models.registry import ModelRegistry

    reg = ModelRegistry(resolve_path(cfg, "models_dir"))
    version = args.version or reg.production()
    bundle = reg.load(version)
    ds = _load(cfg, args.split)
    summary = resolve(ds, bundle, cfg, model_version=version, write_to_output=(args.split == "test"))
    c = summary["counts"]
    print(f"\n{summary['run_id']} [{args.split}] model={version}: {c['s1']} S1 entities, {c['candidates']} candidates, "
          f"{c['links']} links, {c['matched_entities']} matched, {c['singletons']} singletons")
    v = summary["validation"]
    print(f"submission validation: {v['status']}")
    for i in v.get("issues", []):
        print("  ISSUE:", i)
    for w in v.get("warnings", []):
        print("  warning:", w)
    if summary.get("official_validation"):
        print("official validator:", summary["official_validation"]["status"])
        print(summary["official_validation"]["stdout"])
    if summary.get("metrics"):
        m = summary["metrics"]
        print(f"metrics ({summary['metrics_note']}): F0.5 {m['f05']:.4f} P {m['precision']:.4f} R {m['recall']:.4f}")
    if args.split == "test":
        print(f"output -> {resolve_path(cfg, 'output_dir')}")
    if v["status"] != "PASS":
        sys.exit(1)
    return summary


def cmd_all(args, cfg) -> None:
    cmd_profile(args, cfg)
    cmd_blocking(args, cfg)
    cfg = _cfg(args)  # reload with the blocking selection applied
    cmd_experiment(args, cfg)
    if args.tune_trials > 0:  # tune the stack's gradient-boosting + linear bases on validation F0.5
        args.models, args.trials = "hgb,lr", args.tune_trials
        cmd_tune(args, _cfg(args))
        args.models = None
    cfg = _cfg(args)
    cmd_train(args, cfg)
    args.split = "test"
    cmd_predict(args, _cfg(args))


def cmd_synth(args, cfg) -> None:
    from .data.synthetic import write_synthetic

    out = write_synthetic(args.out, args.n_train, args.n_test, args.seed)
    print(f"synthetic data -> {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="entiq", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--set", action="append", help="override: key.path=value")
    p.add_argument("--dataset-dir")
    p.add_argument("--workspace", help="sandbox dir for all generated artefacts (e.g. synthetic runs)")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("profile")
    b = sub.add_parser("blocking")
    b.add_argument("--no-apply", action="store_true")
    for name in ("experiment", "all"):
        e = sub.add_parser(name)
        e.add_argument("--split", default="test")
        e.add_argument("--suite", default="standard")
        e.add_argument("--models", help="comma list overriding the suite")
        e.add_argument("--reference-model", default="hgb")
        e.add_argument("--skip-normalization", action="store_true")
        e.add_argument("--skip-pruning-ab", action="store_true")
        e.add_argument("--tune-trials", type=int, default=8)
        e.add_argument("--no-apply", action="store_true")
        e.add_argument("--model")
        e.add_argument("--no-promote", action="store_true")
        e.add_argument("--version")
    hp = sub.add_parser("tune")
    hp.add_argument("--models", default="hgb,lr")
    hp.add_argument("--trials", type=int, default=12)
    t = sub.add_parser("train")
    t.add_argument("--model")
    t.add_argument("--no-promote", action="store_true")
    pr = sub.add_parser("predict")
    pr.add_argument("--split", default="test")
    pr.add_argument("--version")
    s = sub.add_parser("synth")
    s.add_argument("--out", default=str(PROJECT_ROOT / "data" / "synthetic"))
    s.add_argument("--n-train", type=int, default=3000)
    s.add_argument("--n-test", type=int, default=2000)
    s.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    if args.workspace:
        os.environ["ENTIQ_WORKSPACE"] = str(Path(args.workspace).resolve())
    cfg = _cfg(args)
    setup_logging(args.log_level, resolve_path(cfg, "experiments_dir") / "logs" / f"{time.strftime('%Y%m%d')}.jsonl")
    {"profile": cmd_profile, "blocking": cmd_blocking, "experiment": cmd_experiment, "train": cmd_train,
     "predict": cmd_predict, "all": cmd_all, "synth": cmd_synth, "tune": cmd_tune}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
