"""Unit tests: normalisation, similarity kernels, metric, decision engine, calibration, FS model."""
import random

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import entity_fbeta, evaluate_predictions, evaluate_selection
from src.features.similarity import jaro_winkler, levenshtein, levenshtein_sim, prefix_suffix_sim
from src.matching.decision_engine import DecisionParams, select_pairs, tune_decision
from src.preprocessing.normalizer import Normalizer, phonetic_key


# ----------------------------------------------------------------------------- normalisation
@pytest.mark.parametrize("a,b", [
    ("A.B.C. Medical Supplies Pvt. Ltd.", "ABC Medical Supplies Private Limited"),
    ("A B C Medical Supplies Pvt Ltd", "abc medical supplies pvt. ltd"),
    ("Johnson & Sons Inc", "Johnson and Sons Incorporated"),
    ("PÂTISSERIE DUPONT S.A.R.L.", "Patisserie Dupont SARL"),
    ("Summit Intl Svcs LLC", "Summit International Services L.L.C."),
])
def test_name_variants_share_core(a, b):
    n = Normalizer()
    assert n.name(a)["name_core"] == n.name(b)["name_core"]


def test_legal_suffix_kept_in_canonical_form():
    r = Normalizer().name("ABC Traders Private Limited")
    assert r["name_norm"].endswith("pvt ltd") and r["name_core"] == "abc traders" and r["name_legal"] == "pvt ltd"


def test_name_never_collapses_to_empty():
    r = Normalizer().name("The Company Ltd")
    assert r["name_core"] != ""


def test_dba_trade_name_extracted():
    r = Normalizer().name("Johnson Auto Repair LLC dba QuickFix")
    assert r["name_core"] == "johnson auto repair" and r["name_alt"] == "quickfix"


def test_aggressive_mode_only_changes_core():
    c, a = Normalizer({"mode": "conservative"}), Normalizer({"mode": "aggressive"})
    x = "Summit Global Services Group LLC"
    assert c.name(x)["name_norm"] == a.name(x)["name_norm"]
    assert len(a.name(x)["name_core"]) < len(c.name(x)["name_core"])


def test_address_abbreviations_landmarks_postal():
    n = Normalizer()
    a = n.address("No. 12, 3rd Cross, MG Road, Near SBI ATM, Bengaluru, Karnataka 560034")
    b = n.address("#12, 3 Cross Rd, Mahatma Gandhi Road, Bangalore 560034")
    assert a["postal"] == b["postal"] == "560034"
    assert "sbi" not in a["addr_core"] and "sbi" in a["addr_landmark"]
    assert "bangalore" in a["addr_core"] and "bangalore" in b["addr_core"]
    assert set(a["addr_nums"].split()) >= {"12", "3"}


def test_postcode_is_trailing_token_not_street_number():
    r = Normalizer().address("12345 Main St, Austin, TX 78701")
    assert r["postal"] == "78701"


def test_country_is_open_set():
    assert Normalizer.country("  France ") == "france"
    assert Normalizer.country("Côte d'Ivoire") == "cote d'ivoire"  # never mapped to a fixed list


def test_transliteration_key():
    assert phonetic_key("shree lakshmi") == phonetic_key("sri laxmi")


# ----------------------------------------------------------------------------- similarity
def test_vectorised_levenshtein_matches_reference():
    r = random.Random(0)
    A = ["".join(r.choice("abc d") for _ in range(r.randint(0, 25))) for _ in range(500)]
    B = ["".join(r.choice("abc d") for _ in range(r.randint(0, 25))) for _ in range(500)]
    d, s = levenshtein_sim(A, B)
    assert (d == np.array([levenshtein(a, b) for a, b in zip(A, B)])).all()
    assert ((s >= 0) & (s <= 1)).all()


def test_prefix_suffix_and_jw():
    p, q = prefix_suffix_sim(["summit dental", "abc"], ["summit dent", "xbc"])
    assert p[0] == 1.0 and q[1] == pytest.approx(2 / 3)
    assert jaro_winkler("martha", "marhta") == pytest.approx(0.961, abs=1e-3)


# ----------------------------------------------------------------------------- metric
def test_official_example():
    m = evaluate_predictions({"S1-1": ["S2-47", "S2-193", "S3-812"]}, {"S1-1": ["S2-47", "S3-812"]}, ["S1-1"])
    assert m["f05"] == pytest.approx(0.714, abs=1e-3)


def test_singleton_rules():
    f = entity_fbeta(np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 2]))
    assert list(f) == [1.0, 0.0, 0.0]


def test_macro_average_includes_uncovered_true_links():
    # one entity whose single true link was never a candidate: recall 0 => F 0
    m = evaluate_selection(np.array([0]), np.array([False]), np.array([False]), np.array([1, 0]))
    assert m["f05"] == pytest.approx(0.5)  # entity 0 -> 0, entity 1 (singleton, empty) -> 1


# ----------------------------------------------------------------------------- decision engine
def _toy():
    s1 = np.array([0, 0, 0, 1, 1, 2])
    j = np.array([10, 11, 12, 10, 13, 14])
    s3 = np.array([False, False, True, False, True, False])
    p = np.array([0.95, 0.90, 0.20, 0.40, 0.35, 0.55])
    return s1, j, s3, p


def test_threshold_and_multi_match():
    s1, j, s3, p = _toy()
    sel = select_pairs(s1, j, s3, p, 3, DecisionParams(threshold=0.5, singleton_threshold=0.5))
    assert sel.tolist() == [True, True, False, False, False, True]  # S1-0 gets two links (multi-match)


def test_singleton_gate_and_cap():
    s1, j, s3, p = _toy()
    sel = select_pairs(s1, j, s3, p, 3, DecisionParams(threshold=0.5, singleton_threshold=0.6))
    assert not sel[5]  # best 0.55 < 0.6 -> NO MATCH
    sel = select_pairs(s1, j, s3, p, 3, DecisionParams(threshold=0.5, singleton_threshold=0.5, max_per_source=1))
    assert sel.tolist()[:2] == [True, False]


def test_gap_gate_blocks_ambiguous_entity():
    s1, j, s3, p = _toy()
    sel = select_pairs(s1, j, s3, p, 3, DecisionParams(threshold=0.93, singleton_threshold=0.93, min_gap=0.1))
    assert not sel.any()  # 0.95 accepted but 0.90 rejected is only 0.05 behind -> ambiguous


def test_exclusive_assignment():
    s1, j, s3, p = _toy()
    p2 = p.copy()
    p2[3] = 0.97  # record 10 is better for S1-1 than S1-0
    sel = select_pairs(s1, j, s3, p2, 3, DecisionParams(threshold=0.5, singleton_threshold=0.5, exclusive_assignment=True))
    assert not sel[0] and sel[3]


def test_expected_f_prefers_empty_for_weak_scores():
    s1 = np.array([0, 0, 1])
    sel = select_pairs(s1, np.array([1, 2, 3]), np.zeros(3, bool), np.array([0.1, 0.05, 0.97]), 2, DecisionParams(mode="expected_f"))
    assert sel.tolist() == [False, False, True]


def test_tuner_returns_best_by_f05():
    s1, j, s3, p = _toy()
    truth = np.array([True, True, False, False, False, False])
    prm, table = tune_decision(s1, j, s3, p, truth, np.array([2, 0, 0]), {"threshold": [0.3, 0.5, 0.8], "singleton_delta": [0.0, 0.1]})
    assert table["f05"].iloc[0] == 1.0
    sel = select_pairs(s1, j, s3, p, 3, prm)  # the returned params reproduce the best score
    assert evaluate_selection(s1, sel, truth, np.array([2, 0, 0]))["f05"] == 1.0


# ----------------------------------------------------------------------------- models
def test_fellegi_sunter_weights_monotone():
    from src.models.fellegi_sunter import FellegiSunterModel

    rng = np.random.RandomState(0)
    n = 2000
    y = rng.rand(n) < 0.2
    X = pd.DataFrame({"x_name_best": np.where(y, rng.uniform(.85, 1, n), rng.uniform(0, .9, n)),
                      "name_idf_jaccard": np.where(y, .8, .2), "x_addr_best": np.where(y, .9, .3),
                      "num_first_eq": y.astype(float), "num_first_conflict": (~y).astype(float) * .0,
                      "postal_eq": y.astype(float), "postal_conflict": 0.0, "country_eq": 1.0, "country_conflict": 0.0,
                      "ctx_x_weighted_rank_s1": np.where(y, 1, 3)})
    fs = FellegiSunterModel().fit(X, y)
    p = fs.predict_proba(X)[:, 1]
    assert p[y].mean() > 0.8 and p[~y].mean() < 0.2
    assert fs.weights["name"][-1] > fs.weights["name"][0]


def test_calibrator_isotonic_improves_ece():
    from src.models.calibration import Calibrator, reliability

    rng = np.random.RandomState(1)
    p = rng.rand(5000)
    y = (rng.rand(5000) < p ** 3).astype(int)  # badly calibrated raw scores
    c = Calibrator("isotonic").fit(p, y)
    assert reliability(c.transform(p), y)["ece"] < reliability(p, y)["ece"]
