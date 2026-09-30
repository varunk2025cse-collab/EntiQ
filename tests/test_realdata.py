"""Tests for components added for the real competition data: Indic transliteration, real-noise
normalisation, and geographic sharding/routing."""
import pandas as pd
import pytest

from src.data.sharding import OVERFLOW, RegionRouter, components
from src.preprocessing.indic import transliterate
from src.preprocessing.normalizer import Normalizer


@pytest.mark.parametrize("native,latin", [
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "Ram Marketing Private Limited"),          # Devanagari
    ("আলফা প্রোডাক্টস প্রাইভেট লিমিটেড", "Alfa Prodaktas Private Limited"),        # Bengali (phonetic)
])
def test_indic_name_matches_latin_core(native, latin):
    n = Normalizer()
    a, b = n.name(native), n.name(latin)
    assert a["name_legal"] == "pvt ltd"  # transliterated legal words are recognised
    # transliteration is phonetic, not exact (আলফা -> "aalapha"); the phonetic key folds aa->a, ph->f
    assert a["name_phon"].split()[0][:2] == b["name_phon"].split()[0][:2]


def test_transliteration_schwa_and_passthrough():
    assert transliterate("राम") == "ram"
    assert transliterate("उत्तर प्रदेश") == "uttar pradesh"
    assert transliterate("ABC 12") == "ABC 12"


def test_real_noise_normalisation():
    n = Normalizer()
    r = n.name("Ariadova formerly: Clark Midwest Freight LLC")
    assert r["name_core"] == "ariadova" and r["name_alt"] == "clark midwest freight"
    assert n.name("wilfordhancock.com")["name_core"] == "wilfordhancock"
    assert n.address("1521 FIFTH STREET, SIOUX FALLS, SD")["addr_core"] == n.address("1521 5th Street, Sioux Falls, SD")["addr_core"]
    assert "null" not in n.address("RATAN NIWAS, INDORE, <NULL>")["addr_core"]


def test_address_components():
    assert components("12 Rock Hill Ln, Greenburgh, New York") == ["rock hill ln", "greenburgh", "new york"]
    assert components("<NULL>, , Unit 5") == []


def test_router_keeps_true_pairs_together(synth_dir):
    from src.data.data_loader import load_split

    ds = load_split(synth_dir, "train")
    s1 = ds.s1
    router = RegionRouter(min_share=0.005, alias_support=3).fit_canonical(s1)
    allrec = pd.concat([ds.s1, ds.s2, ds.s3])
    router.learn_aliases(lambda: [allrec]).fit_name_index(s1)
    rec = allrec.set_index("entity_id")
    ok = tot = 0
    for s, ms in ds.ground_truth.items():
        a = rec.loc[s]
        ra = set(router.routes(a.business_name, a.business_address, a.country))
        for m in ms:
            b = rec.loc[m]
            ok += bool(ra & set(router.routes(b.business_name, b.business_address, b.country)))
            tot += 1
    assert tot and ok / tot > 0.9
    assert router.routes("", "", "atlantis") == [OVERFLOW]  # unseen country: never dropped
