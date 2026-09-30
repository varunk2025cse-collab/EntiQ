"""Configurable, non-destructive normalisation of names, addresses and countries.

Each record keeps its original fields and gains derived representations:
  name_norm    canonical form (legal suffixes canonicalised, NOT removed)
  name_core    name_norm minus legal suffixes / stop words (used for matching strength)
  name_alt     trade name (from "X dba Y", "X (Y)") or name_core when absent
  name_phon    transliteration-folded key of name_core
  name_sorted  tokens of name_core sorted (order-invariant form)
  name_acronym initials of name_core tokens
  addr_norm    abbreviation-canonical address
  addr_core    addr_norm minus landmark phrases and postal code
  addr_phon    transliteration-folded key of addr_core
  addr_nums    space-separated numeric tokens
  postal       longest 5-6 digit token (open-set: works for any country's numeric postcodes)
  country_norm trimmed, accent-free, lower-case label (open set — never enumerated)

Conservative vs aggressive mode only differs in how name_core is built (aggressive also
drops generic business-type words) so both can be compared empirically.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Legal-form variants -> canonical token. Multi-word variants are matched first.
LEGAL_PHRASES = {
    "private limited": "pvt ltd", "pvt limited": "pvt ltd", "p ltd": "pvt ltd", "pvt ltd": "pvt ltd",
    "public limited company": "plc", "limited liability company": "llc", "limited liability partnership": "llp",
    "l l c": "llc", "l l p": "llp", "s a r l": "sarl", "s a s u": "sasu", "s a s": "sas", "s a": "sa",
    "and company": "and co", "and sons": "and sons", "and son": "and sons", "et fils": "et fils",
    "et cie": "and co", "and cie": "and co",
}
LEGAL_TOKENS = {
    "private": "pvt", "pvt": "pvt", "limited": "ltd", "ltd": "ltd", "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc", "company": "co", "co": "co", "llc": "llc", "llp": "llp", "plc": "plc",
    "lp": "lp", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "sarl": "sarl", "sas": "sas", "sasu": "sasu",
    "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc", "pte": "pte", "pty": "pty", "srl": "srl", "spa": "spa",
    "cie": "co", "ltda": "ltda", "oy": "oy", "ab": "ab", "kk": "kk", "bhd": "bhd", "sdn": "sdn",
}
LEGAL_CANON = set(LEGAL_TOKENS.values()) | {"sons", "fils", "et"}
NAME_STOP = {"the", "and", "of", "de", "la", "le", "les", "du", "des", "d", "l", "et", "a"}
NAME_ABBR = {
    "intl": "international", "int'l": "international", "svcs": "services", "svc": "services", "srvs": "services",
    "bros": "brothers", "med": "medical", "mfg": "manufacturing", "mgmt": "management", "assoc": "associates",
    "assocs": "associates", "dept": "department", "ctr": "center", "centre": "center", "tech": "technologies",
    "technology": "technologies", "natl": "national", "univ": "university", "hosp": "hospital", "elec": "electricals",
    "electrical": "electricals", "ent": "enterprises", "enterprise": "enterprises", "inds": "industries",
    "industry": "industries", "pharm": "pharmacy", "supp": "supplies", "consult": "consulting", "fin": "financial",
    "constr": "construction", "constns": "constructions", "logi": "logistics", "mtn": "mountain", "st": "saint",
    "ste": "sainte", "grp": "group", "hldgs": "holdings", "sys": "systems", "sol": "solutions", "solns": "solutions",
}
# Business-type words that aggressive mode strips from name_core (generic, low identity value).
GENERIC_WORDS = {"group", "services", "solutions", "enterprises", "industries", "holdings", "international",
                 "global", "company", "associates", "agency", "agencies", "traders", "trading"}

ADDR_ABBR = {
    "street": "st", "str": "st", "st": "st", "road": "rd", "rd": "rd", "avenue": "ave", "av": "ave", "ave": "ave",
    "boulevard": "blvd", "bd": "blvd", "blvd": "blvd", "boul": "blvd", "drive": "dr", "dr": "dr", "lane": "ln",
    "ln": "ln", "court": "ct", "ct": "ct", "place": "pl", "pl": "pl", "parkway": "pkwy", "pkwy": "pkwy",
    "highway": "hwy", "hwy": "hwy", "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt",
    "floor": "fl", "fl": "fl", "building": "bldg", "bldg": "bldg", "north": "n", "south": "s", "east": "e",
    "west": "w", "cross": "crs", "crs": "crs", "main": "main", "marg": "marg", "marga": "marg", "nagara": "nagar",
    "nagar": "nagar", "sector": "sec", "sec": "sec", "opposite": "opp", "opp": "opp", "near": "nr", "nr": "nr",
    "behind": "bhd", "beside": "bsd", "rue": "rue", "r": "rue", "chemin": "ch", "ch": "ch", "allee": "all",
    "all": "all", "impasse": "imp", "imp": "imp", "route": "rte", "rte": "rte", "square": "sq", "sq": "sq",
    "circle": "cir", "cir": "cir", "terrace": "ter", "ter": "ter", "station": "stn", "stn": "stn",
    "number": "no", "no": "no", "door": "door", "mahatma": "m", "gandhi": "g", "mg": "m g",
    "bengaluru": "bangalore", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "gurugram": "gurgaon", "trivandrum": "thiruvananthapuram", "bis": "b",
}
LANDMARK_MARKERS = {"nr", "opp", "bhd", "bsd"}
ADDR_NOISE = {"no", "door", "house", "flat", "plot", "the"}
ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th|er|e|eme|ème)\b")
NUM_RE = re.compile(r"\d+")
DIGIT_ALPHA_RE = re.compile(r"\b(\d+)([a-z]+)\b")
POSTAL_RE = re.compile(r"\b\d{5,6}\b")
INITIALS_RE = re.compile(r"\b(?:[a-z]\s*\.\s*){2,}")
DBA_RE = re.compile(r"\s+(?:d\s*/?\s*b\s*/?\s*a|doing business as|t\s*/\s*a|trading as|aka|a\s*/\s*k\s*/\s*a)\s+", re.I)
PAREN_RE = re.compile(r"\(([^)]*)\)")
NONWORD_RE = re.compile(r"[^0-9a-z]+")
SPACE_RE = re.compile(r"\s+")

# Transliteration folding for phonetic keys (lossy; only used as an *extra* representation).
_PHON_RULES = [(re.compile(p), r) for p, r in [
    (r"shr|shree|sree", "sri"), (r"ksh", "x"), (r"ph", "f"), (r"th", "t"), (r"dh", "d"), (r"bh", "b"),
    (r"kh", "k"), (r"gh", "g"), (r"ch", "c"), (r"sh", "s"), (r"ck", "k"), (r"q", "k"), (r"z", "s"),
    (r"w", "v"), (r"y", "i"), (r"ee", "i"), (r"oo", "u"), (r"aa", "a"), (r"(.)\1+", r"\1"), (r"[aeiou]+$", ""),
]]


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def basic_clean(s: str, accents: bool = True) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    if accents:
        s = strip_accents(s)
    s = s.lower().replace("&", " and ").replace("@", " at ")
    s = INITIALS_RE.sub(lambda m: re.sub(r"[\s.]", "", m.group(0)) + " ", s)  # a.b.c. -> abc
    s = re.sub(r"(?<=[a-z])'(?=[a-z])", "", s)  # o'brien -> obrien, mcdonald's -> mcdonalds
    s = NONWORD_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


def _join_single_letters(tokens: list[str]) -> list[str]:
    """['a','b','c','medical'] -> ['abc','medical'] (spaced-out acronyms)."""
    out, buf = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if len(buf) >= 2:
            out.append("".join(buf))
        else:
            out.extend(buf)
        buf = []
        out.append(t)
    if len(buf) >= 2:
        out.append("".join(buf))
    else:
        out.extend(buf)
    return out


def _canon_legal(text: str) -> str:
    padded = f" {text} "
    for phrase in sorted(LEGAL_PHRASES, key=len, reverse=True):
        padded = padded.replace(f" {phrase} ", f" {LEGAL_PHRASES[phrase]} ")
    return padded.strip()


def phonetic_key(s: str) -> str:
    out = []
    for tok in s.split():
        if tok.isdigit():
            out.append(tok)
            continue
        for rx, rep in _PHON_RULES:
            tok = rx.sub(rep, tok)
        if tok:
            out.append(tok)
    return " ".join(out)


@dataclass
class NormalizerConfig:
    mode: str = "conservative"  # or "aggressive"
    strip_accents: bool = True
    transliteration_keys: bool = True


class Normalizer:
    VERSION = "norm-v3"

    def __init__(self, cfg: NormalizerConfig | dict | None = None) -> None:
        if isinstance(cfg, dict):
            cfg = NormalizerConfig(**{k: v for k, v in cfg.items() if k in NormalizerConfig.__annotations__})
        self.cfg = cfg or NormalizerConfig()

    # ------------------------------------------------------------------ names
    def split_trade_name(self, raw: str) -> tuple[str, str]:
        raw = raw or ""
        m = DBA_RE.split(raw, maxsplit=1)
        if len(m) == 2 and m[1].strip():
            return m[0], m[1]
        p = PAREN_RE.search(raw)
        if p and p.group(1).strip():
            return PAREN_RE.sub(" ", raw), p.group(1)
        return raw, ""

    def name(self, raw: str) -> dict:
        main_raw, alt_raw = self.split_trade_name(raw)
        norm = self._name_norm(main_raw)
        core = self._name_core(norm)
        alt_core = self._name_core(self._name_norm(alt_raw)) if alt_raw else ""
        toks = core.split()
        return {
            "name_norm": norm,
            "name_core": core,
            "name_alt": alt_core or core,
            "name_phon": phonetic_key(core) if self.cfg.transliteration_keys else core,
            "name_sorted": " ".join(sorted(toks)),
            "name_acronym": "".join(t[0] for t in toks if not t.isdigit()) if len(toks) >= 2 else "",
            "name_legal": " ".join(t for t in norm.split() if t in LEGAL_CANON),
        }

    def _name_norm(self, raw: str) -> str:
        s = basic_clean(raw, self.cfg.strip_accents)
        toks = _join_single_letters(s.split())
        toks = [NAME_ABBR.get(t, t) for t in toks]
        s = _canon_legal(" ".join(toks))
        toks = [LEGAL_TOKENS.get(t, t) for t in s.split()]
        return " ".join(toks)

    def _name_core(self, norm: str) -> str:
        toks = [t for t in norm.split() if t not in LEGAL_CANON and t not in NAME_STOP]
        if self.cfg.mode == "aggressive":
            reduced = [t for t in toks if t not in GENERIC_WORDS]
            toks = reduced or toks
        if not toks:  # never collapse to empty: fall back to the canonical form
            return norm
        return " ".join(toks)

    # ------------------------------------------------------------------ addresses
    def _addr_tokens(self, segment: str) -> list[str]:
        s = basic_clean(segment, self.cfg.strip_accents)
        s = ORDINAL_RE.sub(r"\1", s)
        s = DIGIT_ALPHA_RE.sub(r"\1 \2", s)  # 12bis -> 12 bis, 4b -> 4 b
        toks: list[str] = []
        for t in s.split():
            toks.extend(ADDR_ABBR.get(t, t).split())
        return toks

    def address(self, raw: str) -> dict:
        # Work per comma-separated component so landmark phrases ("Near SBI ATM") end at their boundary.
        segments = [self._addr_tokens(seg) for seg in re.split(r"[,;\n]", raw or "")]
        segments = [seg for seg in segments if seg]
        toks = [t for seg in segments for t in seg]
        norm = " ".join(toks)
        postal_m = POSTAL_RE.findall(norm)
        postal = postal_m[-1] if postal_m else ""  # postcodes trail the address; street numbers lead
        core_toks, landmark = [], []
        for seg in segments:
            if seg[0] in LANDMARK_MARKERS:  # landmark component: evidence, not location identity
                landmark.extend(seg)
                continue
            for t in seg:
                if t in LANDMARK_MARKERS:  # inline marker without comma: keep marker + next 2 tokens aside
                    landmark.append(t)
                    continue
                if t in ADDR_NOISE or t == postal:
                    continue
                core_toks.append(t)
        core = " ".join(core_toks)
        nums = NUM_RE.findall(" ".join(core_toks))
        return {
            "addr_norm": norm,
            "addr_core": core,
            "addr_phon": phonetic_key(core) if self.cfg.transliteration_keys else core,
            "addr_nums": " ".join(nums),
            "addr_landmark": " ".join(landmark),
            "postal": postal,
        }

    @staticmethod
    def country(raw: str) -> str:
        return SPACE_RE.sub(" ", strip_accents(unicodedata.normalize("NFKC", raw or ""))).strip().lower()

    # ------------------------------------------------------------------ frames
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new frame with original columns + all derived representations."""
        names = pd.DataFrame([self.name(x) for x in df["business_name"].tolist()], index=df.index)
        addrs = pd.DataFrame([self.address(x) for x in df["business_address"].tolist()], index=df.index)
        out = pd.concat([df.reset_index(drop=True), names.reset_index(drop=True), addrs.reset_index(drop=True)], axis=1)
        out["country_norm"] = df["country"].map(self.country).values
        out["name_missing"] = (out["name_core"] == "").astype(np.int8)
        out["addr_missing"] = (out["addr_core"] == "").astype(np.int8)
        out["country_missing"] = (out["country_norm"] == "").astype(np.int8)
        return out

