"""Rule-based transliteration of Indic scripts to Latin (no external data, no model).

Unicode's nine major Indic blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
Kannada, Malayalam) share a parallel layout inherited from ISCII: the same offset inside each block
encodes the same phoneme. One offset table therefore transliterates all of them. The output is a
phonetic approximation (e.g. "इंटरनेशनल सॉल्यूशंस" -> "intaraneshanal solyushans"), good enough for
character n-gram similarity against the Latin spelling of the same business name.
"""
from __future__ import annotations

BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

# offset -> latin, grouped by role
INDEPENDENT_VOWELS = {0x04: "a", 0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri", 0x0C: "li",
                      0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
CONSONANTS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
              0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
              0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l",
              0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g",
              0x5B: "z", 0x5C: "d", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
VOWEL_SIGNS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e",
               0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "li", 0x57: "au"}
NASAL = {0x00: "n", 0x01: "n", 0x02: "n"}
VISARGA = {0x03: "h"}
VIRAMA = 0x4D
IGNORED = {0x3C, 0x4E, 0x4F, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x64, 0x65, 0x70, 0x71}


def _decode(ch: str) -> tuple[str, int] | None:
    cp = ord(ch)
    for base in BLOCKS:
        if base <= cp < base + 0x80:
            return "x", cp - base
    return None


def has_indic(text: str) -> bool:
    return any(0x0900 <= ord(c) < 0x0D80 for c in text)


def transliterate(text: str) -> str:
    """Indic runs -> Latin; other characters pass through unchanged."""
    if not text or not has_indic(text):
        return text
    out: list[str] = []
    pending_a = False  # consonant waiting for its inherent vowel

    def flush(word_end: bool) -> None:
        nonlocal pending_a
        if pending_a and not word_end:
            out.append("a")
        pending_a = False  # schwa deletion at word end

    for i, ch in enumerate(text):
        d = _decode(ch)
        if d is None:
            flush(word_end=not ch.isalnum())
            out.append(ch)
            continue
        off = d[1]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        nxt_ind = _decode(nxt) if nxt else None
        at_end = nxt_ind is None and not (nxt and nxt.isalnum())
        if off in CONSONANTS:
            flush(word_end=False)
            out.append(CONSONANTS[off])
            pending_a = True
            if at_end:
                pending_a = False
        elif off in VOWEL_SIGNS:
            pending_a = False
            out.append(VOWEL_SIGNS[off])
        elif off == VIRAMA:
            pending_a = False
        elif off in INDEPENDENT_VOWELS:
            flush(word_end=False)
            out.append(INDEPENDENT_VOWELS[off])
        elif off in NASAL:
            flush(word_end=False)
            out.append(NASAL[off])
        elif off in VISARGA:
            flush(word_end=False)
            out.append(VISARGA[off])
        elif 0x66 <= off <= 0x6F:
            flush(word_end=False)
            out.append(str(off - 0x66))
        elif off in IGNORED:
            continue
    flush(word_end=True)
    return "".join(out)
