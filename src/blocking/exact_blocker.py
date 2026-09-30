"""Key-based blockers: exact normalised name keys, acronyms, postal code + name token."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import PAIR_COLS, Blocker, empty_pairs, key_join


class ExactNameBlocker(Blocker):
    """Pairs whose canonical / phonetic / order-invariant / trade-name keys coincide."""

    name = "exact_name"

    def pairs(self, store, target, partition_by_country, chunk):
        r = store.records
        a, b = store.rows("S1"), store.rows(target)
        mg = int(self.params.get("max_group", 200))
        out = []
        for lk, rk in (("name_core", "name_core"), ("name_phon", "name_phon"), ("name_sorted", "name_sorted"),
                       ("name_alt", "name_core"), ("name_core", "name_alt")):
            out.append(key_join(a, r[lk].to_numpy()[a], b, r[rk].to_numpy()[b], mg))
        return pd.concat(out, ignore_index=True).drop_duplicates(["i", "j"]) if out else empty_pairs()


class AcronymBlocker(Blocker):
    """'International Business Machines' <-> 'IBM' (either direction)."""

    name = "acronym"

    def pairs(self, store, target, partition_by_country, chunk):
        r = store.records
        a, b = store.rows("S1"), store.rows(target)
        mg = int(self.params.get("max_group", 50))
        core = r["name_core"].str.replace(" ", "", regex=False).to_numpy()
        first = r["name_core"].str.split(" ").str[0].fillna("").to_numpy()
        acr = r["name_acronym"].to_numpy()
        valid = np.vectorize(lambda s: len(s) >= 2)(acr) if len(acr) else np.array([], bool)
        acr = np.where(valid, acr, "")
        out = [key_join(a, acr[a], b, first[b], mg), key_join(a, first[a], b, acr[b], mg),
               key_join(a, acr[a], b, core[b], mg), key_join(a, core[a], b, acr[b], mg)]
        return pd.concat(out, ignore_index=True).drop_duplicates(["i", "j"])


class PostalTokenBlocker(Blocker):
    """Same postcode AND at least one shared name token (explodes name tokens)."""

    name = "postal_token"

    def pairs(self, store, target, partition_by_country, chunk):
        r = store.records
        mg = int(self.params.get("max_group", 50))

        def keys(rows):
            sub = r.iloc[rows]
            sub = sub[sub.postal != ""]
            toks = sub["name_phon"].str.split(" ")
            ex = pd.DataFrame({"row": sub.index.to_numpy(), "postal": sub.postal.to_numpy(), "tok": toks.to_numpy()})
            ex = ex.explode("tok")
            ex = ex[ex.tok.str.len() >= 3]
            return ex.row.to_numpy(), (ex.postal + "|" + ex.tok).to_numpy()

        li, lk = keys(store.rows("S1"))
        ri, rk = keys(store.rows(target))
        if not len(li) or not len(ri):
            return empty_pairs()
        return key_join(li, lk, ri, rk, mg).drop_duplicates(["i", "j"])[PAIR_COLS]
