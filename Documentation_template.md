# Business Entity Resolution — Methodology

**System:** EntiQ, a cascaded entity-resolution pipeline with an explainable web application.
**Reproduce:** `pip install -r requirements.txt && python -m src.cli all` (see `code/business_entity_resolution/README.md`).
**Result tables** in section 9 are generated from the experiment logs by `python scripts/make_report.py`. No number in them is typed by hand.

---

## 1. Problem framing

For each Source 1 (S1) entity we must return every Source 2/3 record describing the same business. The metric is
F0.5 per S1 entity, macro-averaged over all S1 entities, singletons included. Three consequences shaped the design:

1. **Precision is worth twice recall.** A false merge on a true singleton scores 0 for that entity, and a correct empty
   prediction scores a full 1.0. So the system must be able to abstain ("NO MATCH").
2. **The unit of evaluation is the entity, not the pair.** Pair-level AUC or accuracy can improve while macro F0.5 gets
   worse. Every choice is therefore scored with the exact official metric, which we implemented in closed form:
   `F = (1+β²)·tp / (β²·n_true + n_pred)`. Empty/empty gives 1, and any link on a singleton gives 0. True links that
   blocking never retrieved still count in `n_true`.
3. **One-to-many is allowed.** Nothing assumes one match per source. Caps and exclusivity exist only as options that
   validation may switch on.

## 2. Data handling

* All files are read with `sep="\t"`, `quoting=QUOTE_NONE` and string dtype. Addresses and id lists contain commas,
  and quotes are literal characters.
* The validator checks the required columns, id uniqueness, `S1-/S2-/S3-` prefixes, empty fields, duplicate records,
  encoding damage (U+FFFD), and whether ground-truth ids exist. It **reports and never mutates** the data. The
  original files are never modified.
* The profiler computes per-source volumes, missing rates, duplicate rates, length distributions, country
  distribution, match multiplicity, singleton rate, how many records link to several S1s, and country agreement of
  true pairs. All of these come from the actual files (`experiments/data_profile.json`).
* **Country is an open-set string.** It is normalised (trim, NFKC, accent fold, lower-case) and used only through
  *equality/conflict/missing* features and optional partitioning of retrieval. Nothing is one-hot encoded or
  enumerated. France, which is absent from training, flows through the same code path. Records with a missing
  country are compared against every partition, so they are never dropped.

## 3. Normalisation (non-destructive)

Each record keeps its original fields and gains derived representations:

| field | content |
|---|---|
| `name_norm` | NFKC + accent fold, case fold, `&`→`and`, `A.B.C.`/`A B C`→`abc`, apostrophes, expansion of common abbreviations (`intl`, `svcs`, `bros`, `med`…), **legal forms canonicalised, not removed** (`Private Limited`/`Pvt. Ltd.`→`pvt ltd`, `L.L.C.`→`llc`, `S.A.R.L.`→`sarl`) |
| `name_core` | `name_norm` minus legal forms and stop words. It never collapses to empty; if it would, it falls back to `name_norm` |
| `name_alt` | trade name from `X dba Y`, `t/a`, `aka`, or parentheses |
| `name_phon` | transliteration-folded key (`shri/shree/sree→sri`, `ksh→x`, `ph→f`, doubled letters, trailing vowels) |
| `name_sorted`, `name_acronym` | word-order-invariant form; initials |
| `addr_norm` / `addr_core` | abbreviation canonicalisation (`Road/Rd`, `Avenue/Av./Ave`, `Boulevard/Bd`, `Suite/Ste/#`…), ordinals (`3rd`→`3`), `12bis`→`12 b`, landmark components (`Near SBI ATM`, `Opp.`, `Behind`) moved to `addr_landmark` using the comma boundaries |
| `addr_nums`, `postal` | numeric tokens; last 5–6 digit token as the postcode (the street number leads, the postcode trails) |

The legal form is kept in `name_norm`/`name_legal` because removing it can create collisions
(`X Holdings Ltd` vs `X Holdings LLP`). The `name_legal_eq` / `name_legal_conflict` features let the model decide
how much that matters.

**Experiment A/B:** *conservative* versus *aggressive* (which also strips generic business-type words from
`name_core`) are compared with the same model on the same split. The mode with the better cross-validated F0.5
is written to `configs/selected.yaml`. See section 9.

## 4. Candidate generation (multi-pass blocking, union)

Nine independent blockers, each switchable in `configs/config.yaml`, run for S1→S2 and S1→S3. Their outputs are
**unioned**, never intersected. For every pair we keep which blockers found it (a bitmask) and each blocker's
retrieval score, and both become model features.

| blocker | mechanism |
|---|---|
| `exact_name` | equality of canonical, phonetic, sorted or trade-name keys (stop-keys above `max_group` skipped) |
| `acronym` | initials ↔ single-token name, both directions |
| `name_word_tfidf` | top-k word TF-IDF cosine on core names (ubiquitous terms pruned by `max_df`) |
| `name_char_tfidf` | top-k character 3-gram TF-IDF, handling typos and transliteration |
| `name_char_global` | the same without country partitioning, a safety net for noisy country labels |
| `address_tfidf` | top-k address word TF-IDF, recovering renamed or trade-name businesses at the same address |
| `name_address_char` | top-k character n-grams of name + address |
| `postal_token` | same postcode AND a shared name token |
| `semantic_lsa` | top-k dense retrieval on LSA embeddings (character n-grams of name+address) learned from the corpus |

Sparse top-k uses chunked sparse matrix products with a vectorised per-row top-k. Dense top-k uses chunked BLAS
products with `argpartition`. Nothing performs a full Cartesian comparison. Common terms are capped both
proportionally (`max_df`) and absolutely (`max_df_abs`), so the work per record stays constant as the corpus grows.

**Cascade stage 1: candidate pruning.** Blocking is tuned for recall, so most candidates are easy negatives. A
small gradient-boosted model on near-free features (the nine retrieval scores, four sparse/dense cosines,
country/postcode equality, cheap rank) scores every blocked pair. A pair is kept if its score ≥ τ, or if it is among
its S1's top 3 for that source. τ is chosen on **out-of-fold** scores (grouped by S1) as the largest value that still
keeps ≥ 99.9% of the true links blocking found. The fitted pruner ships inside the model bundle, so test data is
pruned exactly as training data was. `candidate_pairs.tsv` is the pruned set, which is precisely what the matcher
scores. Pruning is kept only if an A/B on validation shows it costs ≤ 0.001 F0.5 (section 9).

**Blocking experiment:** rule sets A (names only) → B (+address, postal) → C (all lexical) → D (all + semantic) →
E (D without country partition), plus a candidate-depth sweep (k × 0.5 / 1.5) and a leave-one-out of each blocker.
Selection rule: the highest candidate recall, then among sets within 0.1 pp of it, the smallest volume.
`candidate_pairs.tsv` is exactly the candidate set the matcher scores.

## 5. Pair features (~110, all numeric, defined in `src/features/`)

* **Name:** exact / canonical / core / phonetic / sorted equality; word TF-IDF, char 2–4-gram TF-IDF, phonetic and
  trade-name cosines; LSA cosine; token Jaccard, overlap and coverage per side; IDF-weighted (rare-token) Jaccard,
  overlap and shared/unshared mass; exact **vectorised Levenshtein** (distance and normalised) on core and
  token-sorted names; prefix and suffix similarity; length difference and ratio; legal-form agreement and conflict;
  acronym match; first-token equality; name frequency on each side (chain/franchise signal).
* **Address:** canonical/core equality; word, char and phonetic cosines; token and IDF-weighted overlaps; Levenshtein;
  numeric-token overlap, conflict and full agreement; street-number agreement and conflict; postcode equality,
  conflict, 3-digit prefix and missingness; landmark presence; LSA cosine of name+address.
* **Country and missingness:** equality, conflict, missing on either side; name/address missing on one or both sides.
* **Cross features:** best name and address evidence, product, min, max, weighted combination, and
  strong/weak indicator quadrants.
* **Retrieval provenance:** the nine blocker scores and the blocker count.
* **Group context (label-free competition features):** rank and gap-to-best of the pair among the S1's
  candidates *and* among the S2/S3 record's S1 candidates, margin over the runner-up, candidate counts, number of
  strong candidates, and a mutual-best flag.
* **Sibling consistency (multi-match support, validation-gated):** name, address and semantic similarity of a
  candidate to its S1's best *other* candidate, plus a near-duplicate flag. Error analysis showed that rank
  features alone punish the second true record of a one-to-many entity ("lost to competitor"). A near-duplicate
  of the leading candidate is likely the same business listed twice, not a rival. The group is switched on only
  if it improves validation F0.5 by more than 0.0005.

Levenshtein is computed exactly with a vectorised row recurrence. The in-row insertion dependency is solved with a
cumulative minimum, and pairs are bucketed by length. A unit test checks it against a reference implementation.

## 6. Models

All models are trained on the labelled candidate pairs (label = pair is in the ground truth). Every model family is
evaluated on **identical candidates, features, folds and decision tuning**:

* Logistic Regression (scaled), Random Forest, Extra Trees, HistGradientBoosting (with early stopping). XGBoost and
  LightGBM are added automatically if installed. All are MIT, BSD or Apache licensed, small, and CPU-only.
* **Fellegi–Sunter:** seven comparison fields (name, rare-name tokens, address, street number, postcode, country,
  candidate rank) are discretised into agreement levels. `m` and `u` are estimated from the training labels with
  Laplace smoothing, weights are `log2(m/u)`, and scores are the sum plus prior log-odds. It is fully interpretable,
  and its weight table is shown in the UI.
* **Semantic cascade:** stage 1 is a gradient-boosted model *without* the semantic features. Only pairs in its
  uncertain band (0.05–0.95) are re-scored by stage 2, which is trained only on such pairs with all features,
  including the semantic LSA embeddings and the stage-1 score. The "semantic model" is learned from the competition
  corpus, so no external pretrained model or data is involved.
* **Stacking:** inner out-of-fold scores from HistGB, LR and FS are logit-transformed and fed, with two raw evidence
  features, to a logistic meta-classifier.
* **Hard-negative mining:** inner out-of-fold scores find non-matches the model scores ≥ 0.3 (same brand, same
  building, common names). Their weight is multiplied (×3), and the model is refit. The variant is kept only if
  validation improves.
* **Ablations:** the gradient-boosted model is also trained without the semantic features, without the context
  features and without the sibling features, to measure what each group contributes.
* **Hyper-parameter search** (`python -m src.cli tune`): a randomised search over gradient-boosting, logistic and
  forest parameters. Each trial is scored by DEV out-of-fold **macro F0.5 after decision tuning**, never by
  training loss. The incumbent configuration is always trial 0, so the search can only keep or improve it.
  Early stopping bounds the number of boosting iterations.

## 7. Decision engine (`src/matching/decision_engine.py`)

The engine takes calibrated scores (isotonic, fitted on out-of-fold predictions) and returns 0…n links per S1, a
confidence and reason codes. The parameters are tuned by grid search **for macro F0.5 on out-of-fold scores**:

* pair threshold `t` (never assumed to be 0.5);
* singleton gate: the entity's best score must reach `t + δ`, otherwise NO MATCH;
* score-gap gate: if the best accepted link leads the best rejected competitor by less than `min_gap`, the entity
  is ambiguous and abstains;
* `max_per_source` (unlimited / 1 / 2) and **exclusive assignment**, a graph constraint where each S2/S3 record
  links to at most one S1 (its best). Both are enabled only if validation shows a gain;
* an alternative **expected-F0.5 mode**: per entity, choose the top-k set maximising
  `E[F] ≈ 1.25·Σ_topk p / (k + 0.25·Σ p)`, and predict nothing when `Π(1−p)` is larger.

Reason codes (`NAME_EXACT`, `NAME_HIGH_SIMILARITY`, `TRADE_NAME_MATCH`, `ADDRESS_HIGH_SIMILARITY`, `POSTAL_MATCH`,
`COUNTRY_MATCH`, `SEMANTIC_SUPPORT`, `MULTI_FIELD_AGREEMENT`, `STRONG_SCORE_GAP`, `MUTUAL_BEST`, conflict codes,
`LOW_CONFIDENCE_SINGLETON`, `AMBIGUOUS_CANDIDATES`, `NO_CANDIDATES`) are derived from the features of each
accepted link.

## 8. Validation protocol (leakage control)

* S1 entities are split into **DEV (80%)** and **HOLDOUT (20%)**, stratified by match multiplicity × country. The
  S2/S3 pools stay complete, as at test time.
* On DEV: grouped K-fold by S1 entity (no entity in both train and validation folds) gives out-of-fold scores, from
  which the calibrator and decision parameters are fitted.
* HOLDOUT is scored **once** per experiment: the model is fitted on all of DEV and uses the decision parameters from
  DEV.
* **Selection** uses DEV cross-validated F0.5 (primary), then precision, then candidate recall, then runtime.
  HOLDOUT is the unbiased estimate we report.
* **Hard-case subset:** HOLDOUT entities with chain or common names, true links with weak name evidence, or
  singletons that have a near-identical-name decoy. They are reported separately.
* **Leakage report** (saved per split): exact record duplicates across splits, shared names, and records linked
  from both splits. Same-name entities across splits are legitimate franchises. The model sees only similarity
  features, never ids or raw strings, so it cannot memorise identities.
* **What is fitted on labels:** model parameters, calibration, thresholds, Fellegi-Sunter m/u, hard-negative
  weights and stacking weights. All of these are fitted inside training folds only. TF-IDF/IDF/LSA statistics are
  unsupervised descriptions of the corpus being resolved: they are computed the same way on train and on test and
  never see a label.
* The final model is refit on 100% of training entities, with calibration and decisions from full-train
  out-of-fold scores.

## 9. Results

<!-- AUTO:RESULTS:START -->
_Run `python scripts/make_report.py` after `python -m src.cli all` to fill this section from `experiments/`._
<!-- AUTO:RESULTS:END -->

## 10. Error analysis

Every false positive and false negative on HOLDOUT is categorised automatically
(`src/evaluation/error_analysis.py`):

* **False positives:** singleton false merge, common business name, name collision, address collision,
  over-normalisation, semantic false positive, ambiguous candidate, threshold issue, weak evidence.
* **False negatives:** blocking failure, lost to a competitor, threshold too strict, trade name, abbreviation,
  transliteration, typo, missing field, address variation, semantic failure.

The reports (`experiments/exp_NNN/errors.csv`, `error_summary.json`) and the Error Analysis page drove the
iterations: trade-name extraction, landmark isolation, the postcode-position rule, context features and the
ambiguity gate.

## 11. Efficiency

* **Analyse once:** each (field, analyser, n-gram) is tokenised once with stateless feature hashing (2^20 buckets),
  in parallel chunks across CPU cores. Every TF-IDF, binary and `max_df`-pruned variant used by features and
  blockers is derived from those cached counts with sparse algebra, replacing 13 separate vectoriser fits.
* **LSA** bases are fitted on a ≤60K-row sample and applied to all records, so the cost does not grow with the corpus.
* **Blocking union** uses integer pair keys (`np.unique`, then scatter-max of the scores) instead of a pandas pivot.
* **Features** are computed in concurrent batches (threads: numpy/scipy kernels release the GIL) and are
  bit-identical to the serial result. Levenshtein is exact and vectorised, with per-record encodings cached.
* **Cascade pruning** removes most candidates before the expensive feature engine, and every model fit trains
  on fewer rows.
* The full preparation (store, candidates, features) is cached per dataset fingerprint, config hash and pruner.
  Per-stage wall time and memory are logged for every run. `scripts/benchmark.py` measures the production
  inference path at 10K/50K/100K records (section 9.6).

## 12. Fair play and licences

Only the provided competition data is used. There are no external databases, APIs, geocoders, web lookups,
pretrained language models or data augmentation. Every dependency is BSD, MIT or Apache-2.0 licensed. The "semantic"
component is an LSA model trained on the competition records themselves, far below the 8B-parameter limit. The small
lists of abbreviations, legal forms and transliteration rules in the normaliser are generic linguistic conventions,
not business identity information.

## 13. Explainability and product

A FastAPI backend and a web application expose each decision: a side-by-side token diff, evidence bars, independent
model votes (calibrated primary, Fellegi-Sunter, logistic regression), per-field Fellegi-Sunter weights, reason
codes, the blocking passes that retrieved each candidate, and a plain-language summary. Runs execute in the
background with live progress. Models are versioned (`entity-model-vN`) and never overwritten, and the competition
outputs change only when a version is explicitly promoted.
