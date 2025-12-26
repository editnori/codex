# SmartSig SIG Concept Extractor (v11)

This document describes the **SIG concept extractor** architecture and the **v11** changes aimed at:

- Eliminating `UNKNOWN` residuals (no more “unknown spans” in output).
- Treating **residual/unmodeled text as first-class signal** (and making it visible in the canonical output).
- Expanding **learning** beyond DX into safe, clinically-relevant classes (route/device/time-of-day/timing-event/site).
- Improving **canonical safety**: never silently drop unmodeled content; avoid “confident hallucinations”.

**New in v11:**

- Adds a deterministic **structure/complexity layer** (`structure`) that segments multi-step SIGs (tapers, repeats, max constraints, day-of-week schedules) into clauses with flags.
- Adds **day-of-week** concept support (`day_of_week:*`) to better represent alternating schedules (e.g., warfarin).
- Adds `aux_insurance:*` noise concepts (e.g., `not covered`, `prior auth`) so these stop polluting residual free-text.
- Adds `aux_drug` as an **optional learning target** to tag drug/product names that appear inside SIG free-text (non-core).
- Improves canonical draft generation for complex SIGs via **multi-clause deterministic drafts**, reducing random capitalization/grammar drift.
- Reduces lexicon noise by skipping ultra-common ambiguous terms (e.g., dose_unit `each`).

---

## What this component is

The extractor converts a free-text medication SIG into:

1. **Spans** (byte/char offsets into the original SIG)
2. Each span has a **class-level category** (e.g., `route`, `freq`, `dose_unit`, `dx`, …)
3. A **classifier decision** inside that category (a preferred ontology concept), plus confidence
4. A **canonical instruction** string + extra notes, generated with LLM support but constrained to the extracted structure

This yields discrete, training-friendly annotations while retaining interpretability and facilitating incremental dictionary growth.

---

## v11 clinical goals

### 1) Residuals are not “unknown” anymore

**Prior behavior:** uncovered text became `category = residual_text` with `normalized.label = "UNKNOWN"`.

**v11 behavior:**

- Residual tokens are always labeled as one of:
  - `STOPWORD` (e.g., “the”, “and”, “of”, “to”, …)
  - `FREE_TEXT` (anything uncovered that is not a stopword)
- Therefore, there is no residual `UNKNOWN` classification in output.

This retains a clear separation:
- **Core spans** (used for structured parsing / downstream execution)
- **Non-core / residual spans** (kept for transparency and future model/lexicon expansion)

### 2) Residual segments are surfaced in the canonical output

A key safety issue is a canonical that looks “complete” while silently dropping leftover text.

**v11 enforces transparency:**

- The extractor computes *extra segments* from uncovered spans.
- If the canonicalizer did not incorporate them, v10 appends them into `canonical.extra_notes` as:

`Unparsed: <original residual segment>`

This makes canonicals less likely to be clinically misleading.

### 3) Learning now covers more safe classes by default

Learning patches are still conservative (confidence threshold), but the default target classes are expanded from DX-only into a safer core set:

- `dx`
- `route`
- `device`
- `time_of_day`
- `timing_event`
- `site`
- plus the existing non-core buckets:
  - `aux_warning`, `aux_auxiliary`, `aux_provider`, `aux_misc`

This is specifically to allow the system to learn clinically meaningful “residuals” like:

- `nebulized` → device/route concepts
- `at onset` → timing-event concepts
- `during the day` → time-of-day concepts
- common application sites like `skin`

---

## Data model

### Span object (output)

Each span in `spans[]` includes:

- `span_id`: stable identifier inside this SIG
- `category`: class-level label (your ontology “class”)
- `role`: `CORE`, `NON_CORE`, or `RESIDUAL`
- `span`: `{ text, start, end }` (offsets into original SIG)
- `normalized`: one of:
  - `CODED`: `{ concept_id, preferred }`
  - `NUMERIC`: `{ kind, value, ... }`
  - `DATE`: `{ iso }`
  - `ICD`: `{ code, system }`
  - `TEXT`: `{ label }`  ← used for residuals, but **never `UNKNOWN`** in v11
  - `REJECTED`: model rejected a dictionary candidate
- `matched_term`: the raw lexicon term matched (if applicable)
- `confidence`: model confidence in the normalization decision
- `source`: `LEXICON`, `AUTO`, or `LEARNED`
- `tags`: semantic tags derived deterministically (e.g., strength-vs-unit dose-unit classification)

### Canonical object (output)

- `canonical.text`: LLM-generated canonical, constrained by extracted structure
- `canonical.extra_notes[]`: non-core or unmodeled notes; now guaranteed to include “Unparsed:” notes when needed
- `canonical.confidence` and `canonical.source`

---

### Structure object (v11)

`structure` is a deterministic, heuristically-derived segmentation and complexity layer:

- `structure.kind`: `SIMPLE` or `COMPLEX`
- `structure.flags[]`: coarse quality/complexity flags (e.g., `HAS_TAPER_OR_TITRATION`, `HAS_DAY_OF_WEEK_SCHEDULE`, `HAS_MAX_CONSTRAINT`, `TRUNCATED_OR_INCOMPLETE`)
- `structure.steps[]`: clause/step segments with:
  - `step_id`, `kind` (`INSTRUCTION|REPEAT|CONSTRAINT|NOTE`)
  - `span.start/end/text`
  - `span_ids` plus convenience lists: `core_span_ids`, `noncore_span_ids`, `residual_span_ids`

This is intentionally conservative: it does not “invent” new clinical semantics; it just groups what was extracted.

---

## Pipeline overview

1. **Lexicon scan**
   - Matches candidate terms from:
     - `smartsig_sig_lexicon_ext*.script` (baseline)
     - `smartsig_sig_lexicon_overlay*.script` (learned aliases)
2. **LLM adjudication of candidates**
   - For each mention, model chooses:
     - accept candidate → CODED span
     - reject candidate → REJECTED span
3. **Auto spans**
   - Dates, ICD patterns, numeric patterns, punctuation tokens, etc.
4. **Residual spans**
   - Uncovered text emitted as `residual_text` spans labeled:
     - `STOPWORD` or `FREE_TEXT`
5. **Learning (optional but default-on)**
   - Candidate residuals (`FREE_TEXT`) are proposed for mapping to known ontology concepts
   - High-confidence patches are written to the overlay script (append-only)
6. **Semantic tagging**
   - Deterministic tags applied (e.g., dose_unit classified as `ROLE:DOSE_FORM` vs `ROLE:STRENGTH_UNIT`)
7. **Canonicalization**
   - Draft + spans + residual segments provided
   - Output canonical is checked and augmented:
     - leftover residual segments are appended as `Unparsed:` notes

---

## Lexicon updates included in v11

The v11 extended lexicon file adds coverage for (building on v10):

- Device variants:
  - `nebulized`, `nebulised`, `nebulize`, `nebulise` → Nebulizer device concept
- Timing event:
  - `at onset`, `onset` → `timing_event:ONSET`
- Time-of-day:
  - `during the day`, `daytime` → `time_of_day:DAYTIME`
- Site:
  - `skin`, `on the skin` → `site:SKIN`

New in v11:

- Day-of-week:
  - `monday/mon`, `tuesday/tue`, ... → `day_of_week:*`
  - `mwf`, `tth` → day-set concepts
- Insurance/noise:
  - `not covered`, `not cov`, `prior auth`, `prior authorization` → `aux_insurance:*`
- Sequence marker:
  - `titrate`, `titration` → `sequence:TITRATE`

These reduce residuals without adding high-entropy concept growth.

---

## Recommended runtime settings

### Conservative production mode (recommended)
- `--learn-unknowns true`
- `--learn-min-confidence 0.95` (or higher)
- Review overlay patches before promoting to baseline lexicon

### Bootstrapping mode (for exploration on large corpora)
- `--learn-min-confidence 0.85`
- Keep patches isolated per batch (separate overlay file) and review

---

## Output viewing

Use the included HTML viewer (`sig_viewer_v11.html`) to visually inspect:

- span coverage
- normalized concepts
- tags (strength vs unit)
- residual segments and “Unparsed:” notes
- canonical text quality

---

## Notes on safety and clinical correctness

The extractor is designed to reduce two classes of failure:

1. **Silent omission** (canonical looks clean but dropped content)
2. **Overconfident inference** (expanding abbreviations or interpreting junk)

v11 explicitly biases toward **transparency**: if content is not modeled, it is carried forward as an “Unparsed:” note rather than silently removed or guessed.

