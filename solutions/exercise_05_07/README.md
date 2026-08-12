# Adobe concept -> Principles-of-Finance taxonomy mapper

Connects the 28 Adobe headwind/tailwind concepts (from the earnings-call
extraction project) to your existing genus/differentia taxonomy extracted
from the OpenStax "Principles of Finance" textbook, via Gemini.

## What it does

1. Loads `genus_statements.owl` back into memory.
2. Applies two corrections found by manually inspecting the taxonomy:
   - The `value` class had two backwards `subclass_of` edges (it was
     asserted as a *subclass* of `present_discounted_value` and
     `present_value_of_the_cash_inflows_of_a_project`, which is reversed --
     both are kinds of value, not the other way around). Fixed by reversing
     those two edges.
   - Three pairs of near-duplicate glossary terms (`economic risk` /
     `economic exposure`, `transaction risk` / `transaction exposure`,
     `translation risk` / `translation exposure` -- same definition, two
     different textbook chapters) are merged into one canonical term each
     before the glossary is connected into the taxonomy.
3. Runs `add_glossary_terms()` -- imported directly from **your**
   `genus_class_statements.py` -- to connect the full glossary into the
   taxonomy as real named classes. Zero LLM calls, exactly as that function
   already promises.
4. For each of the 28 Adobe concepts, calls Gemini once to pick a genus (an
   existing taxonomy class from a curated shortlist, or a brand-new class
   only when nothing fits) and to write a differentia whose *predicate*
   comes from the fixed vocabulary in `wikidata_relation_alignment.csv`
   (`arises_from`, `caused_by`, `impacts`, `contributes_to`, etc.) -- only
   the differentia's specific content is freely written, never the
   predicate.
   - `C06` (the freemium/MAU trade-off) is hardcoded rather than sent to
     Gemini: earlier analysis confirmed it has no home in either the risk
     or income branch, so a new `strategic_tradeoff` genus is recorded
     directly in the script rather than left to the model to invent
     consistently run to run.
5. Writes:
   - `principles_of_finance_taxonomy_extended.owl` -- your corrected,
     glossary-merged base taxonomy (open in Protege).
   - `adobe_taxonomy_extension.owl` -- just the new Adobe-concept classes
     (plus `strategic_tradeoff`), in the same namespace, ready to load
     alongside the file above.
   - `adobe_concept_taxonomy_mapping.csv` -- one row per concept: genus
     class, differentia predicate + text, confidence, rationale.
   - `review_log.md` -- every automated fix applied, any cycle-safety check
     result, and every concept flagged low-confidence (including any case
     where Gemini's answer referenced a class or predicate that doesn't
     actually exist -- those are caught and flagged rather than silently
     written into the output).

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your-key-here   # or GOOGLE_API_KEY
```

On Windows PowerShell: use `$env:GEMINI_API_KEY = "your-key-here"` instead of
`export`, and use a backtick `` ` `` (or just one line) for command
continuation instead of `\`.

This script imports `genus_class_statements.py` directly from your other
project (and, transitively, `genus_similarity.py`, which
`genus_class_statements.py` itself imports for `normalize_genus()`). It does
**not** duplicate that code here -- point `--genus-project-dir` at the
directory containing both files.

## Inputs

Place your own copies of these three files under `inputs/` (or point the
CLI flags anywhere you like):

- `genus_statements.owl`
- `definitions_classified.csv`
- `wikidata_relation_alignment.csv`

`inputs/concepts.csv` and `inputs/mentions.csv` are already included -- they're
the 28-concept / 161-mention Adobe dataset built earlier in this project.

## Run

```bash
python build_taxonomy_extension.py \
    --genus-project-dir /path/to/your/other/project \
    --owl-in inputs/genus_statements.owl \
    --definitions-csv inputs/definitions_classified.csv \
    --wikidata-csv inputs/wikidata_relation_alignment.csv \
    --concepts-csv inputs/concepts.csv \
    --mentions-csv inputs/mentions.csv \
    --out-dir out
```

Add `--dry-run` first to sanity-check the taxonomy parsing / fixes / dedup /
glossary-merge stages without spending any Gemini calls (only the hardcoded
`C06` result is resolved; every other concept gets a placeholder).

## If you see "the specified schema produces a constraint that has too many states"

This is a Gemini structured-output limit, not something in your data that's
wrong. It showed up because `wikidata_relation_alignment.csv` turns out to
have ~446 distinct `canonical_relationship` values (far more than the
handful assumed during design), and an earlier version of this script tried
to constrain `differentia_predicate` to a JSON Schema `enum` of all of them
-- Gemini's constrained-decoding compiler rejects enums that large. Fixed:
that field (and `genus_class`/`new_genus_parent`) are plain strings again,
enforced by the existing post-hoc validation/auto-repair instead of the
schema itself. Also, by default only the ~37 predicates flagged
`is_causal=True` are sent to the model at all (the rest are structural/
factual -- `is_a`, `employer`, `located_in`, etc. -- not useful for a
differentia); pass `--include-all-predicates` if you want the full list.

## `--owl-in` with the glossary already merged in

Some upstream re-exports of `genus_statements.owl` are run with
`--definitions` already applied (i.e. `add_glossary_terms()` has already
connected the full glossary as real term classes, not just genus-phrase
classes). This script auto-detects that (`--glossary-mode auto`, the
default) by checking for a handful of glossary term class names that can
only exist post-merge, and adapts automatically:

- If already merged: skips calling `add_glossary_terms()` again (which
  would otherwise create collision classes like `economic_risk_2`), and
  merges the three near-duplicate pairs directly in the graph instead of
  via the CSV-row dedup.
- If not yet merged: works exactly as before -- dedupe the CSV rows, then
  call `add_glossary_terms()` itself.

Override with `--glossary-mode merged` or `--glossary-mode unmerged` if the
heuristic ever guesses wrong; `review_log.md` always states which mode a
run used.

Also note: some taxonomy re-exports have been observed to drop or rename
classes between runs (e.g. a `metric` class that used to be a common
ancestor no longer exists in one re-export). `CANDIDATE_ROOT_CLASSES` and
the `C06` override's parent (`thing`, deliberately the most conservative
choice) are both written to tolerate this -- a missing root is logged and
skipped rather than crashing.

## If it stops with "cycle(s) exist in the base taxonomy itself"

This means `genus_statements.owl` (as currently exported from your other
project) has a `subclass_of` cycle even after this script's own value-class
fix and glossary merge. This script deliberately does **not** try to break
it here and stops before spending any Gemini calls -- fixing a cycle
correctly means picking the least-evidenced edge to remove, which requires
the original per-pair subsumption vote counts from
`genus_pairwise_similarity.csv`. That data lives in your other project, not
in the flattened OWL this script reads, so any cycle-break attempted here
would be a guess.

The fix is upstream: re-run your `genus_class_statements.py` pipeline
(`process_pairwise_csv` / its CLI) against your **current**
`genus_pairwise_similarity.csv` with cycle-breaking on (the default --
only `--no-break-cycles` turns it off), re-export `genus_statements.owl`,
and point `--owl-in` at that fresh file. Full cycle details are written to
`out/cycles_detected.txt` on every run that finds one, so you can see
exactly which classes are involved. `--allow-cycles` overrides this and
proceeds anyway, but candidate genus lists downstream may then include
spurious classes pulled in only through the cycle -- treat results from
such a run as provisional.

## Notes / things worth reviewing after a run

- Every genus class and differentia predicate Gemini can return is
  constrained to real values via JSON Schema `enum`s (the actual candidate
  class list and the actual `wikidata_relation_alignment.csv` predicate
  list), not just prompt instructions -- so a hallucinated class or
  predicate name should be structurally rare. As a second layer, anything
  that still doesn't validate (e.g. the hardcoded `C06` override's parent
  class, if your taxonomy's vocabulary has changed since this script was
  written) is auto-repaired to fall back under `thing` rather than written
  into the output OWL as a dangling reference, and flagged in
  `review_log.md` with `confidence=low` either way.
- `add_glossary_terms()` defaults to `include_suggested=False` (matching
  your own script's default) -- a handful of glossary terms with a blank
  `genus` field will land under `thing` rather than a more specific class
  unless you pass `--include-suggested`. The `economic risk` /
  `economic exposure` pair was the one case of this actually relevant to
  the risk/income branches, and it's already fixed by the dedup step
  (which copies over the alias's well-formed genus).
- `review_log.md` will call out any Gemini answer that referenced a genus
  class or differentia predicate that doesn't actually exist in the
  taxonomy/vocabulary -- these are automatically downgraded to
  `confidence=low` and flagged rather than written as a dangling OWL
  reference, but are worth a manual look.
- The candidate genus shortlist sent to Gemini is the full descendant set
  of `risk`, `income`, `total_return`, `equity`, `metric`, and `asset`
  (see `CANDIDATE_ROOT_CLASSES` in `build_taxonomy_extension.py`) -- if you
  find concepts consistently miss a branch that should have been a
  candidate, add its root class name to that list.
