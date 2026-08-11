# differentia_pipeline

Extracts and normalizes relationship-value characteristics from
genus-differentia definitions:

```
definition -> term + genus + differentia -> raw relationship/value extraction
    -> relationship normalization -> value normalization -> structured output
```

## Setup

```bash
pip install google-genai --break-system-packages
export GEMINI_API_KEY=...
```

No other dependencies -- everything else is Python's standard library.
Note this package has its **own** `LLMClient`/`GeminiLLMClient`, separate
from the ones used elsewhere in the parent `kg_from_openstax` project, per
the spec's request to keep LLM-specific code isolated behind that
abstraction so the provider can be swapped later without touching anything
else.

## Input format

CSV, a JSON array, or JSONL -- auto-detected by file extension (`.csv`,
`.json`, `.jsonl`/`.ndjson`; anything else falls back to content-sniffing).
Each record needs at least `term`, `genus`, `differentia`.

JSON/JSONL:
```json
{"term": "commercial bank", "genus": "financial institution",
 "differentia": "accepts deposits from customers and provides loans to individuals and businesses"}
```

CSV: a `term,genus,differentia` header plus one row per record. Header
names are matched case-insensitively and whitespace-trimmed, and any
extra columns are simply ignored -- which means you can point this
directly at `definitions_classified.csv` from `aristotelian_classifier.py`
elsewhere in this project, since it already has `term`/`genus`/`differentia`
columns among others.

**Rows missing `genus`/`differentia`** (e.g. `definitions_classified.csv`'s
`is_aristotelian == "no"` rows, or occasional inconsistent rows where a "yes"
row's `differentia` came out blank anyway) are **skipped, not fatal, by
default** -- reading continues, and each skipped row is recorded in
`04_failures.jsonl` with `stage: "read_input"` and its actual error, so one
bad row out of hundreds never blocks the rest. Pass `--strict-input` to
restore the old abort-on-first-bad-row behavior instead. (One exception:
if a JSON *array* file's top-level structure itself is malformed, that
always raises regardless of `--strict-input` -- there's no list of
individual records to recover from in that case.)

Pass `--include-suggested` to fall back to `suggested_genus`/
`suggested_differentia` instead of skipping, when they're present and
non-blank -- each such record's `source` field is set to `"suggested"`
(vs. `"extracted"` for real genus/differentia rows), and that tag survives
all the way through to `01_raw_characteristics.jsonl` and
`03_normalized_characteristics.jsonl`, so you can filter or weight by it
later. This is off by default deliberately: `suggested_genus`/
`suggested_differentia` are an LLM's best-effort reconstruction for a
definition that already failed the genus-differentia test (e.g. an
equation mislabeled as a definition isn't genus-differentia shaped at
all), not real textbook content -- treating it identically to genuinely
extracted data by default risked quietly mixing the two with no way to
tell them apart later. The two flags compose: with `--include-suggested`,
a row is only skipped if genus/differentia are blank *and* there's no
usable suggested fallback either.

## Run

```bash
python -m differentia_pipeline.cli input.jsonl --out differentia_output
```

Useful flags: `--include-suggested`, `--strict-input`, `--extraction-model`,
`--normalization-model`, `--max-workers` (concurrent Stage 2 calls),
`--relationship-batch-size` (unique raw relationships per Stage 4 call),
`--cache-path`, `--log-level`.

Re-running with the same `--out` directory resumes automatically --
`01_raw_characteristics.jsonl` doubles as the checkpoint, and identical
`(term, genus, differentia)` content is also cache-backed, so nothing gets
re-extracted or re-normalized unnecessarily.

## Output (in `--out`)

| File | Contents |
|---|---|
| `01_raw_characteristics.jsonl` | Stage 2/3: one row per record, literal (uncanonicalized) relationship-value pairs, each tagged with `source` (`"extracted"` or `"suggested"`) |
| `02_relationship_mapping.json` | Stage 4: `{raw: canonical}` mapping, plus the reverse `canonical -> [raw, ...]` view, plus raw/canonical counts |
| `03_normalized_characteristics.jsonl` | Stage 5/6: final output -- one row per record with `raw_relationship`, `canonical_relationship`, `raw_value`, `normalized_value` per characteristic, plus `source` |
| `04_failures.jsonl` | Any record that failed at any stage -- including `read_input` for skipped input rows -- with the stage, error, and (where available) the raw content/LLM response. Never silently dropped |
| `llm_cache.jsonl` | Content-addressed LLM response cache (append-only, survives interruption) |

## Module map

| File | Role |
|---|---|
| `config.py` | `PipelineConfig` -- single source of configuration |
| `models.py` | Dataclasses for every record shape, plus the content-hash `record_id` that drives caching and checkpointing |
| `cache.py` | `CacheStore` -- thread-safe, append-only JSONL cache |
| `llm_client.py` | `LLMClient` (abstract) + `GeminiLLMClient` -- the only file that imports `google-genai` |
| `differentia_analyzer.py` | Stage 2/3 -- per-record extraction |
| `relationship_normalizer.py` | Stage 4 -- corpus-wide, sequential-batch normalization with running canonical-vocabulary context |
| `value_normalizer.py` | Stage 5 -- conservative lexical value normalization |
| `pipeline.py` | Orchestrates all stages; concurrency, checkpointing, logging, failure recording |
| `cli.py` | Argument parsing + entry point |

## Design notes worth knowing

- **Why `relationship_normalizer.py`'s batches run sequentially, not
  concurrently** (unlike Stage 2's extraction, which does run concurrently):
  each batch's prompt includes the canonical vocabulary already established
  by prior batches, so a later batch reuses e.g. `used_for` instead of
  inventing `serves_purpose_of` as a near-duplicate. Running batches in
  parallel would mean no batch ever sees another's results, defeating the
  entire point of this stage -- measuring how much the vocabulary
  compresses. I validated this specifically: built a test where batch 2's
  prompt must contain batch 1's established `used_for`/`has_part`, and
  confirmed batch 2 correctly reused them for differently-worded synonyms
  rather than creating duplicates.
- **Checkpointing has no separate state file.**
  `01_raw_characteristics.jsonl` *is* the checkpoint -- on startup, whichever
  record ids already have a row there are skipped. Since `record_id` is a
  hash of `(term, genus, differentia)`, changed input automatically gets a
  new id rather than resuming with stale content, and there's no state file
  to accidentally get out of sync with the data. I tested this directly: ran
  the pipeline, added one new record, reran into the same output directory
  with an extraction client that raises if called on anything except the
  new record -- confirmed exactly one new call was made and all three
  previously-extracted records were correctly skipped, while Stage 4-6
  still recomputed cleanly over the full accumulated set (not just the new
  record).
- **Failures are never silent.** Any record that fails at any stage lands in
  `04_failures.jsonl` with the stage, the term, and the error -- the rest of
  the run continues rather than aborting. Tested with a deliberately-failing
  record alongside a good one: the run completed, the good record processed
  normally, and the bad one was recorded with its actual error message
  rather than vanishing.
- **`LLMClient` is abstract on purpose**, and it made testing this package
  much cleaner: every test above uses a small in-memory fake implementation
  instead of mocking `google.genai` internals, which is exactly the
  swappable-provider design the spec asked for paying off directly in
  testability.
