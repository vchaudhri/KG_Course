"""
io_utils.py
-----------
Reading input records (CSV, JSON array, or JSONL -- auto-detected), and
JSONL append/read helpers used for every intermediate output file.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from .models import InputRecord

T = TypeVar("T")

REQUIRED_FIELDS = ("term", "genus", "differentia")


def _read_csv_rows(path: Path) -> list[dict]:
    """utf-8-sig so a CSV saved by Excel (which adds a BOM) or by this
    project's other tools (which all write utf-8-sig -- see repair_csv_encoding.py
    and friends) reads cleanly either way. Header names are matched
    case-insensitively and whitespace-trimmed, so 'Term', ' Genus ', etc.
    all work; unrelated extra columns (e.g. is_aristotelian, occurrence_count,
    chapters, source_files from aristotelian_classifier.py's own CSV output)
    are simply ignored rather than rejected -- this is what lets you point
    this reader directly at definitions_classified.csv."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        header_map = {name: name.strip().lower() for name in reader.fieldnames if name is not None}
        rows = []
        for row in reader:
            rows.append({header_map[k]: (v or "").strip() for k, v in row.items() if k in header_map})
        return rows


def _read_jsonl_rows(text: str, path: Path, strict: bool = False) -> tuple[list[dict], list[dict]]:
    """Returns (rows, skipped). In strict mode, the first unparseable line
    raises immediately (old behavior); otherwise it's recorded in `skipped`
    and reading continues with the rest of the file."""
    rows: list[dict] = []
    skipped: list[dict] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            error_msg = f"{path}:{line_no}: invalid JSON ({e})"
            if strict:
                raise ValueError(error_msg) from e
            skipped.append({"row_index": line_no, "raw": line, "error": error_msg, "term": ""})
    return rows, skipped


def read_input_records(path: str | Path, include_suggested: bool = False,
                        strict: bool = False) -> tuple[list[InputRecord], list[dict]]:
    """Reads a CSV file, a JSON array of objects, or a JSONL file (one
    object per line). Format is chosen by file extension (.csv / .json /
    .jsonl / .ndjson); anything else falls back to content-sniffing the
    same way earlier versions of this function always worked (JSON array
    if the file starts with '[', JSONL otherwise). Each record must have
    non-empty 'term', 'genus', 'differentia' values.

    include_suggested: if a row's 'genus' and/or 'differentia' is blank but
    'suggested_genus'/'suggested_differentia' are present and non-blank
    (as in aristotelian_classifier.py's definitions_classified.csv, for
    rows where is_aristotelian == "no"), fall back to those instead of
    raising -- the resulting InputRecord.source is set to "suggested"
    rather than "extracted" so this is traceable all the way through to
    the final output. Off by default: suggested_genus/suggested_differentia
    are an LLM's best-effort reconstruction for a definition that already
    failed the genus-differentia test, not real textbook content -- some
    of those rows (e.g. an equation mislabeled as a definition) may not be
    genus-differentia shaped at all, so silently treating them the same as
    genuinely extracted data is opt-in, not the default.

    strict: if True, restores the pre-skip-and-continue behavior -- the
    first row with missing required fields (even after the
    include_suggested fallback), or the first unparseable JSONL line,
    raises ValueError and aborts the whole read. If False (the default),
    such rows are SKIPPED instead: reading continues, and each skipped row
    is returned via the second element of the result tuple rather than
    lost, so callers (Pipeline) can record them to 04_failures.jsonl. Note
    this does NOT apply to a JSON array file that fails to parse at all --
    if the top-level JSON itself is malformed, there's no list of
    individual records to recover from, so that always raises regardless
    of `strict`.

    Returns (records, skipped_rows), where each skipped_rows entry is
    {"row_index": int, "raw": dict|str, "error": str, "term": str}."""
    path = Path(path)
    suffix = path.suffix.lower()
    skipped: list[dict] = []

    if suffix == ".csv":
        raw_records = _read_csv_rows(path)
    else:
        text = path.read_text(encoding="utf-8")
        if suffix in (".jsonl", ".ndjson"):
            raw_records, jsonl_skipped = _read_jsonl_rows(text, path, strict=strict)
            skipped.extend(jsonl_skipped)
        elif suffix == ".json" or text.lstrip().startswith("["):
            raw_records = json.loads(text)  # whole-file parse failure: no partial recovery possible, always raises
        else:
            raw_records, jsonl_skipped = _read_jsonl_rows(text, path, strict=strict)
            skipped.extend(jsonl_skipped)

    records = []
    for i, raw in enumerate(raw_records):
        term = str(raw.get("term", "")).strip()
        genus = str(raw.get("genus", "")).strip()
        differentia = str(raw.get("differentia", "")).strip()
        source = "extracted"

        if include_suggested and (not genus or not differentia):
            suggested_genus = str(raw.get("suggested_genus", "")).strip()
            suggested_differentia = str(raw.get("suggested_differentia", "")).strip()
            if not genus and suggested_genus:
                genus = suggested_genus
                source = "suggested"
            if not differentia and suggested_differentia:
                differentia = suggested_differentia
                source = "suggested"

        missing = [name for name, val in (("term", term), ("genus", genus), ("differentia", differentia)) if not val]
        if missing:
            hint = " (suggested_genus/suggested_differentia fallback was attempted but didn't cover this gap)" \
                if include_suggested else ""
            error_msg = f"record {i} is missing required field(s) {missing}{hint}"
            if strict:
                raise ValueError(f"{path}: {error_msg}: {raw}")
            skipped.append({"row_index": i, "raw": raw, "error": error_msg, "term": term})
            continue

        records.append(InputRecord(term=term, genus=genus, differentia=differentia, source=source))
    return records, skipped


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
