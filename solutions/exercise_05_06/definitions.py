"""
definitions.py
---------------
Pulls glossary-style definitions out of CNXML files and consolidates them
across an entire textbook (many chapter/module files) into one CSV.

OpenStax CNXML represents a glossary entry like this:

    <definition id="def1">
      <term>present value</term>
      <meaning id="m1">The current worth of a future sum of money...</meaning>
    </definition>

usually grouped inside a <glossary> element at the end of a module, but this
extractor doesn't assume that -- it just looks for every <definition> tag
anywhere in the document and pulls its first <term> and first <meaning>
child, however they're nested.

Two-step workflow:

    1. extract_definitions_dir(dir)     -> flat list of Definition, one row
                                            per <definition> tag found, tagged
                                            with which file/chapter it came from.
    2. consolidate_definitions(defs)    -> merges duplicate terms (same term
                                            defined in multiple chapters, or
                                            the same term capitalized differently)
                                            into one row, keeping every distinct
                                            meaning and listing every source file.

    write_definitions_csv(...)          -> does both and writes the CSV.

Run directly:
    python definitions.py --input /path/to/modules_dir --out definitions.csv
    python definitions.py /path/to/modules_dir --out definitions_raw.csv --no-consolidate
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from raw_reader import _local_tag  # reuse the same namespace-stripping helper


@dataclass
class Definition:
    term: str
    meaning: str
    source_file: str
    source_title: str  # the module/chapter's <title>, if found


def _first_by_local_tag(el: etree._Element, local_name: str) -> etree._Element | None:
    """Find the first descendant (or self) of `el` whose tag, ignoring
    namespace, equals `local_name`."""
    for node in el.iter():
        if _local_tag(node.tag) == local_name:
            return node
    return None


def _clean_text(el: etree._Element) -> str:
    text = "".join(el.itertext())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_definitions(path: str | Path) -> list[Definition]:
    """Extract every <definition> element's term + meaning from one file."""
    path = Path(path)
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    try:
        tree = etree.parse(str(path), parser)
    except Exception:
        return []
    root = tree.getroot()

    title_el = _first_by_local_tag(root, "title")
    source_title = _clean_text(title_el) if title_el is not None else path.stem

    definitions: list[Definition] = []
    for def_el in root.iter():
        if _local_tag(def_el.tag) != "definition":
            continue

        term_el = None
        meaning_el = None
        for child in def_el.iter():
            if child is def_el:
                continue
            tag = _local_tag(child.tag)
            if tag == "term" and term_el is None:
                term_el = child
            elif tag == "meaning" and meaning_el is None:
                meaning_el = child

        term = _clean_text(term_el) if term_el is not None else ""
        # Fall back to "everything in <definition> except the term" if there's
        # no explicit <meaning> tag (some CNXML variants just use <para>).
        if meaning_el is not None:
            meaning = _clean_text(meaning_el)
        else:
            full = _clean_text(def_el)
            meaning = full[len(term):].strip(" :—-") if term and full.startswith(term) else full

        if not term:
            continue  # nothing usable

        definitions.append(Definition(
            term=term, meaning=meaning,
            source_file=str(path), source_title=source_title,
        ))

    return definitions


def extract_definitions_dir(directory: str | Path,
                             patterns: tuple[str, ...] = ("*.cnxml", "*.xml")) -> list[Definition]:
    """Extract definitions from every matching file in a directory tree,
    in sorted file order. One row per <definition> tag found -- duplicates
    across chapters are NOT merged here; use consolidate_definitions for that."""
    directory = Path(directory)
    files: set[Path] = set()
    for pattern in patterns:
        files.update(directory.rglob(pattern))

    all_defs: list[Definition] = []
    for f in sorted(files):
        all_defs.extend(extract_definitions(f))
    return all_defs


def _normalize_term(term: str) -> str:
    t = term.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def consolidate_definitions(definitions: list[Definition]) -> list[dict]:
    """Merge definitions that share the same term (case/punctuation-insensitive)
    into one row. Keeps every distinct meaning text (in case a term is
    legitimately redefined or refined in a later chapter) and every source
    file/chapter it appeared in.

    Returns a list of plain dicts (CSV-ready), sorted alphabetically by term,
    with columns: term, meaning, occurrence_count, chapters, source_files.
    """
    groups: dict[str, dict] = {}
    for d in definitions:
        key = _normalize_term(d.term)
        if key not in groups:
            groups[key] = {
                "term": d.term,  # first-seen casing/spelling is the display form
                "meanings": [],
                "chapters": [],
                "source_files": [],
            }
        g = groups[key]
        if d.meaning and d.meaning not in g["meanings"]:
            g["meanings"].append(d.meaning)
        if d.source_title not in g["chapters"]:
            g["chapters"].append(d.source_title)
        if d.source_file not in g["source_files"]:
            g["source_files"].append(d.source_file)

    rows = []
    for g in groups.values():
        rows.append({
            "term": g["term"],
            "meaning": " | ".join(g["meanings"]),
            "occurrence_count": len(g["source_files"]),
            "chapters": "; ".join(g["chapters"]),
            "source_files": "; ".join(g["source_files"]),
        })
    rows.sort(key=lambda r: r["term"].lower())
    return rows


def write_definitions_csv(directory: str | Path, out_path: str | Path,
                           consolidate: bool = True) -> int:
    """End-to-end: read every CNXML file in `directory`, extract definitions,
    optionally consolidate duplicate terms, and write a CSV to `out_path`.
    Returns the number of rows written."""
    definitions = extract_definitions_dir(directory)

    if consolidate:
        rows = consolidate_definitions(definitions)
        fieldnames = ["term", "meaning", "occurrence_count", "chapters", "source_files"]
    else:
        rows = [{"term": d.term, "meaning": d.meaning,
                  "source_title": d.source_title, "source_file": d.source_file}
                 for d in definitions]
        fieldnames = ["term", "meaning", "source_title", "source_file"]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Consolidate CNXML glossary definitions into a CSV.")
    ap.add_argument("--input", default="osbooks-principles-finance", help="Directory of .cnxml/.xml files (searched recursively).")
    ap.add_argument("--out", default="definitions.csv", help="Output CSV path.")
    ap.add_argument("--no-consolidate", action="store_true",
                     help="Write one row per <definition> tag found, without merging duplicate terms.")
    args = ap.parse_args()

    n = write_definitions_csv(args.input, args.out, consolidate=not args.no_consolidate)
    print(f"Wrote {n} rows to {args.out}")