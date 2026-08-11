"""
raw_reader.py
-------------
Step 0: just get the text out. No section hierarchy, no <term> extraction,
no chunking logic -- the bare minimum to confirm we can ingest a given
CNXML file (including ones that break the structured parser, e.g.
collection.xml chapter files or modules with unusual markup).

Entry points:

    read_raw_text(path)         -> single concatenated string for one file
    read_raw_text_dir(dir)      -> {file_path: text} for every .cnxml/.xml
                                    file found (recursive)

    summarize_file(path)        -> FileSummary: tag counts, text volume per
                                    tag, depth, namespaces -- a quick map of
                                    "what's actually in this file" before
                                    you write structural parsing logic for it.
    summarize_dir(dir)          -> {file_path: FileSummary} + an aggregate
                                    FileSummary across the whole directory.

Design: uses etree.parse(..., recover=True) so even malformed XML will
usually still parse (lxml's recovery mode skips/repairs bad bits instead
of raising), then just calls .itertext() on the root to pull every bit of
text content in document order, regardless of what tags it's inside.
This means it will also "work" on collection.xml files, just producing
mostly title/metadata text since those files don't contain prose.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree


def read_raw_text(path: str | Path) -> str:
    """Read one CNXML/XML file and return all its text content concatenated,
    in document order, with no structural interpretation."""
    path = Path(path)
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    tree = etree.parse(str(path), parser)
    root = tree.getroot()

    pieces = [t for t in root.itertext() if t and t.strip()]
    text = "\n".join(p.strip() for p in pieces)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def read_raw_text_dir(directory: str | Path, patterns: tuple[str, ...] = ("*.cnxml", "*.xml")) -> dict[str, str]:
    """Read every matching file in a directory tree (recursive). Returns a
    dict of {file_path_str: raw_text}. Files that fail to parse even in
    recovery mode are included with an error string as their value, rather
    than raising, so one bad file doesn't stop the whole batch."""
    directory = Path(directory)
    files: set[Path] = set()
    for pattern in patterns:
        files.update(directory.rglob(pattern))
    out: dict[str, str] = {}
    for f in sorted(files):
        try:
            out[str(f)] = read_raw_text(f)
        except Exception as e:
            out[str(f)] = f"[ERROR reading file: {e}]"
    return out


@dataclass
class FileSummary:
    path: str
    file_size_bytes: int = 0
    total_elements: int = 0
    max_depth: int = 0
    root_tag: str = ""
    namespaces: dict = field(default_factory=dict)
    tag_counts: Counter = field(default_factory=Counter)          # tag -> occurrences
    tag_own_text_chars: Counter = field(default_factory=Counter)  # tag -> chars of text directly inside it (not in children)
    tag_subtree_text_chars: Counter = field(default_factory=Counter)  # tag -> chars of all text under it, including children
    total_text_chars: int = 0  # same number read_raw_text would report
    parse_error: str | None = None

    def merge(self, other: "FileSummary") -> None:
        """Fold another summary's counts into this one (for directory-level aggregation)."""
        self.file_size_bytes += other.file_size_bytes
        self.total_elements += other.total_elements
        self.max_depth = max(self.max_depth, other.max_depth)
        self.tag_counts.update(other.tag_counts)
        self.tag_own_text_chars.update(other.tag_own_text_chars)
        self.tag_subtree_text_chars.update(other.tag_subtree_text_chars)
        self.total_text_chars += other.total_text_chars
        self.namespaces.update(other.namespaces)


def _local_tag(tag) -> str:
    """Strip namespace URI from an lxml tag, e.g. '{http://.../cnxml}para' -> 'para'.
    Returns a placeholder for comments/PIs, which aren't real elements."""
    if not isinstance(tag, str):
        return f"<{type(tag).__name__}>"
    return etree.QName(tag).localname if tag.startswith("{") else tag


def summarize_file(path: str | Path) -> FileSummary:
    """Walk one CNXML/XML file and report what's structurally in it:
    how many of each tag, how much text lives directly inside each tag
    vs. in its descendants, tree depth, and namespaces declared."""
    path = Path(path)
    summary = FileSummary(path=str(path))
    try:
        summary.file_size_bytes = path.stat().st_size
    except OSError:
        pass

    try:
        parser = etree.XMLParser(recover=True, resolve_entities=False)
        tree = etree.parse(str(path), parser)
        root = tree.getroot()
    except Exception as e:
        summary.parse_error = str(e)
        return summary

    summary.root_tag = _local_tag(root.tag)
    summary.namespaces = {k or "default": v for k, v in (root.nsmap or {}).items()}

    def own_text_len(el) -> int:
        """Chars of text sitting directly inside `el`, i.e. el.text plus the
        .tail of each direct child -- excludes text nested inside child elements."""
        n = len((el.text or "").strip())
        for child in el:
            n += len((child.tail or "").strip())
        return n

    def walk(el, depth: int):
        if not isinstance(el.tag, str):
            return  # skip comments / processing instructions
        tag = _local_tag(el.tag)
        summary.total_elements += 1
        summary.max_depth = max(summary.max_depth, depth)
        summary.tag_counts[tag] += 1
        summary.tag_own_text_chars[tag] += own_text_len(el)
        summary.tag_subtree_text_chars[tag] += len("".join(el.itertext()).strip())
        for child in el:
            walk(child, depth + 1)

    walk(root, 0)
    summary.total_text_chars = len("".join(t for t in root.itertext() if t and t.strip()))
    return summary


def summarize_dir(directory: str | Path,
                   patterns: tuple[str, ...] = ("*.cnxml", "*.xml")) -> tuple[dict[str, FileSummary], FileSummary]:
    """Summarize every matching file in a directory tree. Returns
    ({file_path: FileSummary}, aggregate_FileSummary)."""
    directory = Path(directory)
    files: set[Path] = set()
    for pattern in patterns:
        files.update(directory.rglob(pattern))

    per_file: dict[str, FileSummary] = {}
    aggregate = FileSummary(path=f"[aggregate of {directory}]")
    for f in sorted(files):
        s = summarize_file(f)
        per_file[str(f)] = s
        if not s.parse_error:
            aggregate.merge(s)
    return per_file, aggregate


def print_summary(summary: FileSummary, top_n: int | None = None) -> None:
    """Pretty-print a FileSummary as a table, sorted by tag frequency."""
    print(f"=== {summary.path} ===")
    if summary.parse_error:
        print(f"  PARSE ERROR: {summary.parse_error}")
        return

    print(f"  size: {summary.file_size_bytes:,} bytes   "
          f"elements: {summary.total_elements:,}   "
          f"max depth: {summary.max_depth}   "
          f"total text: {summary.total_text_chars:,} chars")
    if summary.root_tag:
        print(f"  root tag: <{summary.root_tag}>")
    if summary.namespaces:
        ns_str = ", ".join(f"{k}={v}" for k, v in summary.namespaces.items())
        print(f"  namespaces: {ns_str}")

    rows = summary.tag_counts.most_common(top_n)
    if not rows:
        print("  (no elements found)")
        return

    name_w = max(len(t) for t, _ in rows) + 2
    print(f"\n  {'tag':<{name_w}} {'count':>7} {'own text':>12} {'subtree text':>14} {'avg own/tag':>12}")
    print(f"  {'-'*name_w} {'-'*7} {'-'*12} {'-'*14} {'-'*12}")
    for tag, count in rows:
        own = summary.tag_own_text_chars.get(tag, 0)
        subtree = summary.tag_subtree_text_chars.get(tag, 0)
        avg = own / count if count else 0
        print(f"  {tag:<{name_w}} {count:>7,} {own:>12,} {subtree:>14,} {avg:>12.1f}")
    print("\n  'own text' = characters sitting directly inside that tag (not in nested tags)")
    print("  'subtree text' = characters inside that tag including all its descendants")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python raw_reader.py <file_or_dir> [--all] [--summary] [--top N]")
        raise SystemExit(1)

    target = Path(sys.argv[1])
    show_all = "--all" in sys.argv
    do_summary = "--summary" in sys.argv
    top_n = None
    if "--top" in sys.argv:
        top_n = int(sys.argv[sys.argv.index("--top") + 1])

    if do_summary:
        if target.is_file():
            print_summary(summarize_file(target), top_n=top_n)
        else:
            per_file, aggregate = summarize_dir(target)
            print_summary(aggregate, top_n=top_n)
            print(f"\n({len(per_file)} files summarized; pass a single file path for a per-file breakdown)")
            n_errors = sum(1 for s in per_file.values() if s.parse_error)
            if n_errors:
                print(f"\n{n_errors} file(s) failed to parse:")
                for fp, s in per_file.items():
                    if s.parse_error:
                        print(f"  {fp}: {s.parse_error}")
        raise SystemExit(0)

    if target.is_file():
        text = read_raw_text(target)
        print(f"--- {target}  ({len(text)} chars) ---")
        print(text if show_all else text[:2000])
    else:
        results = read_raw_text_dir(target)
        print(f"Found {len(results)} files")
        for fp, text in results.items():
            n = len(text)
            status = "ERROR" if text.startswith("[ERROR") else f"{n} chars"
            print(f"{fp}: {status}")
        if results and not show_all:
            first_fp, first_text = next(iter(results.items()))
            print(f"\n--- Preview of {first_fp} ---")
            print(first_text[:2000])