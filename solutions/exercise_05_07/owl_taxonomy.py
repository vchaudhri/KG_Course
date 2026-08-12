"""
owl_taxonomy.py
----------------
Helpers for loading the existing Principles-of-Finance genus/differentia OWL
taxonomy back into the same in-memory shape genus_class_statements.py works
with (a {name: TermClass} dict plus a (sub, sup) subclass_pairs list),
applying two manually-diagnosed, one-off corrections, and writing small
string-level annotations into a generated OWL file afterwards.

This module deliberately does NOT reimplement genus_class_statements.py's
union-find / cycle-breaking / add_glossary_terms / write_owl logic -- that
module is imported directly by build_taxonomy_extension.py, so this stays a
thin wrapper around code you already trust rather than a second
implementation that could quietly drift from it.
"""
from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
import xml.sax.saxutils as saxutils
from pathlib import Path

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
XML_NS = "http://www.w3.org/XML/1998/namespace"
OWL_THING = f"{OWL_NS}Thing"


# --------------------------------------------------------------------------
# Parsing a genus_class_statements.write_owl() output back into
# (classes, subclass_pairs, base_iri)
# --------------------------------------------------------------------------

def parse_owl_taxonomy(owl_path, thing_name="thing"):
    """Round-trips a write_owl() output back into the exact shapes
    add_glossary_terms() and write_owl() expect. This only works reliably on
    OWL files produced by that same write_owl() function -- it relies on its
    simple, regular structure (one <owl:Class> per class, <rdfs:label>,
    zero or more <originalMention>, zero or more <rdfs:subClassOf>) rather
    than being a general-purpose OWL parser.
    """
    from genus_class_statements import TermClass  # requires sys.path set up by the caller

    tree = ET.parse(owl_path)
    root = tree.getroot()

    base_iri = root.get(f"{{{XML_NS}}}base")
    if not base_iri:
        raise ValueError(f"{owl_path}: <rdf:RDF> has no xml:base attribute -- is this a write_owl() output?")
    base_iri = base_iri.rstrip("#/")
    gk_ns = base_iri + "#"

    def local_name(iri):
        if iri == OWL_THING:
            return thing_name
        return iri.split("#")[-1]

    classes = {}
    subclass_pairs = []

    for cls_el in root.findall(f"{{{OWL_NS}}}Class"):
        about = cls_el.get(f"{{{RDF_NS}}}about")
        if not about:
            continue
        name = local_name(about)

        mentions = [
            (el.text or "").strip()
            for el in cls_el.findall(f"{{{gk_ns}}}originalMention")
        ]
        classes[name] = TermClass(name=name, mentions=mentions)

        for sup_el in cls_el.findall(f"{{{RDFS_NS}}}subClassOf"):
            resource = sup_el.get(f"{{{RDF_NS}}}resource")
            if resource:
                subclass_pairs.append((name, local_name(resource)))

    return classes, subclass_pairs, base_iri


# --------------------------------------------------------------------------
# Manual, evidence-based correction #1: the `value` class's backwards edges
# --------------------------------------------------------------------------

# Diagnosed by manual inspection of genus_statements.owl: the majority-vote
# subsumption tally that produced the original taxonomy emitted `value` as
# the SUBCLASS of these two more-specific value concepts, which is backwards
# -- present_discounted_value and present_value_of_the_cash_inflows_of_a_project
# are each a *kind of* value, not the other way around. Both had no other
# parent besides `metric`, so this is a straight reversal, not a merge.
VALUE_CLASS_FIXES = [
    ("value", "present_discounted_value"),
    ("value", "present_value_of_the_cash_inflows_of_a_project"),
]


def fix_value_class_edges(subclass_pairs, log):
    fixed = list(subclass_pairs)
    for sub, sup in VALUE_CLASS_FIXES:
        if (sub, sup) in fixed:
            fixed.remove((sub, sup))
            reversed_edge = (sup, sub)
            if reversed_edge not in fixed:
                fixed.append(reversed_edge)
            log.append(f"Fixed backwards edge: removed subclass_of({sub}, {sup}), added subclass_of({sup}, {sub})")
        else:
            log.append(
                f"NOTE: expected backwards edge subclass_of({sub}, {sup}) not found -- "
                f"taxonomy may have changed since this fix was diagnosed; skipped"
            )
    return fixed


# --------------------------------------------------------------------------
# Manual, evidence-based correction #2: near-duplicate glossary term pairs
# --------------------------------------------------------------------------

# Diagnosed by manual inspection of definitions_classified.csv: these three
# pairs define the same concept from two different textbook chapters under
# two different names. `adopt_alias_genus_differentia=True` means the
# canonical row's OWN definition didn't parse as Aristotelian (blank genus),
# so we copy the alias's well-formed genus/differentia onto the canonical
# row instead of dropping good content along with the duplicate name.
DUPLICATE_TERM_MERGES = [
    {"canonical": "economic risk", "alias": "economic exposure", "adopt_alias_genus_differentia": True},
    {"canonical": "transaction risk", "alias": "transaction exposure", "adopt_alias_genus_differentia": False},
    {"canonical": "translation risk", "alias": "translation exposure", "adopt_alias_genus_differentia": False},
]


def dedupe_glossary_csv(definitions_csv, out_csv, log):
    with open(definitions_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    by_term = {}
    for row in rows:
        by_term.setdefault((row.get("term") or "").strip().lower(), []).append(row)

    drop_terms = set()
    for merge in DUPLICATE_TERM_MERGES:
        canon_key, alias_key = merge["canonical"].lower(), merge["alias"].lower()
        canon_rows, alias_rows = by_term.get(canon_key), by_term.get(alias_key)
        if not canon_rows or not alias_rows:
            log.append(f"NOTE: duplicate-merge pair ({merge['canonical']!r}, {merge['alias']!r}) not both found in CSV -- skipped")
            continue
        canon_row, alias_row = canon_rows[0], alias_rows[0]
        if merge["adopt_alias_genus_differentia"]:
            for field in ("is_aristotelian", "genus", "differentia", "reason"):
                canon_row[field] = alias_row.get(field, "")
            log.append(
                f"Merged {merge['alias']!r} into {merge['canonical']!r}: adopted the alias's "
                f"genus/differentia ({alias_row.get('genus', '')!r} / {alias_row.get('differentia', '')!r}) "
                f"since the canonical row's own definition wasn't Aristotelian"
            )
        else:
            log.append(f"Merged {merge['alias']!r} into {merge['canonical']!r} (dropped as a duplicate; kept canonical's own definition)")
        drop_terms.add(alias_key)

    kept_rows = [row for row in rows if (row.get("term") or "").strip().lower() not in drop_terms]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return out_csv


# --------------------------------------------------------------------------
# Detecting whether --owl-in already has the glossary merged in
# --------------------------------------------------------------------------

# Some upstream re-exports of genus_statements.owl are run with --definitions
# already applied (add_glossary_terms() has already connected the full
# glossary as real term classes). This script needs to know which shape it's
# looking at: calling add_glossary_terms() again on an already-merged file
# would create collision classes (economic_risk_2, etc.) instead of
# connecting cleanly. These are glossary TERM class names that can only
# exist post-merge -- a majority hit rate is a reliable signal without
# requiring the caller to track and pass this explicitly.
GLOSSARY_MERGE_SIGNAL_CLASSES = [
    "economic_risk", "credit_risk", "capital_gains", "risk_premium",
    "diversifiable_risk", "firm_specific_risk",
]


def glossary_already_merged(classes, min_hits=4):
    hits = sum(1 for name in GLOSSARY_MERGE_SIGNAL_CLASSES if name in classes)
    return hits >= min_hits


# --------------------------------------------------------------------------
# Manual, evidence-based correction #2b: near-duplicate glossary CLASSES
# -- same pairs as DUPLICATE_TERM_MERGES, but applied directly to an
# already-glossary-merged graph (redirecting edges between two existing
# classes) instead of to CSV rows before add_glossary_terms() runs.
# --------------------------------------------------------------------------

DUPLICATE_CLASS_MERGES = [
    ("economic_risk", "economic_exposure"),
    ("transaction_risk", "transaction_exposure"),
    ("translation_risk", "translation_exposure"),
]


def merge_duplicate_classes(classes, subclass_pairs, log):
    from genus_class_statements import TermClass  # requires sys.path set up by the caller

    classes = dict(classes)
    pairs = list(subclass_pairs)

    for canonical, alias in DUPLICATE_CLASS_MERGES:
        if alias not in classes:
            log.append(f"NOTE: duplicate class {alias!r} not found -- nothing to merge into {canonical!r}")
            continue
        if canonical not in classes:
            log.append(f"NOTE: canonical class {canonical!r} not found -- renaming {alias!r} to {canonical!r} directly instead of merging")
            classes[canonical] = TermClass(name=canonical, mentions=classes.pop(alias).mentions)
            pairs = [(canonical if s == alias else s, canonical if p == alias else p) for s, p in pairs]
            continue

        merged_mentions = list(dict.fromkeys(classes[canonical].mentions + classes[alias].mentions))
        classes[canonical] = TermClass(name=canonical, mentions=merged_mentions)
        del classes[alias]

        new_pairs = []
        for s, p in pairs:
            s2 = canonical if s == alias else s
            p2 = canonical if p == alias else p
            if s2 == p2:
                continue  # would-be self-loop introduced by the merge -- drop
            new_pairs.append((s2, p2))
        pairs = sorted(set(new_pairs))

        log.append(f"Merged class {alias!r} into {canonical!r} directly in the graph (glossary was already merged upstream)")

    return classes, pairs


# --------------------------------------------------------------------------
# Small utilities used by build_taxonomy_extension.py
# --------------------------------------------------------------------------

def collect_descendants(children_by_parent, roots):
    """All classes reachable downward (subclasses, sub-subclasses, ...) from
    any of `roots`, via the children_by_parent map genus_class_statements.py's
    build_graph() returns."""
    seen = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        for child in children_by_parent.get(node, []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def slugify(text):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def inject_comments(owl_path, comments, base_iri):
    """Post-process a just-written write_owl() output to add an
    <rdfs:comment> to specific classes -- used to attach each new Adobe
    concept's differentia (and each new genus class's rationale) without
    needing write_owl() itself to know about a field it wasn't designed for.
    Pure string find/replace on our own freshly-generated, regularly
    formatted file -- not a general OWL editor.
    """
    base = base_iri.rstrip("#/")
    text = Path(owl_path).read_text(encoding="utf-8")
    for name, comment_text in comments.items():
        if not comment_text:
            continue
        marker = f'<owl:Class rdf:about="{base}#{name}">'
        if marker not in text:
            continue
        replacement = marker + f"\n        <rdfs:comment>{saxutils.escape(comment_text)}</rdfs:comment>"
        text = text.replace(marker, replacement, 1)
    Path(owl_path).write_text(text, encoding="utf-8")
