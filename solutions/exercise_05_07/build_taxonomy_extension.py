#!/usr/bin/env python3
"""
build_taxonomy_extension.py
-----------------------------
End-to-end pipeline connecting the 28 Adobe headwind/tailwind concepts to
your Principles-of-Finance genus/differentia taxonomy:

  1. Load your existing genus_statements.owl back into memory.
  2. Apply two manually-diagnosed corrections: the `value` class's backwards
     subclass_of edges, and three near-duplicate glossary term pairs
     (economic risk/exposure, transaction risk/exposure, translation
     risk/exposure).
  3. Run add_glossary_terms() (imported from YOUR genus_class_statements.py)
     to merge the full glossary into the taxonomy as real named classes --
     costs zero LLM calls, exactly as that function already promises.
  4. For each of the 28 Adobe concepts, call Gemini to pick a genus (an
     existing class from a curated candidate shortlist, or a new one only
     when nothing fits) and write a differentia that reuses the textbook's
     own relationship-predicate vocabulary from wikidata_relation_alignment.csv.
     (C06 -- the freemium/MAU trade-off -- is hardcoded rather than left to
     the LLM: it was already determined to need a brand-new genus with no
     existing textbook analog, so that decision is recorded directly here.)
  5. Write outputs: the corrected+extended base taxonomy (OWL), a separate
     Adobe-concepts OWL fragment in the same namespace, a CSV mapping table,
     and a review log.

Requires:
  - Your own genus_class_statements.py and genus_similarity.py (from the
    project that originally produced genus_statements.owl) reachable via
    --genus-project-dir.
  - pip install -r requirements.txt
  - GEMINI_API_KEY (or GOOGLE_API_KEY) set in your environment.

Usage:
    export GEMINI_API_KEY=...
    python build_taxonomy_extension.py \\
        --genus-project-dir /path/to/your/other/project \\
        --owl-in inputs/genus_statements.owl \\
        --definitions-csv inputs/definitions_classified.csv \\
        --wikidata-csv inputs/wikidata_relation_alignment.csv \\
        --concepts-csv inputs/concepts.csv \\
        --mentions-csv inputs/mentions.csv \\
        --out-dir out

Add --dry-run to run everything except the Gemini calls (only the hardcoded
C06 override is resolved) -- useful for sanity-checking the taxonomy
parsing/fix/dedup/glossary-merge stages before spending API calls.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import owl_taxonomy as ot

THING_NAME = "thing"

# Genus branches to draw LLM candidates from -- kept short and curated
# (their full descendant sets, post-glossary-merge, are pulled in
# automatically) rather than dumping all ~180+ classes at the model, per the
# risk/income branch mapping already done for this project. Only entries
# actually present in a given --owl-in are used (see build_candidates) --
# taxonomy re-exports have already been observed to drop/rename classes
# (e.g. a former common ancestor "metric" has disappeared in one re-export),
# so this list is treated as a wishlist, not a guarantee.
CANDIDATE_ROOT_CLASSES = ["risk", "income", "total_return", "equity", "asset"]

# C06 (freemium/MAU trade-off) was confirmed, twice, to have no genus home in
# either the risk or income branch -- it's a deliberate, company-controlled
# trade-off, not an externally-imposed risk or a booked gain. Per explicit
# project decision this gets a new genus class rather than further search,
# so it is hardcoded here instead of left to the LLM to invent consistently
# across runs.
C06_OVERRIDE = {
    "genus_choice": "NEW",
    "new_genus_name": "strategic_tradeoff",
    "new_genus_parent": "thing",
    "new_genus_rationale": (
        "Neither the risk branch (externally imposed uncertainty) nor the income/return branch "
        "(realized value creation) captures a deliberate, company-controlled exchange of one value "
        "driver for another -- trading near-term ARR for long-term MAU/funnel growth. Modeled as a "
        "sibling genus to risk and income directly under thing -- deliberately not under a more specific "
        "shared ancestor like the former 'metric' class, since that class isn't guaranteed to exist under "
        "the same name across re-exports of the taxonomy, while 'thing' always does."
    ),
    "differentia_predicate": "arises_from",
    "differentia_text": (
        "a deliberate management choice to prioritize free/low-cost user acquisition (MAU growth, "
        "funnel-building) over near-term ARR -- made under the company's own control rather than "
        "imposed externally"
    ),
    "confidence": "high",
    "rationale": "Manually specified per project decision, not an LLM classification.",
}


def load_concepts(concepts_csv, mentions_csv, max_examples=3):
    with open(concepts_csv, newline="", encoding="utf-8-sig") as f:
        concepts = list(csv.DictReader(f))
    with open(mentions_csv, newline="", encoding="utf-8-sig") as f:
        mentions = list(csv.DictReader(f))

    examples_by_concept = defaultdict(list)
    for m in mentions:
        examples_by_concept[m["concept_id"]].append(m)

    out = []
    for c in concepts:
        examples = examples_by_concept.get(c["concept_id"], [])[:max_examples]
        out.append({
            "concept_id": c["concept_id"],
            "canonical_label": c["canonical_label"],
            "category": c["category"],
            "examples": [
                {"quarter": e["quarter"], "direction": e["direction"], "quote": e["quote"], "rationale": e["rationale"]}
                for e in examples
            ],
        })
    return out


def load_predicate_vocab(wikidata_csv, causal_only=True):
    # wikidata_relation_alignment.csv turns out to have ~446 distinct
    # canonical_relationship values (one row each, no dedup needed) -- most
    # are structural/factual (is_a, employer, located_in, manufacturer, ...),
    # not causal. Only ~37 are flagged is_causal=True, and those are the
    # ones actually useful for a differentia (arises_from, caused_by,
    # impacts, contributes_to, affects, impacted_by, determines, ...).
    # Defaulting to that subset keeps the prompt focused; --include-all-predicates
    # restores the full list if you want broader (noisier) coverage.
    with open(wikidata_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if causal_only:
        rows = [r for r in rows if (r.get("is_causal") or "").strip().lower() == "true"]
    seen = {}
    for r in rows:
        rel = r["canonical_relationship"]
        if rel not in seen:
            seen[rel] = {
                "canonical_relationship": rel,
                "match_property_label": r.get("match_property_label", ""),
                "example_instances": r.get("example_instances", ""),
            }
    return list(seen.values())


def build_candidates(classes, subclass_pairs, definitions_csv_for_examples, build_graph_fn, log):
    term_to_row = {}
    with open(definitions_csv_for_examples, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            term = (row.get("term") or "").strip()
            if term:
                term_to_row[term] = row

    valid_roots = [r for r in CANDIDATE_ROOT_CLASSES if r in classes]
    missing_roots = [r for r in CANDIDATE_ROOT_CLASSES if r not in classes]
    if missing_roots:
        log.append(f"NOTE: candidate root class(es) not found in this taxonomy, skipped: {missing_roots}")

    _, children_by_parent = build_graph_fn(subclass_pairs)
    descendants = ot.collect_descendants(children_by_parent, valid_roots)
    candidate_names = sorted(set(valid_roots) | descendants)

    candidates = []
    for name in candidate_names:
        cls = classes.get(name)
        example_differentia = None
        if cls:
            for mention in cls.mentions:
                row = term_to_row.get(mention)
                if row and row.get("differentia"):
                    example_differentia = row["differentia"]
                    break
        candidates.append({"class_name": name, "example_differentia": example_differentia})
    return candidates


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genus-project-dir", required=True,
                     help="Directory containing your genus_class_statements.py and genus_similarity.py")
    ap.add_argument("--owl-in", required=True)
    ap.add_argument("--definitions-csv", required=True)
    ap.add_argument("--wikidata-csv", required=True)
    ap.add_argument("--concepts-csv", required=True)
    ap.add_argument("--mentions-csv", required=True)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--include-suggested", action="store_true",
                     help="Passed through to add_glossary_terms(): fall back to suggested_genus when a term's genus is blank. Ignored when --owl-in already has the glossary merged in.")
    ap.add_argument("--glossary-mode", choices=["auto", "merged", "unmerged"], default="auto",
                     help="Whether --owl-in already has add_glossary_terms() applied. 'auto' (default) "
                          "detects this from the presence of known glossary term classes; override with "
                          "'merged' or 'unmerged' if the heuristic ever guesses wrong.")
    ap.add_argument("--include-all-predicates", action="store_true",
                     help="Send all ~446 canonical_relationship values from wikidata_relation_alignment.csv "
                          "to Gemini instead of just the ~37 flagged is_causal=True (the default). Most of "
                          "the full list is structural/factual (is_a, employer, located_in, ...) rather than "
                          "causal, so the default is usually the better fit for a differentia predicate.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Skip the Gemini calls (only the hardcoded C06 override is resolved); "
                          "useful for sanity-checking the parsing/fix/dedup/glossary-merge stages first.")
    ap.add_argument("--allow-cycles", action="store_true",
                     help="Proceed even if cycles are detected in the taxonomy after the value-class fix "
                          "and glossary merge, instead of stopping. Not recommended -- see the error message "
                          "printed when cycles are found for why the correct fix is upstream, in your own "
                          "genus_class_statements.py pipeline.")
    args = ap.parse_args()

    sys.path.insert(0, args.genus_project_dir)
    try:
        from genus_class_statements import TermClass, add_glossary_terms, write_owl, build_graph, detect_cycles
    except ImportError as e:
        raise SystemExit(
            f"Could not import genus_class_statements from {args.genus_project_dir!r} ({e}). "
            f"Point --genus-project-dir at the directory containing genus_class_statements.py "
            f"AND genus_similarity.py from your other project."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []

    # 1. Parse existing taxonomy
    classes, subclass_pairs, base_iri = ot.parse_owl_taxonomy(args.owl_in, thing_name=THING_NAME)
    log.append(f"Parsed {args.owl_in}: {len(classes)} classes, {len(subclass_pairs)} subclass_of edges, base_iri={base_iri}")

    # 2. Fix the value class's backwards edges
    subclass_pairs = ot.fix_value_class_edges(subclass_pairs, log)

    # 3/4. Merge the glossary in -- unless --owl-in already has it merged
    # (some upstream re-exports run with --definitions already applied).
    # Re-running add_glossary_terms() on an already-merged file would create
    # collision classes (economic_risk_2, etc.) instead of connecting
    # cleanly, so this branches on which shape --owl-in actually is.
    if args.glossary_mode == "auto":
        merged_already = ot.glossary_already_merged(classes)
        log.append(f"Auto-detected glossary merge state of --owl-in: {'ALREADY MERGED' if merged_already else 'not yet merged'}")
    else:
        merged_already = (args.glossary_mode == "merged")
        log.append(f"Glossary merge state forced via --glossary-mode={args.glossary_mode}")

    if merged_already:
        classes, subclass_pairs = ot.merge_duplicate_classes(classes, subclass_pairs, log)
        definitions_csv_for_examples = args.definitions_csv
    else:
        deduped_csv = out_dir / "_definitions_classified.deduped.csv"
        ot.dedupe_glossary_csv(args.definitions_csv, deduped_csv, log)
        n_before = len(classes)
        classes, subclass_pairs = add_glossary_terms(
            classes, subclass_pairs, deduped_csv,
            include_suggested=args.include_suggested, thing_name=THING_NAME,
        )
        log.append(f"add_glossary_terms(): {len(classes) - n_before} glossary terms connected -> {len(classes)} total classes")
        definitions_csv_for_examples = deduped_csv

    # 5. Safety check: confirm the edits didn't introduce (or inherit) a cycle.
    # This can ONLY be a warn-and-stop here, not a fix: break_cycles() needs the
    # original per-pair subsumption vote counts to know which edge in a cycle is
    # the least-evidenced one to safely remove, and that vote data lives in your
    # genus_pairwise_similarity.csv -- it isn't present in the flattened
    # genus_statements.owl this script reads. So if genus_statements.owl already
    # contains cycles (e.g. it was exported with --no-break-cycles, or with a
    # pairwise CSV that has grown since it was last broken), the correct fix is
    # upstream: re-run genus_class_statements.py's own cycle-breaking against the
    # current pairwise CSV and re-export, then point --owl-in at that file.
    _, children_by_parent = build_graph(subclass_pairs)
    cycles = detect_cycles(children_by_parent, list(classes.keys()))
    if cycles:
        cycles_out = out_dir / "cycles_detected.txt"
        with open(cycles_out, "w", encoding="utf-8") as f:
            f.write(f"{len(cycles)} cycle(s) detected in the taxonomy after the value-class fix and glossary merge:\n\n")
            for c in cycles:
                f.write(" -> ".join(c) + "\n")
        log.append(f"WARNING: {len(cycles)} cycle(s) detected after fixes/merge -- see {cycles_out}")
        if not args.allow_cycles:
            review_out = out_dir / "review_log.md"
            with open(review_out, "w", encoding="utf-8") as f:
                f.write("# Taxonomy extension run -- STOPPED (cycles detected)\n\n")
                for line in log:
                    f.write(f"- {line}\n")
            raise SystemExit(
                f"\n{len(cycles)} cycle(s) exist in the base taxonomy itself (full list in {cycles_out}).\n\n"
                f"This script can't safely break them here -- it would have to pick an edge to remove with "
                f"no evidence about which one is weakest, unlike your own break_cycles(), which uses the "
                f"real subsumption vote tallies from genus_pairwise_similarity.csv. The correct fix is "
                f"upstream: re-run your genus_class_statements.py pipeline (process_pairwise_csv / its CLI) "
                f"against your CURRENT pairwise CSV with cycle-breaking on (the default -- only "
                f"--no-break-cycles turns it off), re-export genus_statements.owl, and point --owl-in at "
                f"that fresh file.\n\n"
                f"Pass --allow-cycles to proceed anyway -- candidate genus lists downstream may then include "
                f"spurious classes pulled in through a cycle (e.g. 'bond'/'debt'/'security' showing up as "
                f"descendants of 'risk' only because of the cycle), and results should be treated as "
                f"provisional until the upstream taxonomy is fixed."
            )
    else:
        log.append("No cycles detected after fixes + glossary merge -- taxonomy is a valid DAG.")

    # Write the corrected + extended base taxonomy -- the "clean rebuild" of
    # your genus_statements.owl: value's edges fixed, near-duplicates merged,
    # full glossary connected.
    base_owl_out = out_dir / "principles_of_finance_taxonomy_extended.owl"
    write_owl(classes, subclass_pairs, base_owl_out, base_iri=base_iri, thing_name=THING_NAME)
    log.append(f"Wrote {base_owl_out}")

    # 6. Load the 28 Adobe concepts + predicate vocabulary + genus candidates
    concepts = load_concepts(args.concepts_csv, args.mentions_csv)
    predicate_vocab = load_predicate_vocab(args.wikidata_csv, causal_only=not args.include_all_predicates)
    candidates = build_candidates(classes, subclass_pairs, definitions_csv_for_examples, build_graph, log)
    # `thing` is always a valid genus/parent target -- write_owl() resolves it to
    # the real owl:Thing regardless of whether it's a literal key in `classes`
    # (it only becomes one via add_glossary_terms(), i.e. on the unmerged path).
    valid_class_names = set(classes.keys()) | {THING_NAME}
    valid_predicates = {p["canonical_relationship"] for p in predicate_vocab}
    log.append(f"Loaded {len(concepts)} Adobe concepts; {len(candidates)} candidate genus classes; {len(predicate_vocab)} predicates in vocabulary")

    # 7. Classify each concept
    client = None
    gemini_matcher = None
    if not args.dry_run:
        import gemini_matcher as _gemini_matcher
        gemini_matcher = _gemini_matcher
        client = gemini_matcher.make_client()

    results = []
    for concept in concepts:
        if concept["concept_id"] == "C06":
            result = dict(C06_OVERRIDE)
        elif args.dry_run:
            result = {
                "genus_choice": "EXISTING", "genus_class": "thing", "differentia_predicate": "",
                "differentia_text": "", "confidence": "low",
                "rationale": "Gemini call skipped by --dry-run; placeholder value only.",
            }
        else:
            result = gemini_matcher.classify_concept(client, args.model, concept, candidates, predicate_vocab, thing_name=THING_NAME)

        # Validate against hallucination: an EXISTING genus_class or NEW
        # genus_parent that doesn't actually exist in the taxonomy, or a
        # differentia_predicate outside the allowed vocabulary, gets AUTO-REPAIRED
        # (falls back to `thing`/blank) rather than written into the output OWL
        # as a dangling reference -- flagged and downgraded to low confidence so
        # it still surfaces for manual review, but never silently corrupts the
        # OWL export. This also covers the C06 hardcoded override: its
        # new_genus_parent ("metric") is only valid if that class still exists
        # under that name in *your* current taxonomy -- if your genus vocabulary
        # has since changed, this catches it instead of emitting a dangling edge.
        problems = []
        if result.get("genus_choice") == "EXISTING" and result.get("genus_class") not in valid_class_names:
            problems.append(f"genus_class {result.get('genus_class')!r} not found in taxonomy -- fell back to {THING_NAME!r}")
            result["genus_class"] = THING_NAME
        if result.get("genus_choice") == "NEW" and result.get("new_genus_parent") not in valid_class_names:
            problems.append(f"new_genus_parent {result.get('new_genus_parent')!r} not found in taxonomy -- fell back to {THING_NAME!r}")
            result["new_genus_parent"] = THING_NAME
        if result.get("differentia_predicate") and result["differentia_predicate"] not in valid_predicates:
            problems.append(f"differentia_predicate {result.get('differentia_predicate')!r} not in allowed vocabulary -- cleared")
            result["differentia_predicate"] = ""
        if problems:
            result["confidence"] = "low"
            result["rationale"] = f"[FLAGGED & AUTO-REPAIRED: {'; '.join(problems)}] " + result.get("rationale", "")
            log.append(f"WARNING {concept['concept_id']}: {'; '.join(problems)}")

        result["concept_id"] = concept["concept_id"]
        result["canonical_label"] = concept["canonical_label"]
        results.append(result)
        print(f"  {concept['concept_id']}: {concept['canonical_label']!r} -> "
              f"{result.get('genus_class') or result.get('new_genus_name')} ({result.get('confidence')})")

    # 8. Fold any NEW genus classes into the graph (deduping by name across concepts),
    #    and build the Adobe concept leaf classes.
    adobe_classes = {}
    adobe_subclass_pairs = []
    comments = {}
    new_genus_seen = set()

    for r in results:
        if r.get("genus_choice") == "NEW":
            gname = r["new_genus_name"]
            if gname not in new_genus_seen:
                new_genus_seen.add(gname)
                adobe_classes[gname] = TermClass(name=gname, mentions=[])
                adobe_subclass_pairs.append((gname, r["new_genus_parent"]))
                comments[gname] = "NEW GENUS -- " + r.get("new_genus_rationale", "")
            genus_for_concept = gname
        else:
            genus_for_concept = r.get("genus_class")
        r["_resolved_genus"] = genus_for_concept

        concept_class_name = r["concept_id"].lower() + "_" + ot.slugify(r["canonical_label"])
        adobe_classes[concept_class_name] = TermClass(name=concept_class_name, mentions=[r["canonical_label"]])
        if genus_for_concept:
            adobe_subclass_pairs.append((concept_class_name, genus_for_concept))
        if r.get("differentia_text"):
            comments[concept_class_name] = f"{r.get('differentia_predicate', '')}: {r['differentia_text']}"
        r["_owl_class_name"] = concept_class_name

    # 9. Write the Adobe-concepts OWL fragment (same namespace as the base file,
    #    so loading both together in Protege resolves cleanly)
    adobe_owl_out = out_dir / "adobe_taxonomy_extension.owl"
    write_owl(adobe_classes, adobe_subclass_pairs, adobe_owl_out, base_iri=base_iri, thing_name=THING_NAME)
    ot.inject_comments(adobe_owl_out, comments, base_iri)
    log.append(f"Wrote {adobe_owl_out}")

    # 10. Write the mapping CSV
    mapping_csv_out = out_dir / "adobe_concept_taxonomy_mapping.csv"
    with open(mapping_csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "concept_id", "canonical_label", "genus_class", "genus_status",
            "differentia_predicate", "differentia_text", "confidence", "rationale",
        ])
        for r in results:
            writer.writerow([
                r["concept_id"], r["canonical_label"], r.get("_resolved_genus", ""),
                "new_class" if r.get("genus_choice") == "NEW" else "existing_class",
                r.get("differentia_predicate", ""), r.get("differentia_text", ""),
                r.get("confidence", ""), r.get("rationale", ""),
            ])
    log.append(f"Wrote {mapping_csv_out}")

    # 11. Review log
    review_out = out_dir / "review_log.md"
    with open(review_out, "w", encoding="utf-8") as f:
        f.write("# Taxonomy extension run -- review log\n\n")
        for line in log:
            f.write(f"- {line}\n")
        low_conf = [r for r in results if r.get("confidence") == "low"]
        if low_conf:
            f.write("\n## Low-confidence / flagged concept placements (review manually)\n\n")
            for r in low_conf:
                f.write(f"- {r['concept_id']} {r['canonical_label']!r}: genus={r.get('_resolved_genus')}, rationale={r.get('rationale')}\n")
    print(f"\nWrote {review_out}")


if __name__ == "__main__":
    main()
