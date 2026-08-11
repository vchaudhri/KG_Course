"""
taxonomy_eval.py
------------------
Evaluates a class taxonomy derived by genus_class_statements.py from
genus_pairwise_similarity.csv. Three things this can and can't do:

WHAT THIS CATCHES AUTOMATICALLY (no ground truth needed):
    1. Structural validity: cycles in the subclass_of graph. A taxonomy
       should be a DAG -- a class can never be its own ancestor -- but
       nothing in the pairwise majority-vote resolution prevents a cycle
       from emerging across three or more classes even when every
       individual pairwise vote looked locally fine (A->B and B->C and
       C->A can each win their own local vote while being globally
       inconsistent). Also reports multiple-inheritance classes, depth,
       branching factor, and connected components.
    2. Transitive-only-merge red flags: if A~B and B~C were each directly
       judged EQUIVALENT but A~C never was (or was judged something else),
       union-find still silently merges all three into one class. A single
       questionable judgment can cascade into merging genuinely different
       concepts. This flags every class where not all member pairs were
       directly confirmed EQUIVALENT, so you know exactly which merges to
       double-check.

WHAT THIS DOES NOT DO -- semantic correctness:
    No automatic check can tell you whether "a financial institution"
    really is the same category as "an organization providing financial
    services" in the sense your textbook means. That needs either a
    domain expert or a comparison against an authoritative external
    reference (e.g. FIBO for finance). What this module DOES provide is an
    efficient workflow for that: sample_for_review() exports a random
    sample of class-merges and subclass_of edges to a CSV for you to mark
    correct/incorrect, and score_review() reads it back and computes
    precision.

Usage:
    python taxonomy_eval.py genus_pairwise_similarity.csv --report

    python taxonomy_eval.py genus_pairwise_similarity.csv \\
        --sample-review --n-classes 15 --n-edges 15
    # ... fill in the "verdict" column in the exported CSVs by hand ...
    python taxonomy_eval.py genus_pairwise_similarity.csv \\
        --score-review class_review.csv edge_review.csv
"""

from __future__ import annotations

import csv
import random
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from genus_class_statements import (
    EQUIVALENT, A_SUBSUMES_B, B_SUBSUMES_A,
    PairRow, TermClass,
    read_pairwise_csv, build_classes, build_subclass_statements, add_glossary_terms,
    build_graph, detect_cycles, compute_edge_votes, break_cycles as break_cycles_fn,
)


# --------------------------------------------------------------------------
# Tier 1: structural / logical validity
# --------------------------------------------------------------------------
# build_graph() and detect_cycles() now live in genus_class_statements.py
# (break_cycles() there needs them too) -- imported above, not redefined here.

def connected_components(all_classes: list[str], subclass_pairs: list[tuple[str, str]]) -> list[set[str]]:
    """Weakly-connected components -- treats subclass_of edges as undirected,
    to find independent 'islands' of the taxonomy (most classes will be
    singleton islands with no subsumption info at all)."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for sub, sup in subclass_pairs:
        adjacency[sub].add(sup)
        adjacency[sup].add(sub)

    seen: set[str] = set()
    components: list[set[str]] = []
    for c in all_classes:
        if c in seen:
            continue
        component = set()
        queue = deque([c])
        seen.add(c)
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in adjacency.get(node, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def compute_depth(all_classes: list[str], parents_by_child: dict[str, list[str]]) -> dict[str, int]:
    """Depth of each class = length of its longest chain of ancestors.
    Roots (no parents) have depth 0. Assumes no cycles (run detect_cycles
    first -- depth is undefined/infinite-loop-prone on a cyclic graph)."""
    depth: dict[str, int] = {}

    def get_depth(c: str, visiting: set[str]) -> int:
        if c in depth:
            return depth[c]
        if c in visiting:
            return 0  # cycle guard -- shouldn't happen if detect_cycles found nothing
        parents = parents_by_child.get(c, [])
        if not parents:
            depth[c] = 0
            return 0
        visiting.add(c)
        d = 1 + max(get_depth(p, visiting) for p in parents)
        visiting.discard(c)
        depth[c] = d
        return d

    for c in all_classes:
        get_depth(c, set())
    return depth


@dataclass
class StructuralReport:
    n_classes: int
    n_subclass_edges: int
    cycles: list[list[str]]
    multi_parent_classes: dict[str, list[str]]
    root_classes: list[str]
    leaf_classes: list[str]
    max_depth: int
    mean_branching_factor: float
    n_connected_components: int
    largest_component_size: int
    class_sizes: dict[str, int]  # class -> number of original mentions
    large_classes: list[tuple[str, int]]  # outliers by size, sorted descending


def structural_report(classes: dict[str, TermClass], subclass_pairs: list[tuple[str, str]]) -> StructuralReport:
    all_names = list(classes.keys())
    parents_by_child, children_by_parent = build_graph(subclass_pairs)

    cycles = detect_cycles(children_by_parent, all_names)
    multi_parent = {c: parents for c, parents in parents_by_child.items() if len(parents) > 1}

    has_parent = set(parents_by_child.keys())
    has_child = set(children_by_parent.keys())
    roots = [c for c in all_names if c not in has_parent and c in (has_parent | has_child)]
    leaves = [c for c in all_names if c not in has_child and c in has_parent]

    depth = compute_depth(all_names, parents_by_child) if not cycles else {}
    max_depth = max(depth.values()) if depth else 0

    branching = [len(kids) for kids in children_by_parent.values() if kids]
    mean_branching = statistics.mean(branching) if branching else 0.0

    components = connected_components(all_names, subclass_pairs)
    non_trivial = [c for c in components if len(c) > 1]

    sizes = {name: len(cls.mentions) for name, cls in classes.items()}
    mean_size = statistics.mean(sizes.values()) if sizes else 0
    stdev_size = statistics.pstdev(sizes.values()) if len(sizes) > 1 else 0
    threshold = mean_size + 2 * stdev_size
    large = sorted([(n, s) for n, s in sizes.items() if s > max(threshold, 3)],
                    key=lambda kv: -kv[1])

    return StructuralReport(
        n_classes=len(all_names),
        n_subclass_edges=len(subclass_pairs),
        cycles=cycles,
        multi_parent_classes=multi_parent,
        root_classes=sorted(roots),
        leaf_classes=sorted(leaves),
        max_depth=max_depth,
        mean_branching_factor=round(mean_branching, 2),
        n_connected_components=len(non_trivial),
        largest_component_size=max((len(c) for c in components), default=0),
        class_sizes=sizes,
        large_classes=large,
    )


# --------------------------------------------------------------------------
# Tier 2: transitive-only-merge red flags
# --------------------------------------------------------------------------

@dataclass
class MergeRedFlag:
    class_name: str
    mentions: list[str]
    unconfirmed_pairs: list[tuple[str, str]]  # member pairs never directly judged EQUIVALENT


def find_transitive_only_merges(classes: dict[str, TermClass], pairs: list[PairRow]) -> list[MergeRedFlag]:
    """For every class with 2+ members, check whether EVERY pair of members
    was DIRECTLY judged EQUIVALENT in the source CSV. If any pair is
    missing entirely, or was judged something other than EQUIVALENT, the
    class is flagged -- it only holds together via transitivity through
    some other member, not because every member was mutually confirmed."""
    direct_equivalent: set[frozenset] = set()
    for p in pairs:
        if p.llm_relationship == EQUIVALENT:
            direct_equivalent.add(frozenset((p.genus_1, p.genus_2)))

    flags: list[MergeRedFlag] = []
    for name, cls in classes.items():
        if len(cls.mentions) < 2:
            continue
        unconfirmed = [
            (a, b) for a, b in combinations(cls.mentions, 2)
            if frozenset((a, b)) not in direct_equivalent
        ]
        if unconfirmed:
            flags.append(MergeRedFlag(class_name=name, mentions=cls.mentions, unconfirmed_pairs=unconfirmed))

    return flags


# --------------------------------------------------------------------------
# Tier 3: human-review sampling (this is the tool, not a substitute for
# actually looking at the data -- see module docstring)
# --------------------------------------------------------------------------

def sample_for_review(classes: dict[str, TermClass], subclass_pairs: list[tuple[str, str]],
                       n_classes: int = 15, n_edges: int = 15, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Random sample of merged classes (2+ mentions) and subclass_of edges,
    for manual correct/incorrect annotation. Returns (class_rows, edge_rows)
    ready to write to CSV."""
    rng = random.Random(seed)

    mergeable = [(name, cls) for name, cls in classes.items() if len(cls.mentions) >= 2]
    class_sample = rng.sample(mergeable, min(n_classes, len(mergeable)))
    class_rows = [
        {"term_name": name, "mentions": " | ".join(cls.mentions), "verdict": "", "notes": ""}
        for name, cls in class_sample
    ]

    edge_sample = rng.sample(subclass_pairs, min(n_edges, len(subclass_pairs)))
    edge_rows = [
        {"subclass": sub, "superclass": sup, "verdict": "", "notes": ""}
        for sub, sup in edge_sample
    ]

    return class_rows, edge_rows


def write_review_csvs(class_rows: list[dict], edge_rows: list[dict],
                       class_out: str | Path, edge_out: str | Path) -> None:
    for rows, path, fields in (
        (class_rows, class_out, ["term_name", "mentions", "verdict", "notes"]),
        (edge_rows, edge_out, ["subclass", "superclass", "verdict", "notes"]),
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def score_review(review_csv: str | Path, verdict_column: str = "verdict") -> dict:
    """Read back a filled-in review CSV (verdict column containing y/n,
    yes/no, or 1/0 -- case-insensitive) and compute precision. Rows left
    blank are excluded from the count (treated as not-yet-reviewed)."""
    yes_values = {"y", "yes", "1", "true", "correct"}
    no_values = {"n", "no", "0", "false", "incorrect"}

    with open(review_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    correct = incorrect = unreviewed = 0
    incorrect_rows = []
    for row in rows:
        verdict = (row.get(verdict_column) or "").strip().lower()
        if verdict in yes_values:
            correct += 1
        elif verdict in no_values:
            incorrect += 1
            incorrect_rows.append(row)
        else:
            unreviewed += 1

    reviewed = correct + incorrect
    precision = correct / reviewed if reviewed else None

    return {
        "n_rows": len(rows),
        "n_reviewed": reviewed,
        "n_unreviewed": unreviewed,
        "n_correct": correct,
        "n_incorrect": incorrect,
        "precision": round(precision, 4) if precision is not None else None,
        "incorrect_rows": incorrect_rows,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_structural_report(report: StructuralReport, flags: list[MergeRedFlag],
                              thing_name: str | None = None) -> None:
    print(f"\n=== Structural report ===")
    print(f"{report.n_classes} classes, {report.n_subclass_edges} subclass_of edges")

    if report.cycles:
        print(f"\n*** {len(report.cycles)} CYCLE(S) DETECTED -- these are logical inconsistencies, "
              f"the taxonomy is not a valid DAG: ***")
        for cyc in report.cycles:
            print(f"  {' -> '.join(cyc)}")
    else:
        print("No cycles detected (valid DAG).")

    print(f"\nRoots: {len(report.root_classes)}, Leaves: {len(report.leaf_classes)}, "
          f"Max depth: {report.max_depth}, Mean branching factor: {report.mean_branching_factor}")
    if thing_name:
        print(f"(Note: with --definitions, '{thing_name}' is the universal root and will legitimately "
              f"have a very high branching factor -- every orphaned genus class plus every unmatched "
              f"glossary term attaches directly to it. That's expected, not a red flag.)")
    print(f"Non-trivial connected components: {report.n_connected_components} "
          f"(largest has {report.largest_component_size} classes)")

    if report.multi_parent_classes:
        print(f"\n{len(report.multi_parent_classes)} class(es) with more than one parent "
              f"(not necessarily wrong, but worth a look):")
        for c, parents in report.multi_parent_classes.items():
            print(f"  {c} -> {parents}")

    if report.large_classes:
        print(f"\n{len(report.large_classes)} unusually large merged class(es) "
              f"(statistical outliers by mention count):")
        for name, size in report.large_classes:
            print(f"  {name}: {size} mentions")

    if flags:
        print(f"\n*** {len(flags)} class(es) formed by TRANSITIVE-ONLY merges "
              f"(not every member pair was directly confirmed EQUIVALENT): ***")
        for flag in flags:
            print(f"  {flag.class_name}  (mentions: {flag.mentions})")
            for a, b in flag.unconfirmed_pairs:
                print(f"      unconfirmed pair: {a!r} <-> {b!r}")
    else:
        print("\nNo transitive-only-merge red flags -- every merged class had all "
              "member pairs directly confirmed EQUIVALENT.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate a class taxonomy derived from genus_pairwise_similarity.csv.")
    ap.add_argument("input", help="Path to genus_pairwise_similarity.csv")
    ap.add_argument("--report", action="store_true", help="Print the structural + red-flag report.")
    ap.add_argument("--definitions", default=None,
                     help="Optional: definitions_classified.csv -- if given, also connects every "
                          "glossary term into the taxonomy before evaluating it (same behavior as "
                          "genus_class_statements.py --definitions), so the report reflects the full "
                          "taxonomy. Without this, only the genus-only layer is evaluated, which will "
                          "show as mostly small, disconnected components.")
    ap.add_argument("--include-suggested", action="store_true",
                     help="When matching a term's genus, fall back to suggested_genus if genus is "
                          "blank. Only relevant with --definitions.")
    ap.add_argument("--thing-name", default="thing",
                     help="Name of the universal root class. Only relevant with --definitions.")
    ap.add_argument("--break-cycles", action="store_true",
                     help="Apply the same cycle-breaking fix genus_class_statements.py now does by "
                          "default (remove the least-evidenced edge in each detected cycle) before "
                          "evaluating, so you can directly verify the fix resolves what --report found. "
                          "Off by default here, since this tool's job is primarily diagnostic -- showing "
                          "the raw taxonomy as-is, not silently fixing it before you see the problem.")
    ap.add_argument("--sample-review", action="store_true",
                     help="Export a random sample of classes and edges for manual review.")
    ap.add_argument("--n-classes", type=int, default=15, help="Sample size for class-merge review.")
    ap.add_argument("--n-edges", type=int, default=15, help="Sample size for subclass_of review.")
    ap.add_argument("--class-review-out", default="class_review.csv")
    ap.add_argument("--edge-review-out", default="edge_review.csv")
    ap.add_argument("--score-review", nargs=2, metavar=("CLASS_REVIEW_CSV", "EDGE_REVIEW_CSV"),
                     help="Score a filled-in pair of review CSVs and print precision.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for --sample-review.")
    args = ap.parse_args()

    if args.score_review:
        for label, path in zip(("class merges", "subclass_of edges"), args.score_review):
            result = score_review(path)
            print(f"\n=== {label} ({path}) ===")
            print(f"{result['n_reviewed']}/{result['n_rows']} reviewed "
                  f"({result['n_unreviewed']} left blank)")
            if result["precision"] is not None:
                print(f"Precision: {result['precision']:.1%} "
                      f"({result['n_correct']}/{result['n_reviewed']})")
                if result["incorrect_rows"]:
                    print("Marked incorrect:")
                    for row in result["incorrect_rows"]:
                        print(f"  {row}")
        raise SystemExit(0)

    pairs, all_genus = read_pairwise_csv(args.input)
    classes, genus_to_name = build_classes(pairs, all_genus)
    subclass_pairs, conflicts = build_subclass_statements(pairs, genus_to_name)

    if args.break_cycles:
        edge_votes = compute_edge_votes(pairs, genus_to_name)
        subclass_pairs, cycle_log = break_cycles_fn(subclass_pairs, edge_votes, list(classes.keys()))
        n_broken = len([l for l in cycle_log if l.startswith("Removed")])
        print(f"--break-cycles: removed {n_broken} edge(s) to eliminate cycles before evaluating")
        for line in cycle_log:
            print(f"  {line}")

    thing_name = None
    if args.definitions:
        n_before = len(classes)
        classes, subclass_pairs = add_glossary_terms(
            classes, subclass_pairs, args.definitions,
            include_suggested=args.include_suggested, thing_name=args.thing_name)
        thing_name = args.thing_name
        print(f"Connected glossary terms from {args.definitions} -> {len(classes)} total classes "
              f"({len(classes) - n_before} added: glossary terms + '{args.thing_name}')")

    if args.report or not args.sample_review:
        report = structural_report(classes, subclass_pairs)
        flags = find_transitive_only_merges(classes, pairs)
        _print_structural_report(report, flags, thing_name=thing_name)

    if args.sample_review:
        class_rows, edge_rows = sample_for_review(
            classes, subclass_pairs, n_classes=args.n_classes, n_edges=args.n_edges, seed=args.seed)
        write_review_csvs(class_rows, edge_rows, args.class_review_out, args.edge_review_out)
        print(f"\nWrote {len(class_rows)} class merges to {args.class_review_out} and "
              f"{len(edge_rows)} subclass_of edges to {args.edge_review_out}.")
        print("Fill in the 'verdict' column (y/n) by hand, then run:")
        print(f"  python taxonomy_eval.py {args.input} --score-review "
              f"{args.class_review_out} {args.edge_review_out}")
