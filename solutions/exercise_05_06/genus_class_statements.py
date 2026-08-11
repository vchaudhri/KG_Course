"""
genus_class_statements.py
---------------------------
Turns genus_pairwise_similarity.csv (from genus_similarity.py --llm) into a
small set of logic-style statements:

    class(<Term-Name>)
    original_mentions(<Term-Name>, "<mention>")
    subclass_of(<Term-B>, <Term-A>)          # Term-B is a subclass of Term-A

Method:
    1. EQUIVALENT pairs define an equivalence relation over the raw genus
       strings. This is transitive -- if A~B and B~C, all three collapse
       into one class even if the specific A~C pair was never judged
       EQUIVALENT (or wasn't classified at all). Computed with a standard
       union-find/disjoint-set structure.
    2. Every distinct genus string that appears in the CSV ends up in
       exactly one class (a class of size 1 if it was never judged
       EQUIVALENT to anything else).
    3. Each class gets one term name: derived from its SHORTEST member
       (ties broken alphabetically), run through the same normalize_genus()
       used elsewhere in this project, then snake_cased into an atom-safe
       identifier (e.g. "a financial institution" -> financial_institution).
       Collisions (two different classes landing on the same name) get a
       numeric suffix so every class(...) atom stays unique.
    4. For every pair labeled A_SUBSUMES_B or B_SUBSUMES_A whose two genus
       strings ended up in DIFFERENT classes, that's one "vote" for a
       subclass_of relationship between those two classes. Two classes can
       have several member-pairs (e.g. class {X, Y} and class {Z, W} give
       four cross-pairs X-Z, X-W, Y-Z, Y-W), so votes are tallied per class
       pair and the MAJORITY direction is emitted once. A genuine tie/
       disagreement is reported as a conflict (printed, not silently
       resolved) and no subclass_of statement is emitted for that pair.
    5. Pairs labeled RELATED, DIFFERENT, or left unclassified (empty
       llm_relationship -- e.g. the CSV was capped with --llm-max-pairs, or
       generated without --llm at all) produce no statement.

Usage:
    python genus_class_statements.py genus_pairwise_similarity.csv --out statements.txt
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from genus_similarity import normalize_genus

EQUIVALENT = "EQUIVALENT"
A_SUBSUMES_B = "A_SUBSUMES_B"
B_SUBSUMES_A = "B_SUBSUMES_A"


# --------------------------------------------------------------------------
# Union-Find / disjoint-set over EQUIVALENT pairs
# --------------------------------------------------------------------------

class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# --------------------------------------------------------------------------
# Reading the CSV
# --------------------------------------------------------------------------

@dataclass
class PairRow:
    genus_1: str
    genus_2: str
    llm_relationship: str  # "" if unclassified/absent


def read_pairwise_csv(path: str | Path) -> tuple[list[PairRow], list[str]]:
    """Returns (rows, all_distinct_genus_values_in_first-seen_order)."""
    rows: list[PairRow] = []
    seen: dict[str, None] = {}  # ordered set
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "llm_relationship" not in reader.fieldnames:
            raise ValueError(
                f"{path} has no 'llm_relationship' column -- regenerate it with "
                f"'python genus_similarity.py ... --llm' first."
            )
        for row in reader:
            g1 = (row.get("genus_1") or "").strip()
            g2 = (row.get("genus_2") or "").strip()
            if not g1 or not g2:
                continue
            rel = (row.get("llm_relationship") or "").strip().upper()
            rows.append(PairRow(genus_1=g1, genus_2=g2, llm_relationship=rel))
            seen.setdefault(g1, None)
            seen.setdefault(g2, None)
    return rows, list(seen.keys())


# --------------------------------------------------------------------------
# Building classes
# --------------------------------------------------------------------------

@dataclass
class TermClass:
    name: str
    mentions: list[str] = field(default_factory=list)


def _snake_case(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^a-z0-9_]", "", text.lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "term"


def build_classes(pairs: list[PairRow], all_genus: list[str]) -> tuple[dict[str, TermClass], dict[str, str]]:
    """Union EQUIVALENT pairs, then name each resulting class.
    Returns (term_name -> TermClass, raw_genus -> term_name)."""
    uf = UnionFind(all_genus)
    for p in pairs:
        if p.llm_relationship == EQUIVALENT:
            uf.union(p.genus_1, p.genus_2)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for g in all_genus:
        members_by_root[uf.find(g)].append(g)

    used_names: set[str] = set()
    classes: dict[str, TermClass] = {}
    genus_to_name: dict[str, str] = {}

    # Sort roots by their eventual base name for deterministic, readable output.
    def base_name_for(members: list[str]) -> str:
        shortest = min(members, key=lambda m: (len(m), m.lower()))
        return _snake_case(normalize_genus(shortest))

    roots_sorted = sorted(members_by_root.items(), key=lambda kv: base_name_for(kv[1]))

    for root, members in roots_sorted:
        name = base_name_for(members)
        if name in used_names:
            suffix = 2
            while f"{name}_{suffix}" in used_names:
                suffix += 1
            name = f"{name}_{suffix}"
        used_names.add(name)

        members_sorted = sorted(members, key=str.lower)
        classes[name] = TermClass(name=name, mentions=members_sorted)
        for m in members:
            genus_to_name[m] = name

    return classes, genus_to_name


# --------------------------------------------------------------------------
# Deriving class-level subclass_of from member-pair votes
# --------------------------------------------------------------------------

def _tally_subsumption_votes(pairs: list[PairRow], genus_to_name: dict[str, str]) -> dict[frozenset, Counter]:
    """key: frozenset({class_x, class_y}) -> Counter of (subclass_name, superclass_name) votes.
    Shared by build_subclass_statements() and compute_edge_votes() so both stay
    in sync with exactly the same tallying logic."""
    votes: dict[frozenset, Counter] = defaultdict(Counter)
    for p in pairs:
        if p.llm_relationship not in (A_SUBSUMES_B, B_SUBSUMES_A):
            continue
        c1, c2 = genus_to_name.get(p.genus_1), genus_to_name.get(p.genus_2)
        if c1 is None or c2 is None or c1 == c2:
            continue  # already unified via EQUIVALENT transitivity, or unmapped -- no assertion

        if p.llm_relationship == A_SUBSUMES_B:
            sub, sup = c2, c1   # genus_2 (B) is the subclass of genus_1 (A)
        else:  # B_SUBSUMES_A
            sub, sup = c1, c2   # genus_1 (A) is the subclass of genus_2 (B)

        votes[frozenset((c1, c2))][(sub, sup)] += 1
    return votes


def build_subclass_statements(pairs: list[PairRow], genus_to_name: dict[str, str]) -> tuple[list[tuple[str, str]], list[str]]:
    """Tally A_SUBSUMES_B/B_SUBSUMES_A votes between class pairs and emit
    one (subclass, superclass) tuple per class pair on majority agreement.
    Returns (subclass_of_pairs, conflict_warnings).

    Note this resolves conflicts PER CLASS PAIR only -- it can still produce
    a globally inconsistent result (a cycle spanning three or more classes)
    even when every individual class-pair vote here looks locally fine. See
    break_cycles() below, which is what actually catches and fixes that."""
    votes = _tally_subsumption_votes(pairs, genus_to_name)

    subclass_pairs: list[tuple[str, str]] = []
    conflicts: list[str] = []

    for class_pair, counter in votes.items():
        if len(counter) == 1:
            (sub, sup), _ = counter.most_common(1)[0]
            subclass_pairs.append((sub, sup))
            continue

        # Genuine disagreement between member-pairs about direction.
        (top_pair, top_n), (second_pair, second_n) = counter.most_common(2)
        names = sorted(class_pair)
        if top_n > second_n:
            subclass_pairs.append(top_pair)
            conflicts.append(
                f"{names[0]} <-> {names[1]}: conflicting subsumption votes {dict(counter)} "
                f"-- used majority direction {top_pair[0]} subclass_of {top_pair[1]}"
            )
        else:
            conflicts.append(
                f"{names[0]} <-> {names[1]}: TIED subsumption votes {dict(counter)} "
                f"-- no subclass_of statement emitted, resolve manually"
            )

    subclass_pairs = sorted(set(subclass_pairs))
    return subclass_pairs, conflicts


def compute_edge_votes(pairs: list[PairRow], genus_to_name: dict[str, str]) -> dict[tuple[str, str], int]:
    """For every (sub, sup) edge that build_subclass_statements() would
    produce, how many individual member-pair judgments support it. Used by
    break_cycles() to identify the weakest (least-evidenced) edge in a
    cycle -- the one with the fewest independent judgments behind it, and
    therefore the safest to remove."""
    votes = _tally_subsumption_votes(pairs, genus_to_name)
    edge_votes: dict[tuple[str, str], int] = {}
    for counter in votes.values():
        winning_pair, winning_count = counter.most_common(1)[0]
        edge_votes[winning_pair] = winning_count
    return edge_votes


# --------------------------------------------------------------------------
# Cycle detection + breaking
# --------------------------------------------------------------------------

def build_graph(subclass_pairs: list[tuple[str, str]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Returns (parents_by_child, children_by_parent)."""
    parents_by_child: dict[str, list[str]] = defaultdict(list)
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for sub, sup in subclass_pairs:
        parents_by_child[sub].append(sup)
        children_by_parent[sup].append(sub)
    return parents_by_child, children_by_parent


def detect_cycles(children_by_parent: dict[str, list[str]], all_classes: list[str]) -> list[list[str]]:
    """Standard DFS cycle detection (white/gray/black coloring). Returns a
    list of cycles found, each as the sequence of class names forming it
    (parent -> child -> ... -> parent)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {c: WHITE for c in all_classes}
    cycles: list[list[str]] = []

    def dfs(node: str, path: list[str]):
        color[node] = GRAY
        path.append(node)
        for child in children_by_parent.get(node, []):
            if color.get(child, WHITE) == WHITE:
                dfs(child, path)
            elif color.get(child) == GRAY:
                idx = path.index(child)
                cycles.append(path[idx:] + [child])
        path.pop()
        color[node] = BLACK

    for c in all_classes:
        if color[c] == WHITE:
            dfs(c, [])
    return cycles


def break_cycles(subclass_pairs: list[tuple[str, str]], edge_votes: dict[tuple[str, str], int],
                  all_classes: list[str], max_iterations: int = 1000) -> tuple[list[tuple[str, str]], list[str]]:
    """Repeatedly detects cycles and removes the WEAKEST edge in each one --
    the (sub, sup) edge with the fewest supporting votes from
    compute_edge_votes() -- until the graph is a valid DAG.

    Why weakest-by-votes rather than some other rule: build_subclass_statements()
    already resolves disagreement WITHIN a single class pair by majority vote,
    but a cycle spanning three or more classes can exist even when every
    individual class-pair vote was locally unanimous -- each edge can look
    fine in isolation while the combination is globally inconsistent. The
    edge with the least independent evidence behind it is the one most
    likely to be the actual mistake, and the safest to remove without
    asserting something the data doesn't strongly support (same
    "when in doubt, don't force it" bias as the TIED-vote handling above and
    factor_dedup.py's density-gated merging).

    Removing one edge often resolves several overlapping cycles at once
    (they commonly share a hub class), so this re-detects from scratch each
    iteration rather than trying to fix every cycle found in one pass.

    Returns (cycle_free_subclass_pairs, removal_log)."""
    current = list(subclass_pairs)
    removed_log: list[str] = []

    for _ in range(max_iterations):
        _, children_by_parent = build_graph(current)
        cycles = detect_cycles(children_by_parent, all_classes)
        if not cycles:
            break

        cycle = cycles[0]
        edges_in_cycle = [(cycle[i + 1], cycle[i]) for i in range(len(cycle) - 1)]  # (sub, sup) per edge
        edges_in_cycle.sort(key=lambda e: edge_votes.get(e, 0))
        weakest = edges_in_cycle[0]

        if weakest in current:
            current.remove(weakest)
            removed_log.append(
                f"Removed {weakest[0]!r} subclass_of {weakest[1]!r} "
                f"({edge_votes.get(weakest, 0)} vote(s)) to break cycle: {' -> '.join(cycle)}"
            )
        else:
            # Shouldn't normally happen, but avoid an infinite loop if it does.
            removed_log.append(f"WARNING: could not resolve cycle (weakest edge already removed): {' -> '.join(cycle)}")
            break
    else:
        removed_log.append(f"WARNING: cycle-breaking did not converge after {max_iterations} iterations")

    return current, removed_log


# --------------------------------------------------------------------------
# Writing statements
# --------------------------------------------------------------------------

def write_statements(classes: dict[str, TermClass], subclass_pairs: list[tuple[str, str]],
                      out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for name in sorted(classes):
            cls = classes[name]
            f.write(f"class({name})\n")
            for mention in cls.mentions:
                escaped = mention.replace('"', '\\"')
                f.write(f'original_mentions({name}, "{escaped}")\n')
            f.write("\n")

        for sub, sup in subclass_pairs:
            f.write(f"subclass_of({sub}, {sup})\n")


def write_owl(classes: dict[str, TermClass], subclass_pairs: list[tuple[str, str]],
              out_path: str | Path, base_iri: str = "http://example.org/genus-kg",
              thing_name: str = "thing") -> None:
    """Emit the same class/original_mentions/subclass_of statements as an OWL 2
    ontology in RDF/XML -- open directly in Protege via File > Open.

        class(X)                  -> owl:Class X
        original_mentions(X, m)   -> a custom gk:originalMention annotation on X,
                                      one per mention (multi-valued)
        subclass_of(X, Y)         -> rdfs:subClassOf between the two owl:Class IRIs

    Every class also gets an rdfs:label (the term name with underscores turned
    back into spaces) so Protege's class hierarchy view reads naturally
    instead of showing raw snake_case IRIs.

    The universal root class (thing_name, default "thing" -- see
    add_glossary_terms()) is special-cased to OWL's own real owl:Thing
    rather than a synthetic class in our namespace: it's not separately
    declared (owl:Thing is a standard, implicit part of OWL itself, every
    reasoner and Protege already treats it as the root of everything), but
    subClassOf references to it correctly resolve to the real
    http://www.w3.org/2002/07/owl#Thing URI.

    base_iri sets the ontology namespace; each other class's IRI is
    f"{base_iri}#{term_name}". Pass your own if you want this to live under
    your project's own namespace instead of the placeholder default.
    """
    import xml.sax.saxutils as saxutils

    OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
    base = base_iri.rstrip("#/")

    sup_by_sub: dict[str, list[str]] = defaultdict(list)
    for sub, sup in subclass_pairs:
        sup_by_sub[sub].append(sup)

    def iri(name: str) -> str:
        if name == thing_name:
            return OWL_THING
        return f"{base}#{name}"

    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<rdf:RDF xmlns="{base}#"')
    lines.append(f'     xml:base="{base}"')
    lines.append('     xmlns:owl="http://www.w3.org/2002/07/owl#"')
    lines.append('     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"')
    lines.append('     xmlns:xml="http://www.w3.org/XML/1998/namespace"')
    lines.append('     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"')
    lines.append('     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">')
    lines.append(f'    <owl:Ontology rdf:about="{base}"/>')
    lines.append("")
    lines.append(f'    <owl:AnnotationProperty rdf:about="{iri("originalMention")}"/>')
    lines.append("")

    for name in sorted(classes):
        if name == thing_name:
            continue  # owl:Thing is implicit/built-in -- not separately declared
        cls = classes[name]
        lines.append(f'    <owl:Class rdf:about="{iri(name)}">')
        lines.append(f'        <rdfs:label>{saxutils.escape(name.replace("_", " "))}</rdfs:label>')
        for mention in cls.mentions:
            lines.append(f'        <originalMention>{saxutils.escape(mention)}</originalMention>')
        for sup in sup_by_sub.get(name, []):
            lines.append(f'        <rdfs:subClassOf rdf:resource="{iri(sup)}"/>')
        lines.append('    </owl:Class>')
        lines.append("")

    lines.append('</rdf:RDF>')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------

def add_glossary_terms(classes: dict[str, TermClass], subclass_pairs: list[tuple[str, str]],
                        definitions_csv: str | Path, include_suggested: bool = False,
                        thing_name: str = "thing") -> tuple[dict[str, TermClass], list[tuple[str, str]]]:
    """Extends a genus-class taxonomy (from build_classes/build_subclass_statements)
    with the actual glossary terms from definitions_classified.csv, so the
    taxonomy connects genus CATEGORIES to the real TERMS that populate them
    -- previously the taxonomy only related genus phrases to each other and
    never referenced the glossary terms at all.

    For each term:
        subclass_of(term, genus_class)   if the term's genus resolves to a
                                          known class (via the same
                                          normalize_genus() matching
                                          factor_candidates.py already uses)
        subclass_of(term, thing)         otherwise -- blank genus, or a
                                          genus that matched no known class

    Every genus class that currently has no parent (a taxonomy root) also
    gets subclass_of(root, thing) -- this is what turns the taxonomy from a
    forest of disconnected genus trees into one connected tree. Classes
    that already have a parent are untouched; they already reach "thing"
    transitively once their root does.

    Every row in definitions_csv is included regardless of is_aristotelian
    status -- a term that failed the Aristotelian test is still a
    legitimate glossary concept worth a place in the taxonomy.

    include_suggested: fall back to suggested_genus when genus is blank,
    same flag/meaning as elsewhere in this project (default off).

    Costs zero LLM calls -- purely a normalized-string lookup against
    classes' existing original_mentions."""
    new_classes: dict[str, TermClass] = dict(classes)
    new_subclass_pairs: list[tuple[str, str]] = list(subclass_pairs)

    mention_to_class: dict[str, str] = {}
    for name, cls in classes.items():
        for mention in cls.mentions:
            mention_to_class[normalize_genus(mention)] = name

    if thing_name not in new_classes:
        new_classes[thing_name] = TermClass(name=thing_name, mentions=[])

    has_parent = {sub for sub, _ in subclass_pairs}
    for name in classes:
        if name != thing_name and name not in has_parent:
            new_subclass_pairs.append((name, thing_name))

    used_names: set[str] = set(new_classes.keys())

    with open(definitions_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = (row.get("term") or "").strip()
            if not term:
                continue

            genus = (row.get("genus") or "").strip()
            if include_suggested and not genus:
                genus = (row.get("suggested_genus") or "").strip()

            term_class_name = _snake_case(normalize_genus(term, singularize=False))
            if term_class_name in used_names:
                suffix = 2
                while f"{term_class_name}_{suffix}" in used_names:
                    suffix += 1
                term_class_name = f"{term_class_name}_{suffix}"
            used_names.add(term_class_name)

            new_classes[term_class_name] = TermClass(name=term_class_name, mentions=[term])

            parent = mention_to_class.get(normalize_genus(genus)) if genus else None
            new_subclass_pairs.append((term_class_name, parent or thing_name))

    return new_classes, new_subclass_pairs


def process_pairwise_csv(in_csv: str | Path, out_path: str | Path,
                          owl_out: str | Path | None = None,
                          owl_iri: str = "http://example.org/genus-kg",
                          definitions_csv: str | Path | None = None,
                          include_suggested: bool = False,
                          thing_name: str = "thing",
                          break_cycles_: bool = True) -> dict:
    pairs, all_genus = read_pairwise_csv(in_csv)
    if not all_genus:
        raise ValueError(f"No genus values found in {in_csv}")

    classes, genus_to_name = build_classes(pairs, all_genus)
    subclass_pairs, conflicts = build_subclass_statements(pairs, genus_to_name)

    cycle_log: list[str] = []
    if break_cycles_:
        edge_votes = compute_edge_votes(pairs, genus_to_name)
        subclass_pairs, cycle_log = break_cycles(subclass_pairs, edge_votes, list(classes.keys()))

    n_genus_classes = len(classes)
    n_glossary_terms = 0
    if definitions_csv:
        thing_was_new = thing_name not in classes
        n_before = len(classes)
        classes, subclass_pairs = add_glossary_terms(
            classes, subclass_pairs, definitions_csv,
            include_suggested=include_suggested, thing_name=thing_name)
        n_glossary_terms = len(classes) - n_before - (1 if thing_was_new else 0)

    write_statements(classes, subclass_pairs, out_path)

    if owl_out:
        write_owl(classes, subclass_pairs, owl_out, base_iri=owl_iri, thing_name=thing_name)

    n_equivalent = sum(1 for p in pairs if p.llm_relationship == EQUIVALENT)
    n_subsumption = sum(1 for p in pairs if p.llm_relationship in (A_SUBSUMES_B, B_SUBSUMES_A))
    n_unclassified = sum(1 for p in pairs if p.llm_relationship not in
                          (EQUIVALENT, A_SUBSUMES_B, B_SUBSUMES_A, "RELATED", "DIFFERENT"))

    return {
        "n_distinct_genus": len(all_genus),
        "n_classes": n_genus_classes,
        "n_merged_away": len(all_genus) - n_genus_classes,
        "n_equivalent_pairs": n_equivalent,
        "n_subsumption_pairs_seen": n_subsumption,
        "n_subclass_of_statements": len(subclass_pairs),
        "n_unclassified_pairs": n_unclassified,
        "conflicts": conflicts,
        "n_cycles_broken": len([l for l in cycle_log if l.startswith("Removed")]),
        "cycle_break_log": cycle_log,
        "n_glossary_terms_added": n_glossary_terms if definitions_csv else None,
        "n_total_classes": len(classes),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Build class()/original_mentions()/subclass_of() statements from "
                    "genus_pairwise_similarity.csv (must have been generated with --llm).")
    ap.add_argument("input", help="Path to genus_pairwise_similarity.csv")
    ap.add_argument("--out", default="genus_statements.txt", help="Output text file path.")
    ap.add_argument("--owl", default=None,
                     help="Also write an OWL 2 ontology (RDF/XML) to this path -- "
                          "open directly in Protege via File > Open.")
    ap.add_argument("--owl-iri", default="http://example.org/genus-kg",
                     help="Base IRI/namespace for the OWL ontology and its classes "
                          "(each class becomes <base-iri>#<term_name>).")
    ap.add_argument("--definitions", default=None,
                     help="Optional: definitions_classified.csv -- if given, also connects every "
                          "glossary term into the taxonomy (subclass_of its resolved genus class, "
                          "or 'thing' if unmatched), and roots every genus class that currently has "
                          "no parent under 'thing' too -- turns the taxonomy from a forest of "
                          "disconnected genus trees into one connected tree. Costs zero LLM calls.")
    ap.add_argument("--include-suggested", action="store_true",
                     help="When matching a term's genus, fall back to suggested_genus if genus is "
                          "blank. Only relevant with --definitions.")
    ap.add_argument("--thing-name", default="thing",
                     help="Name of the universal root class (also OWL's real owl:Thing in the OWL export).")
    ap.add_argument("--no-break-cycles", action="store_true",
                     help="Disable automatic cycle-breaking (on by default). A subclass_of hierarchy "
                          "with cycles isn't a valid DAG -- see break_cycles()'s docstring for why this "
                          "can happen even when every individual class-pair vote looked fine, and why "
                          "removing the least-evidenced edge in each cycle is the fix. Disable only if "
                          "you want to inspect the raw, unfixed graph yourself.")
    args = ap.parse_args()

    summary = process_pairwise_csv(args.input, args.out, owl_out=args.owl, owl_iri=args.owl_iri,
                                    definitions_csv=args.definitions, include_suggested=args.include_suggested,
                                    thing_name=args.thing_name, break_cycles_=not args.no_break_cycles)

    print(f"{summary['n_distinct_genus']} distinct genus values -> {summary['n_classes']} classes "
          f"({summary['n_merged_away']} merged away via EQUIVALENT)")
    print(f"{summary['n_equivalent_pairs']} EQUIVALENT pairs, "
          f"{summary['n_subsumption_pairs_seen']} subsumption pairs seen "
          f"-> {summary['n_subclass_of_statements']} subclass_of statements")
    if summary["n_unclassified_pairs"]:
        print(f"{summary['n_unclassified_pairs']} pairs had no usable llm_relationship (skipped)")
    if summary["conflicts"]:
        print(f"\n{len(summary['conflicts'])} conflict(s) in subsumption direction between class pairs:")
        for c in summary["conflicts"]:
            print(f"  {c}")
    if not args.no_break_cycles:
        if summary["n_cycles_broken"]:
            print(f"\n{summary['n_cycles_broken']} cycle(s) detected and broken by removing the "
                  f"least-evidenced edge in each:")
            for line in summary["cycle_break_log"]:
                print(f"  {line}")
        else:
            print("\nNo cycles detected -- taxonomy is a valid DAG.")
    if args.definitions:
        print(f"\n{summary['n_glossary_terms_added']} glossary term(s) connected into the taxonomy "
              f"-> {summary['n_total_classes']} total classes (genus classes + terms + 'thing')")
    print(f"\nWrote {args.out}" + (f", {args.owl}" if args.owl else ""))
