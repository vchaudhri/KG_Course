"""
wikidata_relation_alignment.py
--------------------------------
Perspectives 1 + 2 of the relation-vocabulary analysis: for each CANONICAL
relationship produced by differentia_pipeline (02_relationship_mapping.json),
determine (1) whether it aligns with a general-purpose Wikidata property,
and (2), for free from the same lookup, whether it (or its match) belongs
to Wikidata's causal-relation family (has cause / has effect / has
contributing factor / influenced by / ...).

Why Wikidata rather than BFO/DOLCE: those are deliberately domain-empty
(BFO explicitly states it excludes "terms particular to material
domains"), so a finance-textbook relation vocabulary would show near-zero
alignment against them -- true, but not a useful signal. Wikidata has both
a foundational layer (instance_of / subclass_of / part_of) AND a
documented causal-relation family AND real business/org relations, giving
a much better chance of a meaningful match while still being a
recognized, external reference rather than something invented for this
project.

Reference set provenance: WIKIDATA_REFERENCE_PROPERTIES below is a curated
~40-property subset (not all ~12,000 Wikidata properties, most of which
are identifier/external-ID properties irrelevant here). Properties marked
id_verified=True had their P-ID confirmed directly against wikidata.org
during research for this module. id_verified=False means the label and
semantics are well-established, common-knowledge Wikidata properties, but
the exact P-number was NOT independently re-confirmed this session --
treat those IDs as "best recollection, verify before citing formally."
This doesn't affect the alignment analysis itself, which classifies on
LABEL + DESCRIPTION semantics, not on the numeric ID.

Usage:
    python wikidata_relation_alignment.py \\
        differentia_output/02_relationship_mapping.json \\
        differentia_output/03_normalized_characteristics.jsonl \\
        --out wikidata_relation_alignment.csv
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from differentia_pipeline.llm_client import GeminiLLMClient, LLMCallError, LLMClient, parse_json_response


@dataclass
class WikidataProperty:
    id: str
    label: str
    description: str
    category: str  # "core" | "causal" | "temporal" | "location_org" | "usage" | "business"
    id_verified: bool


WIKIDATA_REFERENCE_PROPERTIES: list[WikidataProperty] = [
    # --- core structural (confirmed) ---
    WikidataProperty("P31", "instance of", "that class of which this subject is a particular example and member", "core", True),
    WikidataProperty("P279", "subclass of", "next higher class or type; a member of this class is a subtype/instance of that other class", "core", True),
    WikidataProperty("P361", "part of", "object of which the subject is a part", "core", True),
    WikidataProperty("P527", "has part", "part of this subject; inverse of P361", "core", True),
    WikidataProperty("P1269", "facet of", "topic of which this item is an aspect, item that offers a broader perspective on the same topic", "core", False),

    # --- causal-relation family (confirmed via wikidata.org's WikiProject Influence) ---
    WikidataProperty("P828", "has cause", "underlying cause, thing that ultimately resulted in this effect", "causal", True),
    WikidataProperty("P1542", "has effect", "effect of this cause", "causal", True),
    WikidataProperty("P1478", "has immediate cause", "immediate cause of an event, state or process", "causal", True),
    WikidataProperty("P1536", "immediate cause of", "this entity is the immediate cause of the other entity", "causal", True),
    WikidataProperty("P1479", "has contributing factor", "thing that contributes to but does not directly result in this effect", "causal", True),
    WikidataProperty("P1537", "contributing factor of", "thing that is significantly influenced by this cause, but does not directly result from it", "causal", True),
    WikidataProperty("P737", "influenced by", "this person, idea, etc. is informed by that other person, idea, etc.", "causal", True),
    WikidataProperty("P941", "inspired by", "work, human, place or event which inspired this creative work or fictional entity", "causal", True),
    WikidataProperty("P144", "based on", "the work(s) used as the basis for the subject item", "causal", True),
    WikidataProperty("P509", "cause of death", "underlying or immediate cause of death", "causal", True),
    WikidataProperty("P770", "cause of destruction", "event that caused a thing to cease existing or be rendered permanently unusable", "causal", True),

    # --- temporal / process (confirmed) ---
    WikidataProperty("P571", "inception", "date or point in time when the subject came into existence", "temporal", True),
    WikidataProperty("P576", "dissolved, abolished or demolished", "point in time at which the subject ceased to exist", "temporal", True),
    WikidataProperty("P580", "start time", "time an item begins to exist or a statement starts being valid", "temporal", True),
    WikidataProperty("P582", "end time", "time an item ceases to exist or a statement stops being valid", "temporal", True),
    WikidataProperty("P585", "point in time", "time and date something took place", "temporal", True),
    WikidataProperty("P710", "participant", "person, group of people or organization that actively takes part in an event", "temporal", True),
    WikidataProperty("P607", "conflict", "battles, wars, or other events the subject participated in", "temporal", True),

    # --- location / organization structure (confirmed) ---
    WikidataProperty("P131", "located in the administrative territorial entity", "the item is located on the territory of the following administrative entity", "location_org", True),
    WikidataProperty("P706", "located in/on physical feature", "located on the specified landform", "location_org", True),
    WikidataProperty("P740", "location of formation", "location where a group or organization was formed", "location_org", True),
    WikidataProperty("P749", "parent organization", "parent organization of an organization", "location_org", True),
    WikidataProperty("P937", "work location", "location where persons or organizations were actively participating in employment, business, or other work", "location_org", True),
    WikidataProperty("P108", "employer", "person or organization for which the subject works or worked", "location_org", True),
    WikidataProperty("P463", "member of", "organization, club, or musical group to which the subject belongs", "location_org", True),
    WikidataProperty("P159", "headquarters location", "city where an organization's headquarters is or has been situated", "location_org", True),
    WikidataProperty("P17", "country", "sovereign state that this item is in", "location_org", True),
    WikidataProperty("P176", "manufacturer", "manufacturer or producer of this product", "location_org", True),

    # --- usage / study (confirmed) ---
    WikidataProperty("P1535", "used by", "item or concept that makes use of the subject", "usage", True),
    WikidataProperty("P2579", "studied by", "subject is studied by this science or domain", "usage", True),

    # --- business / finance-relevant (labels well-established; P-ID not re-verified this session) ---
    WikidataProperty("P355", "subsidiary", "subsidiary of a company or organization; opposite of parent organization", "business", False),
    WikidataProperty("P1830", "owner of", "entities owned by the subject", "business", False),
    WikidataProperty("P127", "owned by", "owner of the subject", "business", False),
    WikidataProperty("P452", "industry", "industry of a company or organization", "business", False),
    WikidataProperty("P414", "stock exchange", "exchange on which a stock is traded", "business", False),
    WikidataProperty("P1056", "product or material produced", "material or product produced or sold by an organization", "business", False),
    WikidataProperty("P5642", "risk factor", "biological, chemical, physical, social, or economic factor that raises the probability of an adverse outcome", "business", False),
]

VALID_PROPERTY_IDS = {p.id for p in WIKIDATA_REFERENCE_PROPERTIES}
PROPERTY_BY_ID = {p.id: p for p in WIKIDATA_REFERENCE_PROPERTIES}
CAUSAL_PROPERTY_IDS = {p.id for p in WIKIDATA_REFERENCE_PROPERTIES if p.category == "causal"}


def _reference_listing() -> str:
    return "\n".join(f"- {p.label} ({p.id}): {p.description} [{p.category}]" for p in WIKIDATA_REFERENCE_PROPERTIES)


SYSTEM_PROMPT = f"""You are aligning a finance textbook's extracted relationship \
vocabulary against a reference set of general-purpose Wikidata properties, to \
determine (a) whether each relationship corresponds to a recognized general-purpose \
relation, and (b) whether it expresses a causal/influence relationship.

Reference properties:
{_reference_listing()}

For each numbered relationship below (given with a few example (term, value) \
instances it was used for, for context), decide:
1. match_property_id: the id (e.g. "P828") of the BEST matching reference property \
above, or null if none genuinely fits -- don't force a weak match.
2. match_type: "EXACT" (same meaning), "SPECIALIZATION" (the relationship is a more \
specific case of the reference property), "ANALOGOUS" (meaningfully similar but not \
a clean fit), or "NONE" (no reasonable match; use this whenever match_property_id is \
null).
3. is_causal: true if EITHER the matched property is in the "causal" category above, \
OR the relationship's own semantics are clearly causal/influence-bearing (e.g. an \
"increases", "reduces", "leads_to" type relation) even without a strong reference \
match.

Output ONLY valid JSON, no markdown fences, no commentary.
Schema: {{"classifications": {{"<relationship_id>": {{"match_property_id": <"Pxxx" or null>, \
"match_type": "<EXACT|SPECIALIZATION|ANALOGOUS|NONE>", "is_causal": <true|false>}}, ...}}}}
"""

USER_ITEM_TEMPLATE = "Relationship {id}: {name}\n  Examples: {examples}"


# --------------------------------------------------------------------------
# Loading canonical relationships + example instances
# --------------------------------------------------------------------------

def load_canonical_relationships(mapping_path: str | Path) -> list[str]:
    with open(mapping_path, encoding="utf-8") as f:
        data = json.load(f)
    canon_to_raw = data.get("canonical_to_raw", {})
    return sorted(canon_to_raw.keys())


def load_examples_per_relationship(characteristics_path: str | Path,
                                    max_examples: int = 3) -> dict[str, list[tuple[str, str]]]:
    examples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    path = Path(characteristics_path)
    if not path.exists():
        return examples
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            term = row.get("term", "")
            for c in row.get("characteristics", []):
                canon = c.get("canonical_relationship")
                val = c.get("normalized_value") or c.get("raw_value", "")
                if canon and len(examples[canon]) < max_examples:
                    examples[canon].append((term, val))
    return examples


# --------------------------------------------------------------------------
# LLM classification
# --------------------------------------------------------------------------

def _classify_batch(client: LLMClient, batch: list[tuple[str, str, list[tuple[str, str]]]]) -> dict[str, dict]:
    """batch: list of (relationship_id_str, canonical_relationship_name, examples).
    Returns {canonical_relationship_name: {match_property_id, match_type, is_causal}}."""
    items = []
    for rel_id, rel_name, examples in batch:
        ex_str = "; ".join(f'"{t}" -> "{v}"' for t, v in examples) if examples else "(no examples available)"
        items.append(USER_ITEM_TEMPLATE.format(id=rel_id, name=rel_name, examples=ex_str))
    user_prompt = "\n\n".join(items) + "\n\nReturn the JSON now."

    try:
        raw = client.call(SYSTEM_PROMPT, user_prompt, max_output_tokens=max(600, 150 * len(batch)))
        data = parse_json_response(raw)
        classifications = data.get("classifications", {}) if isinstance(data, dict) else {}
    except LLMCallError:
        classifications = {}

    result: dict[str, dict] = {}
    for rel_id, rel_name, _ in batch:
        c = classifications.get(rel_id) if isinstance(classifications, dict) else None
        mpid = c.get("match_property_id") if isinstance(c, dict) else None
        if mpid not in VALID_PROPERTY_IDS:
            mpid = None
        match_type = (c.get("match_type") if isinstance(c, dict) else None) or "NONE"
        if mpid is None:
            match_type = "NONE"
        is_causal = bool(c.get("is_causal", False)) if isinstance(c, dict) else False
        if mpid in CAUSAL_PROPERTY_IDS:
            is_causal = True
        result[rel_name] = {"match_property_id": mpid, "match_type": match_type, "is_causal": is_causal}
    return result


def align_relationships(mapping_path: str | Path, characteristics_path: str | Path,
                         model: str = "gemini-2.5-flash", api_key: str | None = None,
                         batch_size: int = 15, max_workers: int = 5,
                         client: LLMClient | None = None) -> list[dict]:
    canonical_rels = load_canonical_relationships(mapping_path)
    examples_by_rel = load_examples_per_relationship(characteristics_path)

    llm = client or GeminiLLMClient(model=model, api_key=api_key)

    indexed = [(str(i), rel, examples_by_rel.get(rel, [])) for i, rel in enumerate(canonical_rels)]
    batches = [indexed[i:i + batch_size] for i in range(0, len(indexed), batch_size)]

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Batches are independent (fixed external reference set, no running
        # context needed between them -- unlike differentia_pipeline's
        # relationship_normalizer.py), so unlike that module these run concurrently.
        futures = {pool.submit(_classify_batch, llm, b): b for b in batches}
        for fut in as_completed(futures):
            results.update(fut.result())

    rows = []
    for rel in canonical_rels:
        c = results.get(rel, {"match_property_id": None, "match_type": "NONE", "is_causal": False})
        prop = PROPERTY_BY_ID.get(c["match_property_id"]) if c["match_property_id"] else None
        rows.append({
            "canonical_relationship": rel,
            "example_instances": "; ".join(f"{t} -> {v}" for t, v in examples_by_rel.get(rel, [])[:3]),
            "match_property_id": c["match_property_id"] or "",
            "match_property_label": prop.label if prop else "",
            "match_property_id_verified": prop.id_verified if prop else "",
            "match_type": c["match_type"],
            "is_causal": c["is_causal"],
        })
    return rows


def write_alignment_csv(rows: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["canonical_relationship", "example_instances", "match_property_id",
                  "match_property_label", "match_property_id_verified", "match_type", "is_causal"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    n_matched = sum(1 for r in rows if r["match_property_id"])
    n_causal = sum(1 for r in rows if r["is_causal"])
    by_match_type: dict[str, int] = defaultdict(int)
    for r in rows:
        by_match_type[r["match_type"]] += 1
    return {
        "n_canonical_relationships": n,
        "n_matched_to_wikidata": n_matched,
        "match_rate": round(n_matched / n, 3) if n else None,
        "n_causal_relevant": n_causal,
        "by_match_type": dict(by_match_type),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Align differentia_pipeline's canonical relationships against a curated Wikidata "
                    "property reference set, and flag causal/influence-relevant ones.")
    ap.add_argument("mapping", help="Path to 02_relationship_mapping.json (from differentia_pipeline).")
    ap.add_argument("characteristics", help="Path to 03_normalized_characteristics.jsonl "
                                             "(from differentia_pipeline; used for example-instance context).")
    ap.add_argument("--out", default="wikidata_relation_alignment.csv")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--max-workers", type=int, default=5)
    ap.add_argument("--api-key", default=None, help="Gemini API key (else reads GEMINI_API_KEY/GOOGLE_API_KEY env var).")
    args = ap.parse_args()

    rows = align_relationships(args.mapping, args.characteristics, model=args.model,
                                api_key=args.api_key, batch_size=args.batch_size, max_workers=args.max_workers)
    write_alignment_csv(rows, args.out)

    summary = summarize(rows)
    print(f"\n{summary['n_canonical_relationships']} canonical relationships -> "
          f"{summary['n_matched_to_wikidata']} matched a Wikidata property "
          f"({summary['match_rate']:.1%} match rate)" if summary['n_canonical_relationships'] else "\nNo relationships found.")
    print(f"{summary['n_causal_relevant']} flagged as causal/influence-relevant")
    print(f"By match type: {summary['by_match_type']}")
    print(f"\nWrote {args.out}")
