"""
gemini_matcher.py
------------------
Per-concept genus + differentia classification against the extended
Principles-of-Finance taxonomy, via Gemini structured output.

Requires: pip install google-genai
Requires: GEMINI_API_KEY (or GOOGLE_API_KEY) set in your environment.
"""
from __future__ import annotations

import json
import os

from google import genai
from google.genai import types

# What the model must return, per concept. genus_class/new_genus_* are
# conditionally required depending on genus_choice -- enforced by prompt
# instructions and re-checked by build_taxonomy_extension.py after the call
# (that check auto-repairs any dangling/invalid reference -- falls back to
# `thing` for a bad genus_class/new_genus_parent, clears a bad
# differentia_predicate -- before it reaches the output OWL).
#
# genus_class, new_genus_parent, and differentia_predicate were originally
# built as real JSON Schema `enum`s constrained to the actual candidate/
# vocabulary lists, for a structural (not just prompt-level) guarantee
# against hallucination. In practice the predicate vocabulary alone has
# ~446 distinct canonical_relationship values (it's a much finer-grained
# list than the ~20 illustrative ones used during design), and Gemini's
# structured-output compiler rejects enums that large with "the specified
# schema produces a constraint that has too many states for serving". So
# these three fields are plain strings again -- the vocabulary is still
# given to the model as prompt text (see build_prompt), just not
# schema-enforced -- and the post-hoc validation/auto-repair in
# build_taxonomy_extension.py is the only enforcement layer.
def build_response_schema():
    return {
        "type": "object",
        "properties": {
            "genus_choice": {"type": "string", "enum": ["EXISTING", "NEW"]},
            "genus_class": {
                "type": "string",
                "description": "Required if genus_choice=EXISTING: the exact class_name from the candidate list, unchanged.",
            },
            "new_genus_name": {
                "type": "string",
                "description": "Required if genus_choice=NEW: a new snake_case class name.",
            },
            "new_genus_parent": {
                "type": "string",
                "description": "Required if genus_choice=NEW: the exact class_name of an existing candidate class (or 'thing') to be this new genus's parent.",
            },
            "new_genus_rationale": {"type": "string"},
            "differentia_predicate": {
                "type": "string",
                "description": "Exact canonical_relationship value from the predicate vocabulary list -- never invent a new one.",
            },
            "differentia_text": {
                "type": "string",
                "description": "The Adobe-specific condition/content half of the differentia, in the same terse, factual prose style as the textbook's own differentiae.",
            },
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "rationale": {"type": "string"},
        },
        "required": ["genus_choice", "differentia_predicate", "differentia_text", "confidence", "rationale"],
    }

SYSTEM_INSTRUCTION = """You are helping connect company-specific "headwind/tailwind" factors from Adobe \
earnings calls to an existing genus/differentia taxonomy extracted from an OpenStax "Principles of \
Finance" textbook. For each Adobe concept you are given:
  - its canonical label, category, and a few representative verbatim quotes/rationale from earnings calls
  - a shortlist of CANDIDATE existing taxonomy classes it might belong under, each with a real textbook \
    differentia example when one is available, so you can see the actual house style
  - a fixed vocabulary of relationship predicates (from a textbook-to-Wikidata alignment) that MUST be \
    reused for the predicate half of your answer -- do not invent a new predicate name

Your job, in Aristotelian genus/differentia form:
  1. Pick the best-fitting EXISTING class from the candidate list as the genus. Only choose NEW if none \
     of the candidates are a reasonable fit for what this concept actually is -- this should be rare.
  2. Write a differentia as (predicate, text): predicate must be one of the given canonical_relationship \
     values; text is the SPECIFIC Adobe condition that distinguishes this concept within its genus, in \
     the same terse, factual register as the textbook's own differentiae -- not a marketing paraphrase \
     of the quote, and not the quote itself.
  3. Genus placement should reflect the STRUCTURAL nature of the factor, not which direction (headwind \
     vs. tailwind) it happened to realize as in a given quarter -- direction is already recorded \
     separately per mention. A factor that is externally driven and can swing either way (e.g. FX, \
     macro demand cyclicality) belongs under the risk-family genus regardless of which sign it had \
     recently. A factor that is a genuine, largely one-directional value driver (e.g. new product \
     monetization, enterprise bookings growth) belongs under the income/return-family genus.
  4. Confidence should reflect how strong the fit actually is, not how confidently you can write prose \
     about it. Use "low" whenever you are genuinely unsure, even if you still have to pick something.
"""


def build_prompt(concept, candidates, predicate_vocab):
    lines = []
    lines.append(f"CONCEPT: {concept['canonical_label']} (id={concept['concept_id']}, category={concept['category']})")
    lines.append("Representative quotes/rationale from earnings calls:")
    for q in concept["examples"]:
        lines.append(f"  - [{q['quarter']} {q['direction']}] \"{q['quote']}\" -- rationale: {q['rationale']}")
    lines.append("")
    lines.append("CANDIDATE GENUS CLASSES (pick one of these class_name values, or propose NEW if none fit):")
    for c in candidates:
        if c.get("example_differentia"):
            lines.append(f"  - {c['class_name']}  (e.g. textbook differentia: {c['example_differentia']!r})")
        else:
            lines.append(f"  - {c['class_name']}")
    lines.append("")
    lines.append("ALLOWED DIFFERENTIA PREDICATES -- canonical_relationship (Wikidata label): example usage")
    for p in predicate_vocab:
        lines.append(f"  - {p['canonical_relationship']} ({p['match_property_label']}): {p['example_instances']}")
    return "\n".join(lines)


def classify_concept(client, model, concept, candidates, predicate_vocab, thing_name="thing"):
    prompt = build_prompt(concept, candidates, predicate_vocab)
    schema = build_response_schema()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1,
        ),
    )
    return json.loads(response.text)


def make_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment before running this script, "
            "e.g.:  export GEMINI_API_KEY=your-key-here"
        )
    return genai.Client(api_key=api_key)
