"""
differentia_analyzer.py
------------------------
Stage 2/3: extract raw relationship-value characteristics from a single
differentia. Purely per-record -- no corpus-wide state (that's Stage 4,
in relationship_normalizer.py).
"""

from __future__ import annotations

from .cache import CacheStore
from .llm_client import LLMCallError, LLMClient, parse_json_response
from .models import RawCharacteristic

SYSTEM_PROMPT = """You analyze the "differentia" of Aristotelian-style definitions \
(term = genus + differentia) from a textbook, and extract every distinct semantic \
characteristic the differentia expresses.

Represent each characteristic as a (relationship, value) pair.

Rules:
1. Extract every distinct characteristic explicitly stated or directly entailed by \
the differentia.
2. A differentia may yield zero, one, or multiple characteristics -- extract as many \
relationship-value pairs as are actually present, don't force exactly one.
3. Do not add facts based only on general world knowledge; extract only what the \
differentia itself supports.
4. Preserve the meaning of the original differentia -- don't paraphrase away \
important qualifiers.
5. Use concise, literal relationship names (snake_case, e.g. "accepts_deposits_from"). \
Do NOT force these into a predefined ontology -- use whatever literal relationship \
name best fits this specific differentia, even if it's very similar to but not \
identical to a relationship you'd use for a different differentia. A later, separate \
normalization pass (which you are not doing now) will consolidate the vocabulary \
across the whole corpus.
6. Use concise values while preserving important qualifiers (e.g. "individuals and \
businesses", not just "customers", if the differentia specifies both).
7. Avoid duplicate or redundant relationship-value pairs.
8. If the differentia supports no meaningful relationship-value pair, return an \
empty list.

The term and genus are given as context only, to help interpret an ambiguous \
differentia -- extract characteristics only from information the differentia itself \
supports, not from what you separately know about the term or genus.

Output ONLY valid JSON, no markdown fences, no commentary.
Schema: {"characteristics": [{"relationship": "<snake_case>", "value": "<concise text>"}, ...]}
"""

USER_TEMPLATE = """Term: {term}
Genus: {genus}
Differentia: {differentia}

Return the JSON now."""


class DifferentiaAnalyzer:
    def __init__(self, llm_client: LLMClient, cache: CacheStore | None = None,
                 max_output_tokens: int = 800):
        self.llm = llm_client
        self.cache = cache
        self.max_output_tokens = max_output_tokens

    def extract_characteristics(self, term: str, genus: str, differentia: str) -> list[RawCharacteristic]:
        cache_key = None
        if self.cache is not None:
            cache_key = self.cache.make_key("extract", self.llm.model, term, genus, differentia)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return [RawCharacteristic(**c) for c in cached]

        user_prompt = USER_TEMPLATE.format(term=term, genus=genus, differentia=differentia)
        raw = self.llm.call(SYSTEM_PROMPT, user_prompt, max_output_tokens=self.max_output_tokens)
        data = parse_json_response(raw)
        characteristics = self._validate(data, differentia)

        if self.cache is not None:
            self.cache.set(cache_key, [{"relationship": c.relationship, "value": c.value} for c in characteristics])

        return characteristics

    @staticmethod
    def _validate(data: object, differentia: str) -> list[RawCharacteristic]:
        if not isinstance(data, dict) or "characteristics" not in data:
            raise LLMCallError(f"Response missing 'characteristics' key: {data!r}")
        items = data["characteristics"]
        if not isinstance(items, list):
            raise LLMCallError(f"'characteristics' was not a list: {items!r}")

        seen: set[tuple[str, str]] = set()
        result: list[RawCharacteristic] = []
        for item in items:
            if not isinstance(item, dict) or "relationship" not in item or "value" not in item:
                raise LLMCallError(f"Malformed characteristic entry: {item!r}")
            rel = str(item["relationship"]).strip()
            val = str(item["value"]).strip()
            if not rel or not val:
                continue
            key = (rel.lower(), val.lower())
            if key in seen:
                continue  # defense in depth -- prompt already asks the model to avoid duplicates
            seen.add(key)
            result.append(RawCharacteristic(relationship=rel, value=val))

        return result
