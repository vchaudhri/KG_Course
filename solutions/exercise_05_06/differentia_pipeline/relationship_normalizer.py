"""
relationship_normalizer.py
----------------------------
Stage 4: analyze the vocabulary of unique RAW relationships across the
whole corpus and map each to a smaller set of canonical relationships.

This is the one genuinely corpus-wide, order-sensitive component. Batches
are processed SEQUENTIALLY (not concurrently, unlike Stage 2 extraction),
because each batch's prompt includes the canonical relationships already
established by prior batches -- without that running context, batch 2
would have no way to know batch 1 already settled on "receives_from" and
would be liable to invent "accepts_from" as a near-duplicate. That
defeats the entire point of this stage, which is measuring how much the
vocabulary compresses.
"""

from __future__ import annotations

from .llm_client import LLMCallError, LLMClient, parse_json_response

SYSTEM_PROMPT = """You are building a canonical relationship vocabulary for a \
knowledge graph, from a list of literal "raw" relationship names extracted \
independently from many textbook definitions.

Your job: map each raw relationship to a canonical relationship name that \
captures its semantics.

Rules:
1. If an EXISTING canonical relationship (listed below, if any) already captures \
the same semantics as a raw relationship, reuse it exactly (same string) -- do not \
create a near-duplicate.
2. If no existing canonical relationship adequately represents the meaning, invent a \
new one (concise, snake_case).
3. Do NOT assume that lexically similar raw relationships are semantically \
equivalent -- judge by meaning, not surface form. E.g. "has" and "produces" might \
both start with a possessive-sounding verb but mean very different things depending \
on context; don't merge them just because they're short common verbs.
4. Conversely, do not be afraid to merge lexically DIFFERENT raw relationships that \
express the same semantics (e.g. "is used for", "used to", "has purpose", "serves \
the purpose of" -> all "used_for").
5. Every raw relationship in the input list must appear as a key in your output \
mapping, mapped to exactly one canonical relationship.

Output ONLY valid JSON, no markdown fences, no commentary.
Schema: {"mapping": {"<raw_relationship>": "<canonical_relationship>", ...}}
"""

USER_TEMPLATE = """Existing canonical relationships established so far (reuse these \
where the semantics genuinely match -- this list may be empty on the first batch):
{existing}

Raw relationships to map in this batch:
{raw_list}

Return the JSON now."""


class RelationshipNormalizer:
    def __init__(self, llm_client: LLMClient, batch_size: int = 40, max_output_tokens: int = 2000):
        self.llm = llm_client
        self.batch_size = batch_size
        self.max_output_tokens = max_output_tokens

    def normalize_vocabulary(self, unique_relationships: list[str]) -> dict[str, str]:
        """Returns {raw_relationship: canonical_relationship} covering every
        relationship in `unique_relationships`. Order of the input doesn't
        affect correctness, but does affect which relationship "wins" as the
        canonical name when several raw ones merge -- callers that want
        deterministic output should pass a sorted list."""
        mapping: dict[str, str] = {}
        canonical_so_far: set[str] = set()

        batches = [unique_relationships[i:i + self.batch_size]
                   for i in range(0, len(unique_relationships), self.batch_size)]

        for batch in batches:
            batch_mapping = self._normalize_batch(batch, canonical_so_far)
            mapping.update(batch_mapping)
            canonical_so_far.update(batch_mapping.values())

        return mapping

    def _normalize_batch(self, batch: list[str], canonical_so_far: set[str]) -> dict[str, str]:
        existing_str = ", ".join(sorted(canonical_so_far)) if canonical_so_far else "(none yet)"
        raw_list_str = "\n".join(f"- {r}" for r in batch)
        user_prompt = USER_TEMPLATE.format(existing=existing_str, raw_list=raw_list_str)

        try:
            raw = self.llm.call(SYSTEM_PROMPT, user_prompt, max_output_tokens=self.max_output_tokens)
            data = parse_json_response(raw)
            batch_mapping = self._validate(data, batch)
        except LLMCallError:
            # Don't let one bad batch take down the whole normalization pass --
            # fall back to treating each of this batch's raw relationships as
            # its own canonical relationship, so the pipeline still completes
            # and the (uncompressed) result is clearly visible for review.
            batch_mapping = {r: r for r in batch}

        return batch_mapping

    @staticmethod
    def _validate(data: object, batch: list[str]) -> dict[str, str]:
        if not isinstance(data, dict) or "mapping" not in data or not isinstance(data["mapping"], dict):
            raise LLMCallError(f"Response missing a 'mapping' object: {data!r}")
        raw_mapping = data["mapping"]

        result: dict[str, str] = {}
        for r in batch:
            canon = raw_mapping.get(r)
            if not canon or not isinstance(canon, str):
                # Model dropped this one from its response -- fall back to
                # identity rather than silently losing the relationship.
                canon = r
            result[r] = canon.strip()
        return result
