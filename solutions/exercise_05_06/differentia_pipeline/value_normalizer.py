"""
value_normalizer.py
---------------------
Stage 5: conservative, purely lexical value normalization. No LLM calls,
no similarity-based merging -- deliberately narrow in scope per the spec:
trim/collapse whitespace, strip a leading article, normalize casing. Words
like "of", "for", "from", "by", "with" are NOT stripped since they can
carry real semantic weight in a value (e.g. "loans to individuals" vs
"loans from individuals" are not the same fact).

Structured as a class with normalize_value() as the single entry point,
delegating to _lexical_normalize(), specifically so a later subclass (or
a swapped-in implementation) can add embedding-based normalization or
entity linking without touching any caller of normalize_value().
"""

from __future__ import annotations

import re

_LEADING_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


class ValueNormalizer:
    def normalize_value(self, raw_value: str) -> str:
        return self._lexical_normalize(raw_value)

    def _lexical_normalize(self, raw_value: str) -> str:
        v = raw_value.strip()
        v = _WHITESPACE_RE.sub(" ", v)
        v = _LEADING_ARTICLE_RE.sub("", v)
        v = v.strip()
        v = v.lower()
        return v
