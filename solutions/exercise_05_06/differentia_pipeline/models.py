"""
models.py
---------
Core data structures, using plain dataclasses (not Pydantic -- the rest of
this project has stayed dependency-light, and dataclasses plus explicit
validation in llm_client.py's response parsing cover what's needed here
without adding a new dependency). Every record carries a stable `id`
derived by hashing (term, genus, differentia), which is what makes both
the LLM cache and the Stage 2 checkpoint/resume mechanism work: identical
input always produces the same id, so re-running on unchanged input is a
no-op, and changed input naturally gets a new id rather than silently
reusing stale results.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def make_record_id(term: str, genus: str, differentia: str) -> str:
    h = hashlib.sha256(f"{term}\x1f{genus}\x1f{differentia}".encode("utf-8")).hexdigest()
    return h[:16]


@dataclass
class InputRecord:
    term: str
    genus: str
    differentia: str
    source: str = "extracted"  # "extracted" (real genus/differentia) or "suggested"
                                # (fell back to aristotelian_classifier.py's
                                # suggested_genus/suggested_differentia -- see io_utils.py)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = make_record_id(self.term, self.genus, self.differentia)


@dataclass
class RawCharacteristic:
    relationship: str
    value: str


@dataclass
class RawExtractionResult:
    """Stage 2/3 output: one row per input record, raw characteristics only."""
    record_id: str
    term: str
    genus: str
    differentia: str
    source: str = "extracted"
    characteristics: list[RawCharacteristic] = field(default_factory=list)


@dataclass
class NormalizedCharacteristic:
    raw_relationship: str
    canonical_relationship: str
    raw_value: str
    normalized_value: str


@dataclass
class RecordResult:
    """Stage 6 final output: one row per input record, fully normalized."""
    record_id: str
    term: str
    genus: str
    differentia: str
    source: str = "extracted"
    characteristics: list[NormalizedCharacteristic] = field(default_factory=list)


@dataclass
class FailureRecord:
    stage: str          # "extract" | "normalize_relationships" | "normalize_values"
    record_id: str
    term: str
    error: str
    raw_response: str | None = None
