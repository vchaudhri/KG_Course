"""
pipeline.py
-----------
Orchestrates the full run: Stage 1 read -> Stage 2/3 extract (concurrent,
checkpointed) -> Stage 4 relationship normalization (sequential, corpus-
wide) -> Stage 5/6 value normalization + final structured output.

Checkpointing design: 01_raw_characteristics.jsonl IS the checkpoint --
there's no separate state file to keep in sync. On startup, Pipeline reads
whichever record ids already have a row there and skips re-extracting
them. Since each record's id is a hash of (term, genus, differentia),
changed input naturally gets a new id rather than resuming with stale
content. Stages 4-6 always re-run in full over the complete accumulated
raw file (cheap, and Stage 4's own LLM calls are cache-backed too), so a
resumed run's final output is never a stale partial mix of old and new
normalization passes.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .cache import CacheStore
from .config import PipelineConfig
from .differentia_analyzer import DifferentiaAnalyzer
from .io_utils import append_jsonl, read_input_records, read_jsonl, write_json
from .llm_client import GeminiLLMClient, LLMCallError
from .models import (
    FailureRecord, InputRecord, NormalizedCharacteristic,
    RawCharacteristic, RawExtractionResult, RecordResult,
)
from .relationship_normalizer import RelationshipNormalizer
from .value_normalizer import ValueNormalizer

RAW_CHARACTERISTICS_FILE = "01_raw_characteristics.jsonl"
RELATIONSHIP_MAPPING_FILE = "02_relationship_mapping.json"
NORMALIZED_CHARACTERISTICS_FILE = "03_normalized_characteristics.jsonl"
FAILURES_FILE = "04_failures.jsonl"


def _configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("differentia_pipeline")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    return logger


class Pipeline:
    def __init__(self, config: PipelineConfig,
                 extraction_client=None, normalization_client=None):
        self.config = config
        self.log = _configure_logging(config.log_level)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache = CacheStore(config.cache_path)

        # Callers (tests) can inject their own LLMClient implementations;
        # production use just gets Gemini clients built from config.
        self.extraction_client = extraction_client or GeminiLLMClient(
            model=config.extraction_model, api_key=config.api_key, max_retries=config.max_retries)
        self.normalization_client = normalization_client or GeminiLLMClient(
            model=config.normalization_model, api_key=config.api_key, max_retries=config.max_retries)

        self.analyzer = DifferentiaAnalyzer(self.extraction_client, cache=self.cache)
        self.relationship_normalizer = RelationshipNormalizer(
            self.normalization_client, batch_size=config.relationship_batch_size)
        self.value_normalizer = ValueNormalizer()

        self.failures: list[FailureRecord] = []

    # ---------------------------------------------------------------
    # Stage 2/3: per-record extraction
    # ---------------------------------------------------------------

    def process_record(self, record: InputRecord) -> RawExtractionResult | None:
        """Stages 1-3 for a single record. Returns None (and appends to
        self.failures) rather than raising, so one bad record never takes
        down a batch run."""
        try:
            characteristics = self.analyzer.extract_characteristics(record.term, record.genus, record.differentia)
            return RawExtractionResult(
                record_id=record.id, term=record.term, genus=record.genus,
                differentia=record.differentia, source=record.source, characteristics=characteristics,
            )
        except LLMCallError as e:
            self.log.warning(f"Extraction failed for {record.term!r}: {e}")
            self.failures.append(FailureRecord(
                stage="extract", record_id=record.id, term=record.term, error=str(e)))
            return None
        except Exception as e:  # noqa: BLE001 -- deliberately broad: never let one record crash the run
            self.log.warning(f"Unexpected error extracting {record.term!r}: {e}")
            self.failures.append(FailureRecord(
                stage="extract", record_id=record.id, term=record.term, error=repr(e)))
            return None

    # ---------------------------------------------------------------
    # Full corpus run
    # ---------------------------------------------------------------

    def process_file(self) -> dict:
        records, skipped_input_rows = read_input_records(
            self.config.input_path, include_suggested=self.config.include_suggested,
            strict=self.config.strict_input)
        self.log.info(f"Read {len(records)} input records from {self.config.input_path}")

        if skipped_input_rows:
            self.log.warning(f"Skipped {len(skipped_input_rows)} input row(s) with missing/invalid data "
                              f"(see {FAILURES_FILE} for details; pass --strict-input to abort on these instead)")
            for skip in skipped_input_rows:
                raw = skip.get("raw")
                self.failures.append(FailureRecord(
                    stage="read_input",
                    record_id="",
                    term=skip.get("term", ""),
                    error=skip["error"],
                    raw_response=json.dumps(raw, default=str) if raw is not None else None,
                ))

        raw_path = self.config.output_dir / RAW_CHARACTERISTICS_FILE
        already_done_ids = {row["record_id"] for row in read_jsonl(raw_path)}
        todo = [r for r in records if r.id not in already_done_ids]
        self.log.info(f"{len(already_done_ids)} record(s) already extracted (resuming); "
                       f"{len(todo)} to process now")

        if todo:
            self._run_extraction(todo, raw_path)

        all_raw_rows = read_jsonl(raw_path)
        self.log.info(f"{len(all_raw_rows)} record(s) with extracted characteristics total")

        mapping = self._run_relationship_normalization(all_raw_rows)
        self._run_value_normalization_and_write(all_raw_rows, mapping)
        self._write_failures()

        summary = self._build_summary(records, all_raw_rows, mapping, len(skipped_input_rows))
        self.cache.close()
        return summary

    def _run_extraction(self, todo: list[InputRecord], raw_path: Path) -> None:
        self.log.info(f"Extracting characteristics for {len(todo)} record(s) "
                       f"({self.config.max_workers} concurrent workers)...")
        completed = 0
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {pool.submit(self.process_record, r): r for r in todo}
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if result is not None:
                    append_jsonl(raw_path, result)
                if completed % 10 == 0 or completed == len(todo):
                    self.log.info(f"  {completed}/{len(todo)} extraction calls done")

    def _run_relationship_normalization(self, all_raw_rows: list[dict]) -> dict[str, str]:
        unique_relationships = sorted({
            c["relationship"] for row in all_raw_rows for c in row["characteristics"]
        })
        self.log.info(f"Normalizing {len(unique_relationships)} unique raw relationship(s)...")

        mapping = self.relationship_normalizer.normalize_vocabulary(unique_relationships)

        reverse: dict[str, list[str]] = {}
        for raw_rel, canon in mapping.items():
            reverse.setdefault(canon, []).append(raw_rel)

        write_json(self.config.output_dir / RELATIONSHIP_MAPPING_FILE, {
            "mapping": mapping,
            "canonical_to_raw": {k: sorted(v) for k, v in sorted(reverse.items())},
            "n_raw_relationships": len(unique_relationships),
            "n_canonical_relationships": len(reverse),
        })
        self.log.info(f"{len(unique_relationships)} raw relationships -> {len(reverse)} canonical "
                       f"({len(unique_relationships) - len(reverse)} merged away)")
        return mapping

    def _run_value_normalization_and_write(self, all_raw_rows: list[dict], mapping: dict[str, str]) -> None:
        out_path = self.config.output_dir / NORMALIZED_CHARACTERISTICS_FILE
        if out_path.exists():
            out_path.unlink()  # stages 4-6 always rewrite in full, see module docstring

        for row in all_raw_rows:
            try:
                normalized_chars = []
                for c in row["characteristics"]:
                    raw_char = RawCharacteristic(relationship=c["relationship"], value=c["value"])
                    canonical = mapping.get(raw_char.relationship, raw_char.relationship)
                    normalized_value = self.value_normalizer.normalize_value(raw_char.value)
                    normalized_chars.append(NormalizedCharacteristic(
                        raw_relationship=raw_char.relationship,
                        canonical_relationship=canonical,
                        raw_value=raw_char.value,
                        normalized_value=normalized_value,
                    ))
                result = RecordResult(
                    record_id=row["record_id"], term=row["term"], genus=row["genus"],
                    differentia=row["differentia"], source=row.get("source", "extracted"),
                    characteristics=normalized_chars,
                )
                append_jsonl(out_path, result)
            except Exception as e:  # noqa: BLE001
                self.log.warning(f"Value normalization failed for {row.get('term')!r}: {e}")
                self.failures.append(FailureRecord(
                    stage="normalize_values", record_id=row.get("record_id", ""),
                    term=row.get("term", ""), error=repr(e)))

    def _write_failures(self) -> None:
        if not self.failures:
            return
        path = self.config.output_dir / FAILURES_FILE
        if path.exists():
            path.unlink()
        for f in self.failures:
            append_jsonl(path, f)
        self.log.warning(f"{len(self.failures)} failure(s) recorded in {path}")

    def _build_summary(self, records: list[InputRecord], all_raw_rows: list[dict],
                        mapping: dict[str, str], n_input_rows_skipped: int = 0) -> dict:
        n_canonical = len(set(mapping.values())) if mapping else 0
        n_raw = len(mapping)
        n_suggested = sum(1 for row in all_raw_rows if row.get("source") == "suggested")
        return {
            "n_input_records": len(records),
            "n_input_rows_skipped": n_input_rows_skipped,
            "n_extracted": len(all_raw_rows),
            "n_from_suggested_genus_differentia": n_suggested,
            "n_failures": len(self.failures),
            "n_raw_relationships": n_raw,
            "n_canonical_relationships": n_canonical,
            "compression_ratio": round(1 - n_canonical / n_raw, 3) if n_raw else None,
            "cache_size": len(self.cache),
            "output_dir": str(self.config.output_dir),
        }
