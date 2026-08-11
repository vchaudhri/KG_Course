"""
cli.py
------
Usage:
    python -m differentia_pipeline.cli input.csv --out differentia_output
    python -m differentia_pipeline.cli input.jsonl --out differentia_output
"""

from __future__ import annotations

import argparse

from .config import PipelineConfig
from .pipeline import Pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Extract and normalize relationship-value characteristics from genus-differentia definitions.")
    ap.add_argument("input", help="Path to input file: .csv, .json (array), or .jsonl/.ndjson "
                                   "(format is auto-detected from the extension). Each record needs "
                                   "term, genus, differentia -- extra CSV columns are ignored, so this "
                                   "can point directly at definitions_classified.csv (rows must have "
                                   "genus/differentia filled in, i.e. is_aristotelian == 'yes').")
    ap.add_argument("--out", default="differentia_output", help="Output directory.")
    ap.add_argument("--include-suggested", action="store_true",
                     help="Fall back to suggested_genus/suggested_differentia (from "
                          "aristotelian_classifier.py's definitions_classified.csv) when genus/differentia "
                          "are blank, instead of erroring on that row. Off by default -- see io_utils.py's "
                          "read_input_records docstring for why this isn't the default.")
    ap.add_argument("--strict-input", action="store_true",
                     help="Abort the whole run on the first input row with missing/invalid data "
                          "(old behavior). Off by default -- bad rows are skipped and recorded in "
                          "04_failures.jsonl instead, so one malformed row out of hundreds doesn't "
                          "block everything else.")
    ap.add_argument("--extraction-model", default="gemini-2.5-flash")
    ap.add_argument("--normalization-model", default="gemini-2.5-flash")
    ap.add_argument("--api-key", default=None, help="Gemini API key (else reads GEMINI_API_KEY/GOOGLE_API_KEY env var).")
    ap.add_argument("--max-workers", type=int, default=5, help="Concurrent Stage 2 extraction calls.")
    ap.add_argument("--max-retries", type=int, default=3, help="Per-call retry limit before recording a failure.")
    ap.add_argument("--relationship-batch-size", type=int, default=40,
                     help="Unique raw relationships bundled into each Stage 4 LLM call.")
    ap.add_argument("--cache-path", default=None, help="LLM cache file (default: <out>/llm_cache.jsonl).")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    config = PipelineConfig(
        input_path=args.input,
        output_dir=args.out,
        include_suggested=args.include_suggested,
        strict_input=args.strict_input,
        extraction_model=args.extraction_model,
        normalization_model=args.normalization_model,
        api_key=args.api_key,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        relationship_batch_size=args.relationship_batch_size,
        cache_path=args.cache_path,
        log_level=args.log_level,
    )

    pipeline = Pipeline(config)
    summary = pipeline.process_file()

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
