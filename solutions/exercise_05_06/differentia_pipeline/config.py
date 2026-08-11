"""
config.py
---------
Single source of configuration for the whole pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path = Path("differentia_output")
    include_suggested: bool = False  # see io_utils.read_input_records docstring
    strict_input: bool = False       # False (default): skip bad input rows and record them in
                                      # 04_failures.jsonl. True: abort the whole read on the first one.

    extraction_model: str = "gemini-2.5-flash"
    normalization_model: str = "gemini-2.5-flash"
    api_key: str | None = None

    max_workers: int = 5           # concurrent Stage 2 extraction calls
    max_retries: int = 3           # per LLM call, before it's recorded as a failure
    relationship_batch_size: int = 40  # unique raw relationships per Stage 4 LLM call

    cache_path: Path | None = None  # defaults to output_dir/llm_cache.jsonl
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.output_dir = Path(self.output_dir)
        if self.cache_path is None:
            self.cache_path = self.output_dir / "llm_cache.jsonl"
        else:
            self.cache_path = Path(self.cache_path)
