"""
cache.py
--------
A simple content-addressed cache so identical (model, stage, input-text)
combinations never trigger a repeated LLM call, even across separate runs.

Backed by an append-only JSONL file: loaded fully into memory at startup,
appended to (and flushed) on every write. Append-only means a crash mid-run
never corrupts previously-cached entries -- worst case you lose the entry
currently being written, never the ones before it.

Thread-safe: Stage 2 extraction runs many calls concurrently via a thread
pool, and they all share one CacheStore instance.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


class CacheStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mem: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        self._fh = open(self.path, "a", encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec["value"]
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a truncated last line from a prior crash

    @staticmethod
    def make_key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._mem.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._mem[key] = value
            self._fh.write(json.dumps({"key": key, "value": value}) + "\n")
            self._fh.flush()

    def __len__(self) -> int:
        return len(self._mem)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "CacheStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
