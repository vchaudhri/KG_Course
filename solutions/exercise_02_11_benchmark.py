from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    database: str
    rows: int
    average_ms: float

    