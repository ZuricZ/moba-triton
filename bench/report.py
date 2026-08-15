"""Renders benchmark rows to CSV and markdown.

CSV carries every field, for merging result sets across machines. Markdown
carries a readable subset, for looking at.

Both include skipped and errored rows with their reasons. A table that omits
what it could not measure looks complete and is not, which is the failure mode
the whole availability-gating design exists to prevent.
"""

import csv
import math
from dataclasses import asdict, fields
from pathlib import Path

from bench.suite import BenchRow

_COLUMNS = [
    "backend", "seq_len", "block_size", "top_k", "dtype",
    "latency_ms", "tflops", "sparsity", "speedup_vs_dense",
    "speedup_ceiling", "attainment", "status", "note",
]


def to_csv(rows: list[BenchRow], path: str | Path) -> None:
    """Writes every field of every row to CSV.

    Args:
        rows: Benchmark rows.
        path: Destination file; parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in fields(BenchRow)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _fmt(value) -> str:
    if isinstance(value, float):
        return "-" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def to_markdown(rows: list[BenchRow]) -> str:
    """Renders a human-readable results table.

    Args:
        rows: Benchmark rows.

    Returns:
        A markdown string, including a device header line.
    """
    if not rows:
        return "_No results._"

    d = rows[0]
    lines = [
        f"**Device:** {d.gpu_name} ({d.capability}), {d.total_memory_mb} MB | "
        f"torch {d.torch_version}, triton {d.triton_version}, "
        f"CUDA {d.cuda_version}",
        "",
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(_COLUMNS)) + "|",
    ]
    for row in rows:
        data = asdict(row)
        lines.append("| " + " | ".join(_fmt(data[c]) for c in _COLUMNS) + " |")
    return "\n".join(lines)
