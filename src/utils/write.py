"""Write parsed episode records to Parquet."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

EPISODE_SCHEMA = pa.schema(
    [
        ("sr_episode_id", pa.int64()),
        ("speaker", pa.string()),
        ("date", pa.string()),
        ("year", pa.int32()),
        ("program_type", pa.string()),
        ("episode_url", pa.string()),
        ("mp3_url", pa.string()),
        ("length_minutes", pa.float64()),
        ("short_summary", pa.string()),
    ]
)


def write_parquet(
    episodes: Sequence[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write episode dictionaries to a Zstandard-compressed Parquet file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(episodes), schema=EPISODE_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return path.resolve()

