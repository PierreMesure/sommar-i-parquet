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
        ("sr_audio_id", pa.int64()),
        ("source_title", pa.string()),
        ("speaker", pa.string()),
        ("date", pa.string()),
        ("year", pa.int32()),
        ("program_type", pa.string()),
        ("episode_url", pa.string()),
        ("mp3_url", pa.string()),
        ("length_seconds", pa.int64()),
        ("audio_file_size_bytes", pa.int64()),
        ("image_url", pa.string()),
        ("image_credit", pa.string()),
        ("short_summary", pa.string()),
        ("wikidata_id", pa.string()),
    ]
)

MUSIC_SCHEMA = pa.schema(
    [
        ("sr_episode_id", pa.int64()),
        ("track_number", pa.int32()),
        ("title", pa.string()),
        ("artist", pa.string()),
        ("composer", pa.string()),
        ("lyricist", pa.string()),
        ("album", pa.string()),
        ("record_label", pa.string()),
        ("is_theme_song", pa.bool_()),
    ]
)


def write_parquet(
    episodes: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    schema: pa.Schema = EPISODE_SCHEMA,
) -> Path:
    """Write dictionaries to a Zstandard-compressed Parquet file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(episodes), schema=schema)
    pq.write_table(table, path, compression="zstd")
    return path.resolve()
