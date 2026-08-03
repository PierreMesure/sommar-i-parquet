"""Write parsed episode records to Parquet."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

EPISODE_SCHEMA = pa.schema(
    [
        ("sr_episode_id", pa.int64()),
        ("sr_audio_id", pa.int64()),
        ("source_title", pa.string()),
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
    ]
)

SPEAKER_SCHEMA = pa.schema(
    [
        ("sr_episode_id", pa.int64()),
        ("speaker_index", pa.int32()),
        ("speaker_appearance_id", pa.string()),
        ("speaker", pa.string()),
        ("wikidata_id", pa.string()),
    ]
)

SPEAKER_APPEARANCE_SCHEMA = pa.schema(
    [
        *EPISODE_SCHEMA,
        ("speaker_index", pa.int32()),
        ("speaker_appearance_id", pa.string()),
        ("speaker", pa.string()),
        ("wikidata_id", pa.string()),
    ]
)

SPEAKER_METADATA_SCHEMA = pa.schema(
    [
        ("wikidata_id", pa.string()),
        ("wikipedia_url", pa.string()),
        ("gender", pa.string()),
        ("gender_id", pa.string()),
        ("birth_date", pa.string()),
        ("death_date", pa.string()),
        ("citizenships", pa.list_(pa.string())),
        ("citizenship_ids", pa.list_(pa.string())),
        ("occupations", pa.list_(pa.string())),
        ("occupation_ids", pa.list_(pa.string())),
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

SR_IMAGE_URL_PREFIX = "https://static-cdn.sr.se"


def _speaker_initials(speakers: Sequence[str]) -> str:
    initials = [
        "".join(part[0] for part in speaker.strip().split() if part)[:2].upper()
        for speaker in speakers
    ]
    return (
        " · ".join(initial[0] for initial in initials)
        if len(initials) > 2
        else " · ".join(initials)
    )


def write_frontend_json(
    episodes: Sequence[dict[str, Any]],
    speakers: Sequence[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write the compact, static-frontend representation of the archive."""
    speakers_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for speaker in speakers:
        speakers_by_episode[int(speaker["sr_episode_id"])].append(speaker)
    for appearances in speakers_by_episode.values():
        appearances.sort(key=lambda appearance: int(appearance["speaker_index"]))

    appearance_counts = Counter(
        appearance["speaker"]
        for appearances in speakers_by_episode.values()
        for appearance in appearances
    )
    records = []
    for episode in episodes:
        episode_id = int(episode["sr_episode_id"])
        names = [appearance["speaker"] for appearance in speakers_by_episode[episode_id]]
        returning_names = [name for name in names if appearance_counts[name] > 1]
        image_url = episode.get("image_url") or ""
        records.append(
            {
                "id": episode_id,
                "date": episode["date"],
                "type": episode["program_type"],
                "minutes": (int(episode["length_seconds"]) + 30) // 60,
                "image": image_url.removeprefix(SR_IMAGE_URL_PREFIX) or None,
                "description": episode["short_summary"],
                "speakers": names,
                "initials": _speaker_initials(names),
                "returning": bool(returning_names),
                "group": returning_names[0] if returning_names else None,
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"episodes": records}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return path.resolve()


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
