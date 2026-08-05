"""Write parsed episode records to Parquet."""

from __future__ import annotations

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
        ("episode_speakers", pa.list_(pa.string())),
        ("speaker_ages", pa.list_(pa.int32())),
    ]
)

SPEAKER_METADATA_FIELDS = [
    ("wikidata_label", pa.string()),
    ("wikidata_description", pa.string()),
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

SPEAKER_SCHEMA = pa.schema(
    [
        ("wikidata_id", pa.string()),
        ("speaker", pa.string()),
        ("sr_names", pa.list_(pa.string())),
        ("episode_count", pa.int32()),
        ("episode_ids", pa.list_(pa.int64())),
        ("ages_at_episodes", pa.list_(pa.int32())),
        *SPEAKER_METADATA_FIELDS,
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
        *SPEAKER_METADATA_FIELDS,
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
    """Write normalized, compact data for the static frontend."""
    speakers_by_id: dict[str, dict[str, Any]] = {}
    for speaker in speakers:
        qid = str(speaker["wikidata_id"])
        record: dict[str, Any] = {
            "name": speaker["speaker"],
            "count": int(speaker["episode_count"]),
            "ages": [
                age for age in speaker.get("ages_at_episodes", [])
                if age is not None
            ],
        }
        optional_values = {
            "aliases": [
                name
                for name in speaker.get("sr_names", [])
                if name != speaker["speaker"]
            ],
            "label": speaker.get("wikidata_label"),
            "description": speaker.get("wikidata_description"),
            "wiki": speaker.get("wikipedia_url"),
            "gender": speaker.get("gender"),
            "born": speaker.get("birth_date"),
            "died": speaker.get("death_date"),
            "citizenships": speaker.get("citizenships", []),
            "occupations": speaker.get("occupations", []),
        }
        record.update(
            {key: value for key, value in optional_values.items() if value}
        )
        speakers_by_id[qid] = record

    records = []
    for episode in episodes:
        episode_id = int(episode["sr_episode_id"])
        speaker_ids = list(episode.get("episode_speakers") or [])
        missing_speakers = [qid for qid in speaker_ids if qid not in speakers_by_id]
        if missing_speakers:
            raise ValueError(
                f"Episode {episode_id} references missing speakers: "
                f"{', '.join(missing_speakers)}"
            )
        names = [speakers_by_id[qid]["name"] for qid in speaker_ids]
        image_url = episode.get("image_url") or ""
        records.append(
            {
                "id": episode_id,
                "date": episode["date"],
                "type": episode["program_type"],
                "minutes": (int(episode["length_seconds"]) + 30) // 60,
                "image": image_url.removeprefix(SR_IMAGE_URL_PREFIX) or None,
                "description": episode["short_summary"],
                "speakers": speaker_ids,
                "ages": list(episode.get("speaker_ages") or []),
                "initials": _speaker_initials(names),
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"speakers": speakers_by_id, "episodes": records},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
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
