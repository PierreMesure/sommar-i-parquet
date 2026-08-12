"""Read Whisper transcripts and turn them into timestamped analysis units."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any

import pyarrow.parquet as pq


WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class TranscriptUnit:
    """A small consecutive block used to detect semantic transitions."""

    unit_id: str
    sr_episode_id: int
    unit_index: int
    start_seconds: float
    end_seconds: float
    pause_before_seconds: float
    text: str
    word_count: int
    segment_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeTranscript:
    sr_episode_id: int
    source_title: str | None
    transcript_path: str
    transcript_engine: str | None
    transcript_model: str | None
    units: tuple[TranscriptUnit, ...]


def normalize_text(text: str) -> str:
    """Collapse ASR whitespace while retaining punctuation and casing."""
    return WHITESPACE.sub(" ", text).strip()


def load_episode_metadata(path: Path) -> dict[int, dict[str, Any]]:
    """Load canonical episode metadata keyed by SR episode ID."""
    return {
        int(row["sr_episode_id"]): row
        for row in pq.read_table(path).to_pylist()
    }


def _unit_id(episode_id: int, index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{episode_id}:u{index:04d}:{digest}"


def build_units(
    *,
    episode_id: int,
    segments: list[dict[str, Any]],
    target_words: int = 80,
    min_words: int = 24,
    preserve_pause_seconds: float = 1.5,
) -> tuple[TranscriptUnit, ...]:
    """Group Whisper segments into small units without crossing long pauses."""
    cleaned: list[dict[str, Any]] = []
    for segment in segments:
        text = normalize_text(str(segment.get("text") or ""))
        if not text:
            continue
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            continue
        cleaned.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "word_count": len(text.split()),
            }
        )

    grouped: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_words = 0
    for segment in cleaned:
        pause = (
            max(0.0, segment["start"] - current[-1]["end"])
            if current
            else 0.0
        )
        should_close = bool(
            current
            and current_words >= min_words
            and (current_words >= target_words or pause >= preserve_pause_seconds)
        )
        if should_close:
            grouped.append(current)
            current = []
            current_words = 0
        current.append(segment)
        current_words += int(segment["word_count"])
    if current:
        grouped.append(current)

    if len(grouped) > 1:
        last_words = sum(int(item["word_count"]) for item in grouped[-1])
        last_pause = max(0.0, grouped[-1][0]["start"] - grouped[-2][-1]["end"])
        if last_words < min_words and last_pause < preserve_pause_seconds:
            grouped[-2].extend(grouped.pop())

    units: list[TranscriptUnit] = []
    previous_end: float | None = None
    for index, group in enumerate(grouped):
        text = normalize_text(" ".join(str(item["text"]) for item in group))
        start = float(group[0]["start"])
        end = float(group[-1]["end"])
        units.append(
            TranscriptUnit(
                unit_id=_unit_id(episode_id, index, text),
                sr_episode_id=episode_id,
                unit_index=index,
                start_seconds=start,
                end_seconds=end,
                pause_before_seconds=max(0.0, start - previous_end) if previous_end is not None else 0.0,
                text=text,
                word_count=len(text.split()),
                segment_count=len(group),
            )
        )
        previous_end = end
    return tuple(units)


def load_transcripts(
    transcripts_dir: Path,
    *,
    unit_words: int = 80,
    preserve_pause_seconds: float = 1.5,
    limit: int | None = None,
) -> list[EpisodeTranscript]:
    """Load a stable snapshot of complete transcript JSON files."""
    paths = sorted(
        transcripts_dir.glob("*.json"),
        key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem),
    )
    if limit is not None:
        paths = paths[:limit]

    transcripts: list[EpisodeTranscript] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logging.warning("Skipping incomplete transcript %s: %s", path, error)
            continue
        metadata = payload.get("sommar_i_parquet") or {}
        try:
            episode_id = int(metadata.get("episode_id") or path.stem)
        except ValueError:
            logging.warning("Skipping transcript with no numeric episode ID: %s", path)
            continue
        units = build_units(
            episode_id=episode_id,
            segments=list(payload.get("segments") or []),
            target_words=unit_words,
            preserve_pause_seconds=preserve_pause_seconds,
        )
        if not units:
            logging.warning("Skipping transcript without usable segments: %s", path)
            continue
        transcripts.append(
            EpisodeTranscript(
                sr_episode_id=episode_id,
                source_title=metadata.get("source_title"),
                transcript_path=str(path),
                transcript_engine=metadata.get("engine"),
                transcript_model=metadata.get("model") or metadata.get("model_path"),
                units=units,
            )
        )
    return transcripts
