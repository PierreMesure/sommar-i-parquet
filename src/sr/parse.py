"""Turn raw SR API records into the stable MVP episode schema."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

STOCKHOLM = ZoneInfo("Europe/Stockholm")
DOTNET_DATE = re.compile(r"^/Date\((-?\d+)(?:[+-]\d{4})?\)/$")
TRAILING_YEAR = re.compile(r"\s+(\d{4})$")


def _parse_sr_datetime(value: str) -> datetime:
    """Parse both SR's .NET timestamp and its older ISO timestamp format."""
    match = DOTNET_DATE.match(value)
    if match:
        milliseconds = int(match.group(1))
        return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _speaker_from_title(title: str, year: int) -> str:
    """Remove SR's historical trailing year when it matches the episode date."""
    match = TRAILING_YEAR.search(title)
    if match and int(match.group(1)) == year:
        return title[: match.start()].strip()
    return title.strip()


def _program_type(month: int) -> str | None:
    if 6 <= month <= 8:
        return "Sommar"
    if month in {12, 1}:
        return "Vinter"
    return None


def _audio(episode: dict[str, Any]) -> dict[str, Any]:
    return episode.get("downloadpodfile") or episode.get("listenpodfile") or {}


def parse_episode(episode: dict[str, Any]) -> dict[str, Any]:
    """Parse one raw SR episode into a JSON-serializable dictionary."""
    published = _parse_sr_datetime(episode["publishdateutc"]).astimezone(STOCKHOLM)
    audio = _audio(episode)
    duration_seconds = audio.get("duration")

    return {
        "sr_episode_id": int(episode["id"]),
        "speaker": _speaker_from_title(episode.get("title", ""), published.year),
        "date": published.date().isoformat(),
        "year": published.year,
        "program_type": _program_type(published.month),
        "episode_url": episode.get("url"),
        "mp3_url": audio.get("url"),
        "length_minutes": (
            round(float(duration_seconds) / 60, 2)
            if duration_seconds is not None
            else None
        ),
        "short_summary": episode.get("description"),
    }


def parse_episodes(episodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse and deterministically order raw episodes."""
    parsed = [parse_episode(episode) for episode in episodes]
    return sorted(parsed, key=lambda episode: (episode["date"], episode["sr_episode_id"]))

