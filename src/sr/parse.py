"""Turn raw SR API records into the stable MVP episode schema."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

STOCKHOLM = ZoneInfo("Europe/Stockholm")
MINIMUM_DURATION_MINUTES = 15
DOTNET_DATE = re.compile(r"^/Date\((-?\d+)(?:[+-]\d{4})?\)/$")
TRAILING_YEAR = re.compile(r"\s+(\d{4})$")
ORIGINAL_DATE = re.compile(
    r"\bfrån den (?P<day>\d{1,2}) "
    r"(?P<month>januari|februari|mars|april|maj|juni|juli|augusti|"
    r"september|oktober|november|december) "
    r"(?P<year>\d{4})\b",
    re.IGNORECASE,
)
SWEDISH_MONTHS = {
    "januari": 1,
    "februari": 2,
    "mars": 3,
    "april": 4,
    "maj": 5,
    "juni": 6,
    "juli": 7,
    "augusti": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "december": 12,
}
SPECIAL_TITLE_PATTERNS = (
    ("podcast_promo", re.compile(r"\bpoddtips\b", re.IGNORECASE)),
    ("trailer", re.compile(r"\btrailer\b", re.IGNORECASE)),
    ("host_announcement", re.compile(r"\bpresenteras\b", re.IGNORECASE)),
    ("anniversary", re.compile(r"\bsommar (?:50|60) år\b", re.IGNORECASE)),
    ("anniversary", re.compile(r"\bsommar i p1 50 år\b", re.IGNORECASE)),
    (
        "recap",
        re.compile(
            r"\bsommar (?:i )?backspegeln\b|\bsommar bonus\b",
            re.IGNORECASE,
        ),
    ),
    ("documentary", re.compile(r"\bbakom kulisserna\b", re.IGNORECASE)),
    ("question_show", re.compile(r"^fråga värden\b", re.IGNORECASE)),
    ("multi_guest_special", re.compile(r"\blive på poddfest\b", re.IGNORECASE)),
    ("standalone_music", re.compile(r"^pojken på månen\b", re.IGNORECASE)),
    (
        "alternate_language",
        re.compile(
            r"\benglish version\b|\bin english\b|"
            r"humanity has not yet failed|courage to speak out",
            re.IGNORECASE,
        ),
    ),
)


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
    """Normalize SR title conventions into a speaker name."""
    title = re.sub(r"^Sommar rekommenderar:\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^Sommar i P1 med\s+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\d{4}:\s*", "", title)
    title = re.sub(r"\s*\(Repris från \d{4}\)\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(
        r"\s*-\s*repris Vinter \d{4}\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\s*-\s*(?:På Vintergatan|Vinter(?:\s+\d{4})?)\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = title.rstrip(" -")

    match = TRAILING_YEAR.search(title)
    if match and int(match.group(1)) == year:
        return title[: match.start()].strip()
    return title.strip()


def _program_type(month: int, title: str, summary: str) -> str | None:
    if 6 <= month <= 9:
        return "Sommar"
    if month in {12, 1}:
        return "Vinter"

    text = f"{title} {summary}"
    if re.search(r"\bvinter(?:program|värd|prat| i p1)\b|på vintergatan", text, re.IGNORECASE):
        return "Vinter"
    if re.search(r"\bsommar(?:program|värd|prat| i p1)\b", text, re.IGNORECASE):
        return "Sommar"
    return None


def _audio(episode: dict[str, Any]) -> dict[str, Any]:
    return episode.get("downloadpodfile") or episode.get("listenpodfile") or {}


def _effective_date(episode: dict[str, Any], published: datetime) -> datetime:
    """Recover an original date when SR published an archive item out of season."""
    if published.month in {1, 6, 7, 8, 9, 12}:
        return published
    if not episode.get("title", "").casefold().startswith("sommar i p1 med "):
        return published

    match = ORIGINAL_DATE.search(episode.get("description", ""))
    if not match:
        return published
    return published.replace(
        year=int(match.group("year")),
        month=SWEDISH_MONTHS[match.group("month").casefold()],
        day=int(match.group("day")),
    )


def parse_episode(episode: dict[str, Any]) -> dict[str, Any]:
    """Parse one raw SR episode into a JSON-serializable dictionary."""
    published = _parse_sr_datetime(episode["publishdateutc"]).astimezone(STOCKHOLM)
    published = _effective_date(episode, published)
    audio = _audio(episode)
    duration_seconds = audio.get("duration")
    title = episode.get("title", "")
    summary = episode.get("description") or ""

    return {
        "sr_episode_id": int(episode["id"]),
        "sr_audio_id": audio.get("id"),
        "source_title": title,
        "speaker": _speaker_from_title(title, published.year),
        "date": published.date().isoformat(),
        "year": published.year,
        "program_type": _program_type(published.month, title, summary),
        "episode_url": episode.get("url"),
        "mp3_url": audio.get("url"),
        "length_seconds": int(duration_seconds) if duration_seconds is not None else None,
        "audio_file_size_bytes": audio.get("filesizeinbytes"),
        "image_url": episode.get("imageurltemplate") or episode.get("imageurl"),
        "image_credit": episode.get("photographer"),
        "short_summary": episode.get("description"),
    }


def exclusion_reason(episode: dict[str, Any]) -> str | None:
    """Explain why a parsed record is not a host-led Sommar/Vinter episode."""
    title = episode["speaker"]
    for reason, pattern in SPECIAL_TITLE_PATTERNS:
        if pattern.search(title):
            return reason

    duration = episode["length_seconds"]
    if duration is None or episode["mp3_url"] is None:
        return "missing_audio"
    if duration < MINIMUM_DURATION_MINUTES * 60:
        return "too_short"
    if episode["program_type"] is None:
        return "outside_season"
    return None


def parse_episodes(
    episodes: Iterable[dict[str, Any]],
    *,
    include_specials: bool = False,
) -> list[dict[str, Any]]:
    """Parse, filter, and deterministically order raw episodes."""
    parsed = [parse_episode(episode) for episode in episodes]
    if not include_specials:
        reasons = Counter(
            reason
            for episode in parsed
            if (reason := exclusion_reason(episode)) is not None
        )
        parsed = [episode for episode in parsed if exclusion_reason(episode) is None]
        if reasons:
            logging.info(
                "Excluded %d non-host records: %s",
                reasons.total(),
                ", ".join(f"{reason}={count}" for reason, count in sorted(reasons.items())),
            )
    return sorted(parsed, key=lambda episode: (episode["date"], episode["sr_episode_id"]))


def _music_text(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _is_theme_song(title: str) -> bool:
    """Recognize SR's Sommar and Vinter theme-title variants."""
    tokens = [
        token
        for token in re.findall(r"[^\W_]+", title.casefold())
        if token != "signatur"
    ]
    return (
        len(tokens) >= 2 and set(tokens) == {"sommar"}
    ) or tokens == ["vintergatan"]


def _music_metadata_score(song: dict[str, Any]) -> int:
    return sum(
        bool((song.get(field) or "").strip())
        for field in ("title", "artist", "composer", "lyricist", "albumname", "recordlabel")
    )


def parse_music_playlists(
    playlists: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize official SR episode playlists into one row per song play."""
    rows: list[dict[str, Any]] = []
    for playlist in playlists:
        episode_id = int(playlist["sr_episode_id"])

        def song_start(song: dict[str, Any]) -> datetime:
            value = song.get("starttimeutc")
            return _parse_sr_datetime(value) if value else datetime.max.replace(tzinfo=UTC)

        unique_songs: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = {}
        for song in playlist.get("songs", []):
            key = (
                _music_text(song.get("title")),
                _music_text(song.get("artist")),
                song.get("starttimeutc"),
                song.get("stoptimeutc"),
            )
            existing = unique_songs.get(key)
            if existing is None or _music_metadata_score(song) > _music_metadata_score(existing):
                unique_songs[key] = song

        songs = sorted(unique_songs.values(), key=song_start)
        for track_number, song in enumerate(songs, start=1):
            title = (song.get("title") or "").strip()
            rows.append(
                {
                    "sr_episode_id": episode_id,
                    "track_number": track_number,
                    "title": title or None,
                    "artist": (song.get("artist") or "").strip() or None,
                    "composer": (song.get("composer") or "").strip() or None,
                    "lyricist": (song.get("lyricist") or "").strip() or None,
                    "album": (song.get("albumname") or "").strip() or None,
                    "record_label": (song.get("recordlabel") or "").strip() or None,
                    "is_theme_song": _is_theme_song(title),
                }
            )
    return sorted(rows, key=lambda row: (row["sr_episode_id"], row["track_number"]))
