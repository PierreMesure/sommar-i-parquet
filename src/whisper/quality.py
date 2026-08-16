"""High-confidence transcript quality checks and reversible cleanup."""

from __future__ import annotations

from collections import Counter
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


WORD = re.compile(r"[a-zåäö]+", re.IGNORECASE)
CHARACTER_LOOP = re.compile(r"(.)\1{19,}", re.IGNORECASE)
PUNCTUATION = re.compile(r"[^a-zåäö\s]", re.IGNORECASE)

# These deliberately target SR production framing, not ordinary references to
# Sommar, Vinter, music, producers, websites, or the app in the programme body.
OPENING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "programme_identification",
        re.compile(
            r"\bdet här (?:programmet )?(?:är|var) "
            r"(?:(?:ett )?poddradioprogram från|podd(?:radio)?versionen av)?\s*"
            r"(?:sommar|vinter) i p\s*1\b",
            re.IGNORECASE,
        ),
    ),
    (
        "copyright_music_notice",
        re.compile(
            r"\b(?:av upphovsrättsliga skäl .*?musiken .*?(?:förkortad|nedkortad)"
            r"|musiken .*?(?:förkortad|nedkortad).*?upphovsrättsliga skäl)\b"
            r"|^\s*musiken (?:är|har) (?:förkortad|nedkortad)\.?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "programme_announcement",
        re.compile(r"\bdags för (?:sommar|vinter)\s*i p\s*1\b", re.IGNORECASE),
    ),
    (
        "sr_play_promotion",
        re.compile(
            r"\balla (?:sommar|vinter)program finns .*?"
            r"(?:sveriges radio play|vår app)\b",
            re.IGNORECASE,
        ),
    ),
)

CLOSING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "programme_signoff",
        re.compile(
            r"^\s*(?:det var|glad) (?:sommar|vinter)(?:en)? i p\s*1\b",
            re.IGNORECASE,
        ),
    ),
    (
        "producer_credit",
        re.compile(r"^\s*producent(?:en)?\s+[a-zåäö]", re.IGNORECASE),
    ),
    (
        "sr_play_promotion",
        re.compile(
            r"\b(?:lyssna|hör|hittar|finns|sök(?: på)?).*?"
            r"(?:sveriges radio(?: play)?|sverigesradio\.se|vår app|appen|på webben)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "music_list_promotion",
        re.compile(
            r"\b(?:hela )?(?:musiklistan|lista över musiken|listan (?:med|över) musiken)"
            r".*?(?:hemsidan|sveriges\s*radio(?:s)?|sverigesradio)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "legacy_web_promotion",
        re.compile(
            r"\b(?:ladda ner .*?(?:webbradioapparat|sr\.se)"
            r"|lyssna .*?musiken .*?(?:webben|30 dagar)"
            r"|köpa .*?program utan musik"
            r"|sms[- ]?påminnelser?"
            r"|sommars hemsida .*?musik"
            r"|gå in på sr\.se)\b",
            re.IGNORECASE,
        ),
    ),
)

# A production credit marks the end of the programme proper.  Recent SR podcast
# files append a short recommendation for another Sommar/Vinter programme after
# this credit, so this is intentionally a structural boundary rather than a
# keyword-based recommendation detector.
PRODUCTION_CREDIT_PATTERN = re.compile(
    r"^\s*(?:(?:och|samt)\s+)?(?:producent(?:en)?|redaktör(?:en)?|tekniker(?:n)?|slutmix)\b"
    r"(?:\s*(?:,|:)|\s+(?:var|är))?\s+[A-ZÅÄÖ]",
    re.IGNORECASE,
)

# Fallback for files where the ASR missed the production credit.  These terms
# are only meaningful very near the end, and are used only if no credit
# boundary can be found.
RECOMMENDED_EPISODE_CUE_PATTERN = re.compile(
    r"(?:\b(?:sommar|vinter)pratade\b"
    r"|\b(?:förra årets|en annan|tidigare)\b.*?\b(?:sommar|vinter)värd(?:en)?\b"
    r"|\b(?:var|blev)\s+(?:sommar|vinter)pratare\b)",
    re.IGNORECASE,
)


def artifact_reasons(text: str) -> list[str]:
    """Return only high-confidence ASR repetition-artifact reasons."""
    words = WORD.findall(text.casefold())
    reasons: list[str] = []
    if len(words) >= 20:
        dominant = max(Counter(words).values()) / len(words)
        if dominant >= 0.7:
            reasons.append("dominant_token_loop")
    if len(words) <= 3 and len(text) >= 120 and CHARACTER_LOOP.search(text):
        reasons.append("character_noise_loop")
    if not words and len(text) >= 120 and len(PUNCTUATION.findall(text)) >= 100:
        reasons.append("punctuation_noise_loop")
    return reasons


def transcript_artifacts(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Return each corrupted segment together with its detection reasons."""
    matches: list[dict[str, Any]] = []
    for index, segment in enumerate(transcript.get("segments") or []):
        text = str(segment.get("text") or "")
        reasons = artifact_reasons(text)
        if reasons:
            matches.append(
                {
                    "segment_index": index,
                    "start_seconds": segment.get("start"),
                    "end_seconds": segment.get("end"),
                    "text": text,
                    "reasons": reasons,
                }
            )
    return matches


def _matching_reasons(
    text: str,
    patterns: Iterable[tuple[str, re.Pattern[str]]],
) -> list[str]:
    return [reason for reason, pattern in patterns if pattern.search(text)]


def transcript_boilerplate(
    transcript: dict[str, Any],
    *,
    opening_seconds: float = 45.0,
    closing_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    """Find position-sensitive SR framing without matching programme content."""
    segments = list(transcript.get("segments") or [])
    duration = max((float(segment.get("end", 0) or 0) for segment in segments), default=0.0)
    matches: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "")
        start = float(segment.get("start", 0) or 0)
        end = float(segment.get("end", start) or start)
        reasons: list[str] = []
        if start <= opening_seconds:
            reasons.extend(_matching_reasons(text, OPENING_PATTERNS))
        if duration and end >= duration - closing_seconds:
            reasons.extend(_matching_reasons(text, CLOSING_PATTERNS))
        if reasons:
            matches.append(
                {
                    "segment_index": index,
                    "start_seconds": start,
                    "end_seconds": end,
                    "text": text,
                    "reasons": list(dict.fromkeys(reasons)),
                }
            )
    return matches


def _strip_segments(
    transcript: dict[str, Any],
    matches: Iterable[dict[str, Any]],
    *,
    metadata_key: str,
    store_raw_segments: bool = False,
) -> list[dict[str, Any]]:
    reasons_by_index = {
        int(item["segment_index"]): list(item["reasons"])
        for item in matches
    }
    retained: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for index, segment in enumerate(transcript.get("segments") or []):
        reasons = reasons_by_index.get(index)
        if reasons:
            removed.append({"segment": segment, "reasons": reasons})
        else:
            retained.append(segment)
    transcript["segments"] = retained
    if removed:
        metadata = transcript.setdefault("sommar_i_parquet", {})
        stored = [item["segment"] for item in removed] if store_raw_segments else removed
        metadata.setdefault(metadata_key, []).extend(stored)
        if store_raw_segments:
            metadata.setdefault("removed_introduction_segment_audit", []).extend(
                {
                    "start_seconds": item["segment"].get("start"),
                    "end_seconds": item["segment"].get("end"),
                    "text": item["segment"].get("text"),
                    "reasons": item["reasons"],
                }
                for item in removed
            )
    return removed


def strip_boilerplate(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Remove detected SR framing and retain exact evidence in metadata."""
    return _strip_segments(
        transcript,
        transcript_boilerplate(transcript),
        metadata_key="removed_introduction_segments",
        store_raw_segments=True,
    )


def _production_credit_boundary(
    transcript: dict[str, Any], *, search_seconds: float = 180.0
) -> tuple[float, float, str, str] | None:
    """Find the final production-credit boundary in live or prior-cleaned data."""
    segments = list(transcript.get("segments") or [])
    duration = max((float(item.get("end", 0) or 0) for item in segments), default=0.0)
    candidates: list[tuple[float, float, str, str]] = []
    for segment in segments:
        start = float(segment.get("start", 0) or 0)
        if duration and start < duration - search_seconds:
            continue
        text = str(segment.get("text") or "")
        if PRODUCTION_CREDIT_PATTERN.search(text):
            candidates.append((start, float(segment.get("end", start) or start), text, "segment"))

    # Existing transcripts may already have had the credit removed by the
    # boilerplate pass.  Its audit record remains an equally reliable boundary.
    metadata = transcript.get("sommar_i_parquet") or {}
    for item in metadata.get("removed_introduction_segment_audit", []):
        if "producer_credit" not in item.get("reasons", []):
            continue
        start = float(item.get("start_seconds", 0) or 0)
        end = float(item.get("end_seconds", start) or start)
        candidates.append((start, end, str(item.get("text") or ""), "boilerplate_audit"))
    return max(candidates, default=None, key=lambda candidate: candidate[0])


def _recommended_episode_cue_boundary(
    transcript: dict[str, Any], *, search_seconds: float = 180.0
) -> tuple[float, float, str, str] | None:
    """Find the first explicit cross-programme cue in the final minutes."""
    segments = list(transcript.get("segments") or [])
    duration = max((float(item.get("end", 0) or 0) for item in segments), default=0.0)
    for segment in segments:
        start = float(segment.get("start", 0) or 0)
        if duration and start < duration - search_seconds:
            continue
        text = str(segment.get("text") or "")
        if RECOMMENDED_EPISODE_CUE_PATTERN.search(text):
            end = float(segment.get("end", start) or start)
            return start, end, text, "trailing_recommendation_cue"
    return None


def strip_recommended_episode_extract(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Remove the production credit and every trailing recommendation segment.

    SR's appended cross-promotion is separated from the programme by its final
    production credit.  Keeping this operation independent from text matching
    makes it robust to the many ways the recommendation is introduced.
    """
    boundary = _production_credit_boundary(transcript)
    if boundary is None:
        boundary = _recommended_episode_cue_boundary(transcript)
    if boundary is None:
        return []
    start, end, credit_text, source = boundary
    segments = list(transcript.get("segments") or [])
    removed = [segment for segment in segments if float(segment.get("start", 0) or 0) >= start]
    if not removed:
        return []

    transcript["segments"] = [
        segment for segment in segments if float(segment.get("start", 0) or 0) < start
    ]
    words = list(transcript.get("word_segments") or [])
    removed_words = [word for word in words if float(word.get("start", 0) or 0) >= start]
    if "word_segments" in transcript:
        transcript["word_segments"] = [
            word for word in words if float(word.get("start", 0) or 0) < start
        ]

    metadata = transcript.setdefault("sommar_i_parquet", {})
    metadata.setdefault("removed_recommended_episode_extract_segments", []).extend(removed)
    metadata.setdefault("removed_recommended_episode_extract_audit", []).append(
        {
            "boundary_start_seconds": start,
            "boundary_end_seconds": end,
            "production_credit_text": credit_text,
            "boundary_source": source,
            "removed_segment_count": len(removed),
            "removed_word_segment_count": len(removed_words),
        }
    )
    return removed


def strip_introductions(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Backward-compatible name for position-sensitive boilerplate cleanup."""
    return strip_boilerplate(transcript)


def strip_artifacts(
    transcript: dict[str, Any], artifacts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Remove detected ASR artifacts while retaining text and reasons."""
    return _strip_segments(
        transcript,
        artifacts,
        metadata_key="removed_artifact_segments",
    )


def quarantine_transcript(path: Path, artifacts: list[dict[str, Any]]) -> Path:
    """Move a faulty transcript aside without overwriting a prior attempt."""
    del artifacts  # The caller stores the reasons in the transcript itself.
    destination_dir = path.parent / "faulty"
    destination_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = destination_dir / f"{path.stem}-{stamp}{path.suffix}"
    shutil.move(path, destination)
    return destination
