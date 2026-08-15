"""Transcript quality checks and reversible cleanup."""

from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORD = re.compile(r"[a-zåäö]+", re.IGNORECASE)
CHARACTER_LOOP = re.compile(r"(.)\1{19,}", re.IGNORECASE)
PUNCTUATION = re.compile(r"[^a-zåäö\s]", re.IGNORECASE)
INTRODUCTION_PATTERNS = (
    re.compile(r"\bdet här programmet (?:är|var) sommar i p1\b", re.IGNORECASE),
    re.compile(r"\bdet här är (?:poddversionen|ett poddradioprogram) (?:av|från) sommar i p1\b", re.IGNORECASE),
    re.compile(r"\b(?:av upphovsrättsliga skäl|musiken (?:har )?(?:är )?förkortats? av upphovsrättsliga skäl)\b", re.IGNORECASE),
    re.compile(r"\bdags för sommar i p1\b", re.IGNORECASE),
)


def artifact_reasons(text: str) -> list[str]:
    """Return only high-confidence ASR repetition artifact reasons."""
    words = WORD.findall(text.lower())
    reasons = []
    if len(words) >= 20:
        dominant = max(words.count(word) for word in set(words)) / len(words)
        if dominant >= 0.7:
            reasons.append("dominant_token_loop")
    if len(words) <= 3 and len(text) >= 120 and CHARACTER_LOOP.search(text):
        reasons.append("character_noise_loop")
    if not words and len(text) >= 120 and len(PUNCTUATION.findall(text)) >= 100:
        reasons.append("punctuation_noise_loop")
    return reasons


def transcript_artifacts(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Return each corrupted segment together with its detection reasons."""
    return [
        {"segment_index": index, "reasons": reasons}
        for index, segment in enumerate(transcript.get("segments") or [])
        if (reasons := artifact_reasons(str(segment.get("text") or "")))
    ]


def strip_introductions(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Remove known SR boilerplate while retaining the removed raw segments for audit."""
    retained, removed = [], []
    for segment in transcript.get("segments") or []:
        text = str(segment.get("text") or "")
        is_opening = float(segment.get("start", 0) or 0) <= 30
        if is_opening and any(pattern.search(text) for pattern in INTRODUCTION_PATTERNS):
            removed.append(segment)
        else:
            retained.append(segment)
    transcript["segments"] = retained
    if removed:
        metadata = transcript.setdefault("sommar_i_parquet", {})
        metadata.setdefault("removed_introduction_segments", []).extend(removed)
    return removed


def strip_artifacts(transcript: dict[str, Any], artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove detected artifacts while retaining their text and detection reasons."""
    artifact_reasons_by_index = {item["segment_index"]: item["reasons"] for item in artifacts}
    retained, removed = [], []
    for index, segment in enumerate(transcript.get("segments") or []):
        reasons = artifact_reasons_by_index.get(index)
        if reasons:
            removed.append({"segment": segment, "reasons": reasons})
        else:
            retained.append(segment)
    transcript["segments"] = retained
    if removed:
        transcript.setdefault("sommar_i_parquet", {})["removed_artifact_segments"] = removed
    return removed


def quarantine_transcript(path: Path, artifacts: list[dict[str, Any]]) -> Path:
    """Move a faulty transcript aside without overwriting a prior attempt."""
    destination_dir = path.parent / "faulty"
    destination_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = destination_dir / f"{path.stem}-{stamp}{path.suffix}"
    shutil.move(path, destination)
    return destination
