"""LLM chapter planning and local timestamp resolution for transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Iterable

from pydantic import BaseModel, Field


WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


class ChapterBoundary(BaseModel):
    """A quoted anchor immediately before one coherent chapter transition."""

    after_words: str = Field(
        description=(
            "6–20 consecutive words copied verbatim from immediately before "
            "the chapter transition."
        ),
        min_length=10,
        max_length=300,
    )


class ChapterPlan(BaseModel):
    """First-pass output: only boundaries, never topic labels."""

    boundaries: list[ChapterBoundary] = Field(
        description="Chronological chapter-transition anchors. Omit intro/outro boundaries.",
        max_length=14,
    )


class ChapterLabels(BaseModel):
    """Second-pass output for one chapter."""

    title: str = Field(description="A short Swedish chapter title", min_length=2, max_length=80)
    keywords: list[str] = Field(
        description="One to six concise Swedish topic keywords or keyphrases.",
        min_length=1,
        max_length=6,
    )
    summary: str = Field(
        description="A one-sentence Swedish summary of the chapter.",
        min_length=10,
        max_length=400,
    )


@dataclass(frozen=True)
class ResolvedBoundary:
    """A model-suggested anchor snapped to the nearest segment end."""

    after_words: str
    segment_index: int
    score: float


def words(text: str) -> list[str]:
    """Normalise Swedish transcript text for anchor matching."""
    return [word.casefold() for word in WORD.findall(text)]


def transcript_text(segments: Iterable[dict[str, Any]]) -> str:
    return " ".join(str(segment.get("text") or "").strip() for segment in segments).strip()


def resolve_anchor(
    anchor: str,
    segments: list[dict[str, Any]],
) -> ResolvedBoundary:
    """Find the segment end best matching a verbatim-ish boundary phrase.

    The model sees plain transcript text, while timestamps live on Whisper
    segments. We compare the anchor with word windows ending at every segment
    and return both a score and the snap location for audit.
    """
    target = words(anchor)
    if len(target) < 2:
        raise ValueError("Boundary anchors require at least two words.")
    all_words: list[str] = []
    candidate_ends: list[tuple[int, int]] = []
    for index, segment in enumerate(segments):
        all_words.extend(words(str(segment.get("text") or "")))
        if all_words:
            candidate_ends.append((index, len(all_words)))
    if not candidate_ends:
        raise ValueError("Transcript has no usable words.")

    best: tuple[float, int] | None = None
    # Permit a few missing/extra words and ASR punctuation differences while
    # still requiring a phrase near the end of a timestamped segment.
    for segment_index, end in candidate_ends:
        for width in range(max(2, len(target) - 4), len(target) + 5):
            candidate = all_words[max(0, end - width):end]
            if not candidate:
                continue
            sequence = SequenceMatcher(a=target, b=candidate).ratio()
            overlap = len(set(target) & set(candidate)) / max(len(set(target)), 1)
            score = 0.7 * sequence + 0.3 * overlap
            if best is None or score > best[0]:
                best = (score, segment_index)
    assert best is not None
    return ResolvedBoundary(after_words=anchor, score=best[0], segment_index=best[1])


def resolve_boundaries(
    boundaries: Iterable[ChapterBoundary],
    segments: list[dict[str, Any]],
    *,
    minimum_score: float = 0.64,
    min_segments_per_chapter: int = 3,
) -> list[ResolvedBoundary]:
    """Resolve, de-duplicate, and reject weak/too-close boundary anchors."""
    resolved = [resolve_anchor(boundary.after_words, segments) for boundary in boundaries]
    resolved.sort(key=lambda item: item.segment_index)
    accepted: list[ResolvedBoundary] = []
    for boundary in resolved:
        if boundary.score < minimum_score:
            continue
        if boundary.segment_index < min_segments_per_chapter:
            continue
        if accepted and boundary.segment_index - accepted[-1].segment_index < min_segments_per_chapter:
            # Prefer the better anchor when two suggestions snap to one place.
            if boundary.score > accepted[-1].score:
                accepted[-1] = boundary
            continue
        accepted.append(boundary)
    return accepted


def chapter_segment_ranges(
    segment_count: int, boundaries: Iterable[ResolvedBoundary]
) -> list[tuple[int, int]]:
    """Return inclusive segment index ranges for all chapters."""
    ends = [boundary.segment_index for boundary in boundaries]
    ranges: list[tuple[int, int]] = []
    start = 0
    for end in ends:
        if start <= end:
            ranges.append((start, end))
        start = end + 1
    if start < segment_count:
        ranges.append((start, segment_count - 1))
    return ranges
