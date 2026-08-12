"""Pause-aware semantic change-point detection for transcript chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np

from src.nlp.transcripts import EpisodeTranscript, TranscriptUnit, normalize_text


@dataclass(frozen=True)
class Boundary:
    """Diagnostics for a possible cut immediately before one analysis unit."""

    sr_episode_id: int
    boundary_index: int
    timestamp_seconds: float
    pause_seconds: float
    semantic_distance: float
    semantic_score: float
    pause_score: float
    boundary_score: float
    likely_music_break: bool
    selected: bool = False
    selection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def score_boundaries(
    transcript: EpisodeTranscript,
    unit_embeddings: np.ndarray,
    *,
    window_units: int = 3,
    semantic_weight: float = 0.82,
    strong_pause_seconds: float = 5.0,
) -> list[Boundary]:
    """Compare rolling context on each side of every possible boundary."""
    units = transcript.units
    if len(units) != len(unit_embeddings):
        raise ValueError("Every transcript unit needs exactly one embedding.")
    if len(units) < 2:
        return []

    embeddings = _normalise_rows(np.asarray(unit_embeddings, dtype=np.float32))
    distances: list[float] = []
    for index in range(1, len(units)):
        left = embeddings[max(0, index - window_units):index].mean(axis=0)
        right = embeddings[index:min(len(units), index + window_units)].mean(axis=0)
        left /= max(float(np.linalg.norm(left)), 1e-12)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        distances.append(max(0.0, 1.0 - float(np.dot(left, right))))

    values = np.asarray(distances, dtype=np.float32)
    low = float(np.quantile(values, 0.25))
    high = float(np.quantile(values, 0.90))
    scale = max(high - low, 1e-6)
    semantic_scores = np.clip((values - low) / scale, 0.0, 1.5)

    boundaries: list[Boundary] = []
    for offset, index in enumerate(range(1, len(units))):
        pause = max(0.0, units[index].pause_before_seconds)
        pause_score = min(math.log1p(pause) / math.log(4.0), 1.5)
        score = (
            semantic_weight * float(semantic_scores[offset])
            + (1.0 - semantic_weight) * pause_score
        )
        likely_music_break = pause >= strong_pause_seconds
        if likely_music_break:
            score = max(score, 1.8)
        boundaries.append(
            Boundary(
                sr_episode_id=transcript.sr_episode_id,
                boundary_index=index,
                timestamp_seconds=units[index].start_seconds,
                pause_seconds=pause,
                semantic_distance=float(values[offset]),
                semantic_score=float(semantic_scores[offset]),
                pause_score=pause_score,
                boundary_score=score,
                likely_music_break=likely_music_break,
            )
        )
    return boundaries


def select_boundaries(
    units: Sequence[TranscriptUnit],
    boundaries: Sequence[Boundary],
    *,
    min_chunk_words: int = 140,
    target_chunk_words: int = 450,
    max_chunk_words: int = 1800,
    boundary_threshold: float = 0.85,
    length_penalty: float = 0.14,
) -> list[int]:
    """Globally choose coherent cuts with soft length preferences."""
    if not units:
        return [0]
    if len(boundaries) != max(0, len(units) - 1):
        raise ValueError("Boundary count must be one less than unit count.")

    prefix = [0]
    for unit in units:
        prefix.append(prefix[-1] + unit.word_count)
    boundary_scores = {boundary.boundary_index: boundary.boundary_score for boundary in boundaries}
    count = len(units)
    negative_infinity = float("-inf")
    best = [negative_infinity] * (count + 1)
    previous: list[int | None] = [None] * (count + 1)
    best[0] = 0.0

    for end in range(1, count + 1):
        for start in range(end - 1, -1, -1):
            words = prefix[end] - prefix[start]
            single_oversized_unit = end == start + 1
            if words > max_chunk_words and not single_oversized_unit:
                continue
            entire_transcript = start == 0 and end == count
            if words < min_chunk_words and not entire_transcript:
                continue
            if best[start] == negative_infinity:
                continue
            relative_length = (words - target_chunk_words) / max(target_chunk_words, 1)
            score = best[start] - length_penalty * relative_length * relative_length
            if end < count:
                score += boundary_scores[end] - boundary_threshold
            if score > best[end]:
                best[end] = score
                previous[end] = start

    if previous[count] is None:
        return [0, count]
    selected = [count]
    cursor = count
    while cursor:
        cursor = previous[cursor] or 0
        selected.append(cursor)
    return sorted(set(selected))


def _chunk_id(episode_id: int, index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{episode_id}:c{index:04d}:{digest}"


def segment_transcript(
    transcript: EpisodeTranscript,
    unit_embeddings: np.ndarray,
    *,
    window_units: int = 3,
    semantic_weight: float = 0.82,
    strong_pause_seconds: float = 5.0,
    **selection_options: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return final chunks and complete boundary diagnostics for one episode."""
    boundaries = score_boundaries(
        transcript,
        unit_embeddings,
        window_units=window_units,
        semantic_weight=semantic_weight,
        strong_pause_seconds=strong_pause_seconds,
    )
    selected = select_boundaries(transcript.units, boundaries, **selection_options)
    selected_set = set(selected[1:-1])
    boundary_by_index = {boundary.boundary_index: boundary for boundary in boundaries}

    boundary_rows: list[dict[str, Any]] = []
    for boundary in boundaries:
        is_selected = boundary.boundary_index in selected_set
        if not is_selected:
            reason = None
        elif boundary.likely_music_break:
            reason = "music_break"
        elif boundary.boundary_score >= selection_options.get("boundary_threshold", 0.85):
            reason = "semantic_transition"
        else:
            reason = "length_constraint"
        boundary_rows.append(
            {
                **boundary.to_dict(),
                "selected": is_selected,
                "selection_reason": reason,
            }
        )

    chunks: list[dict[str, Any]] = []
    for chunk_index, (start_index, end_index) in enumerate(zip(selected, selected[1:])):
        group = transcript.units[start_index:end_index]
        text = normalize_text(" ".join(unit.text for unit in group))
        starting_boundary = boundary_by_index.get(start_index)
        chunks.append(
            {
                "chunk_id": _chunk_id(transcript.sr_episode_id, chunk_index, text),
                "sr_episode_id": transcript.sr_episode_id,
                "chunk_index": chunk_index,
                "start_seconds": group[0].start_seconds,
                "end_seconds": group[-1].end_seconds,
                "duration_seconds": group[-1].end_seconds - group[0].start_seconds,
                "word_count": sum(unit.word_count for unit in group),
                "unit_count": len(group),
                "unit_start_index": start_index,
                "unit_end_index": end_index,
                "text": text,
                "boundary_score": starting_boundary.boundary_score if starting_boundary else None,
                "pause_before_seconds": starting_boundary.pause_seconds if starting_boundary else 0.0,
                "transcript_path": transcript.transcript_path,
                "transcript_engine": transcript.transcript_engine,
                "transcript_model": transcript.transcript_model,
            }
        )
    return chunks, boundary_rows
