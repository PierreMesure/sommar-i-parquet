"""Controlled topic catalogue parsing and semantic multi-label assignment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import unicodedata
from typing import Mapping, Sequence

import numpy as np


TOPIC_PATTERN = re.compile(
    r"^- \*\*(?P<label>.+?)\*\* — (?P<description>.+?) "
    r"_\(raw: (?P<raw_ids>[0-9, ]+)\)_$"
)


@dataclass(frozen=True)
class CuratedTopic:
    topic_id: str
    label: str
    description: str
    raw_topic_ids: tuple[int, ...]
    positive_examples: tuple[str, ...] = ()
    negative_examples: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        row = asdict(self)
        row["raw_topic_ids"] = list(self.raw_topic_ids)
        row["positive_examples"] = list(self.positive_examples)
        row["negative_examples"] = list(self.negative_examples)
        row["query_text"] = topic_query_text(self)
        return row


def topic_slug(label: str) -> str:
    folded = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", folded.lower()).strip("-")


def parse_curated_topic_proposal(path: Path) -> list[CuratedTopic]:
    """Parse accepted topic rows, stopping before the rejection appendix."""
    topics: list[CuratedTopic] = []
    seen_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "## Raw clusters intentionally not promoted":
            break
        match = TOPIC_PATTERN.match(line)
        if not match:
            continue
        label = match.group("label")
        topic_id = topic_slug(label)
        if topic_id in seen_ids:
            raise ValueError(f"Duplicate curated topic ID: {topic_id}")
        seen_ids.add(topic_id)
        topics.append(
            CuratedTopic(
                topic_id=topic_id,
                label=label,
                description=match.group("description").rstrip("."),
                raw_topic_ids=tuple(
                    int(value) for value in match.group("raw_ids").split(", ")
                ),
            )
        )
    if not topics:
        raise ValueError(f"No curated topics found in {path}")
    return topics


def topic_query_text(topic: CuratedTopic) -> str:
    parts = [
        f"Ämne: {topic.label}.",
        f"Texten handlar huvudsakligen om {topic.description}.",
    ]
    if topic.positive_examples:
        parts.append("Typiska exempel: " + "; ".join(topic.positive_examples) + ".")
    return " ".join(parts)


def apply_topic_guidance(
    topics: Sequence[CuratedTopic],
    guidance_path: Path,
) -> list[CuratedTopic]:
    """Apply reviewed/generated Swedish definitions and boundary examples."""
    payload = json.loads(guidance_path.read_text(encoding="utf-8"))
    guidance = {str(row["topic_id"]): row for row in payload["topics"]}
    expected = {topic.topic_id for topic in topics}
    if set(guidance) != expected:
        missing = sorted(expected - set(guidance))
        extra = sorted(set(guidance) - expected)
        raise ValueError(f"Guidance/topic mismatch; missing={missing}, extra={extra}")
    return [
        CuratedTopic(
            topic_id=topic.topic_id,
            label=topic.label,
            description=str(guidance[topic.topic_id]["description_sv"]),
            raw_topic_ids=topic.raw_topic_ids,
            positive_examples=tuple(guidance[topic.topic_id]["positive_examples"]),
            negative_examples=tuple(guidance[topic.topic_id]["negative_examples"]),
        )
        for topic in topics
    ]


def assign_chunk_topics(
    chunks: Sequence[dict],
    chunk_embeddings: np.ndarray,
    topics: Sequence[CuratedTopic],
    topic_embeddings: np.ndarray,
    *,
    min_similarity: float,
    topic_min_similarities: Mapping[str, float] | None = None,
    min_winner_margin: float = 0.01,
    accept_ambiguous_above: float | None = 0.56,
    supporting_min_similarity: float | None = 0.495,
    supporting_min_winner_margin: float = 0.008,
    negative_embeddings: np.ndarray | None = None,
    min_negative_margin: float | None = None,
    secondary_min_similarity: float | None = None,
    secondary_max_score_gap: float = 0.015,
) -> tuple[list[dict], list[dict]]:
    """Classify chunks against a controlled topic catalogue.

    The best topic must pass both its absolute topic-specific floor and a
    margin over the runner-up. This deliberately leaves ambiguous chunks
    unassigned. An optional second label uses a separate, stricter floor and
    is disabled by default.

    Returns accepted assignments and one decision/audit row per chunk.
    """
    chunk_vectors = np.asarray(chunk_embeddings, dtype=np.float32)
    topic_vectors = np.asarray(topic_embeddings, dtype=np.float32)
    if len(chunks) != len(chunk_vectors):
        raise ValueError("Every chunk needs exactly one embedding")
    if len(topics) != len(topic_vectors):
        raise ValueError("Every curated topic needs exactly one embedding")
    if len(topics) < 2:
        raise ValueError("At least two topics are required to calculate a winner margin")
    if (
        min_winner_margin < 0
        or supporting_min_winner_margin < 0
        or secondary_max_score_gap < 0
    ):
        raise ValueError("Score margins cannot be negative")
    negative_vectors: np.ndarray | None = None
    if negative_embeddings is not None:
        negative_vectors = np.asarray(negative_embeddings, dtype=np.float32)
        if negative_vectors.ndim != 3 or negative_vectors.shape[:2] != (len(topics), 2):
            raise ValueError("negative_embeddings must have shape (topics, 2, dimensions)")
        if negative_vectors.shape[2] != chunk_vectors.shape[1]:
            raise ValueError("Negative and chunk embedding dimensions differ")
    topic_floors = dict(topic_min_similarities or {})
    unknown_topic_ids = set(topic_floors) - {topic.topic_id for topic in topics}
    if unknown_topic_ids:
        raise ValueError(f"Thresholds contain unknown topics: {sorted(unknown_topic_ids)}")
    similarities = chunk_vectors @ topic_vectors.T
    candidate_indices = np.argpartition(similarities, kth=-2, axis=1)[:, -2:]
    rows: list[dict] = []
    decisions: list[dict] = []
    for chunk_index, topic_indices in enumerate(candidate_indices):
        selected = sorted(
            topic_indices,
            key=lambda index: float(similarities[chunk_index, index]),
            reverse=True,
        )
        best_index, runner_up_index = selected
        best_topic = topics[best_index]
        runner_up_topic = topics[runner_up_index]
        best_score = float(similarities[chunk_index, best_index])
        runner_up_score = float(similarities[chunk_index, runner_up_index])
        winner_margin = best_score - runner_up_score
        negative_similarity = (
            float(np.max(chunk_vectors[chunk_index] @ negative_vectors[best_index].T))
            if negative_vectors is not None
            else None
        )
        negative_margin = (
            best_score - negative_similarity
            if negative_similarity is not None
            else None
        )
        required_similarity = float(topic_floors.get(best_topic.topic_id, min_similarity))
        passes_negative_boundary = (
            min_negative_margin is None
            or negative_margin is None
            or negative_margin >= min_negative_margin
        )
        passes_winner_margin = (
            winner_margin >= min_winner_margin
            or (
                accept_ambiguous_above is not None
                and best_score >= accept_ambiguous_above
            )
        )
        strong_accepted = (
            best_score >= required_similarity
            and passes_winner_margin
            and passes_negative_boundary
        )
        supporting_accepted = (
            not strong_accepted
            and supporting_min_similarity is not None
            and best_score >= supporting_min_similarity
            and winner_margin >= supporting_min_winner_margin
            and passes_negative_boundary
        )
        accepted = strong_accepted or supporting_accepted
        evidence_tier = "strong" if strong_accepted else "supporting" if supporting_accepted else None
        if (
            supporting_min_similarity is not None
            and best_score < supporting_min_similarity
        ):
            rejection_reason = "below_supporting_threshold"
        elif supporting_min_similarity is None and best_score < required_similarity:
            rejection_reason = "below_topic_threshold"
        elif not passes_winner_margin and not supporting_accepted:
            rejection_reason = "ambiguous_winner"
        elif not passes_negative_boundary:
            rejection_reason = "negative_near_miss"
        elif not accepted:
            rejection_reason = "below_topic_threshold"
        else:
            rejection_reason = None
        chunk = chunks[chunk_index]
        decision = {
            "chunk_id": str(chunk["chunk_id"]),
            "sr_episode_id": int(chunk["sr_episode_id"]),
            "best_topic_id": best_topic.topic_id,
            "best_topic_label": best_topic.label,
            "best_similarity": best_score,
            "required_similarity": required_similarity,
            "runner_up_topic_id": runner_up_topic.topic_id,
            "runner_up_topic_label": runner_up_topic.label,
            "runner_up_similarity": runner_up_score,
            "winner_margin": winner_margin,
            "negative_similarity": negative_similarity,
            "negative_margin": negative_margin,
            "accepted": accepted,
            "evidence_tier": evidence_tier,
            "rejection_reason": rejection_reason,
            "word_count": int(chunk["word_count"]),
            "start_seconds": float(chunk["start_seconds"]),
            "end_seconds": float(chunk["end_seconds"]),
            "excerpt": str(chunk["text"])[:500],
        }
        decisions.append(decision)
        if not accepted:
            continue
        common = {
            "chunk_id": decision["chunk_id"],
            "sr_episode_id": decision["sr_episode_id"],
            "word_count": decision["word_count"],
            "start_seconds": decision["start_seconds"],
            "end_seconds": decision["end_seconds"],
            "excerpt": decision["excerpt"],
        }
        rows.append(
            {
                **common,
                "topic_id": best_topic.topic_id,
                "topic_label": best_topic.label,
                "similarity": best_score,
                "chunk_topic_rank": 1,
                "winner_margin": winner_margin,
                "evidence_tier": evidence_tier,
            }
        )
        if secondary_min_similarity is not None:
            runner_up_floor = max(
                secondary_min_similarity,
                float(topic_floors.get(runner_up_topic.topic_id, min_similarity)),
            )
            if (
                runner_up_score >= runner_up_floor
                and winner_margin <= secondary_max_score_gap
            ):
                rows.append(
                    {
                        **common,
                        "topic_id": runner_up_topic.topic_id,
                        "topic_label": runner_up_topic.label,
                        "similarity": runner_up_score,
                        "chunk_topic_rank": 2,
                        "winner_margin": -winner_margin,
                        "evidence_tier": "secondary",
                    }
                )
    return rows, decisions


def aggregate_episode_topics(
    chunks: Sequence[dict],
    assignments: Sequence[dict],
    *,
    min_coverage: float = 0.03,
    min_evidence_words: int = 140,
    single_chunk_min_similarity: float = 0.515,
    min_supporting_chunks: int = 2,
    supporting_min_coverage: float = 0.08,
    supporting_min_evidence_words: int = 250,
) -> list[dict]:
    """Aggregate matched chunk evidence into auditable episode-level tags."""
    episode_words: dict[int, int] = defaultdict(int)
    for chunk in chunks:
        episode_words[int(chunk["sr_episode_id"])] += int(chunk["word_count"])
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in assignments:
        grouped[(int(row["sr_episode_id"]), str(row["topic_id"]))].append(row)
    rows: list[dict] = []
    for (episode_id, topic_id), evidence in grouped.items():
        evidence_words = sum(int(row["word_count"]) for row in evidence)
        coverage = evidence_words / max(episode_words[episode_id], 1)
        strong_chunk_count = sum(
            str(row.get("evidence_tier", "strong")) in {"strong", "secondary"}
            for row in evidence
        )
        max_similarity = max(float(row["similarity"]) for row in evidence)
        passes_strong_rule = (
            strong_chunk_count >= 1
            and evidence_words >= min_evidence_words
            and coverage >= min_coverage
            and (
                len(evidence) >= 2
                or max_similarity >= single_chunk_min_similarity
            )
        )
        passes_supporting_rule = (
            len(evidence) >= min_supporting_chunks
            and evidence_words >= supporting_min_evidence_words
            and coverage >= supporting_min_coverage
        )
        if not passes_strong_rule and not passes_supporting_rule:
            continue
        mean_similarity = sum(
            float(row["similarity"]) * int(row["word_count"]) for row in evidence
        ) / evidence_words
        rows.append(
            {
                "sr_episode_id": episode_id,
                "topic_id": topic_id,
                "topic_label": evidence[0]["topic_label"],
                "chunk_count": len(evidence),
                "strong_chunk_count": strong_chunk_count,
                "evidence_words": evidence_words,
                "coverage": coverage,
                "mean_similarity": mean_similarity,
                "max_similarity": max_similarity,
                "score": coverage * mean_similarity,
                "representative_excerpt": max(
                    evidence, key=lambda row: float(row["similarity"])
                )["excerpt"],
            }
        )
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[int(row["sr_episode_id"])].append(row)
    ranked: list[dict] = []
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: (-row["score"], row["topic_id"]))
        ranked.extend({**row, "rank": rank} for rank, row in enumerate(episode_rows, 1))
    return ranked


def sample_topic_calibration_candidates(
    decisions: Sequence[dict],
    *,
    per_score_band: int = 2,
    score_bands: Sequence[tuple[float, float]] = (
        (0.0, 0.48),
        (0.48, 0.50),
        (0.50, 0.52),
        (0.52, 0.54),
        (0.54, 0.56),
        (0.56, 1.0),
    ),
) -> list[dict]:
    """Select deterministic score-stratified examples for editorial review."""
    if per_score_band < 1:
        raise ValueError("per_score_band must be positive")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in decisions:
        grouped[str(row["best_topic_id"])].append(row)
    sampled: list[dict] = []
    for topic_id in sorted(grouped):
        topic_rows = grouped[topic_id]
        selected_ids: set[str] = set()
        for lower, upper in score_bands:
            candidates = sorted(
                (
                    row for row in topic_rows
                    if lower <= float(row["best_similarity"]) < upper
                ),
                key=lambda row: (float(row["best_similarity"]), str(row["chunk_id"])),
            )
            if len(candidates) <= per_score_band:
                chosen = candidates
            else:
                positions = np.linspace(0, len(candidates) - 1, per_score_band, dtype=int)
                chosen = [candidates[int(position)] for position in positions]
            for row in chosen:
                chunk_id = str(row["chunk_id"])
                if chunk_id not in selected_ids:
                    selected_ids.add(chunk_id)
                    sampled.append(dict(row))
    return sampled


def estimate_topic_threshold(
    reviewed_candidates: Sequence[dict],
    *,
    default_threshold: float,
    target_precision: float = 0.85,
    min_relevant_examples: int = 2,
) -> tuple[float, dict]:
    """Estimate a conservative floor from human/LLM-reviewed candidates.

    Candidate sampling is deliberately stratified rather than representative,
    so the result is a boundary estimate, not a corpus precision estimate.
    """
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be between 0 and 1")
    rows = sorted(
        reviewed_candidates,
        key=lambda row: float(row["best_similarity"]),
        reverse=True,
    )
    candidates: list[tuple[float, float, int, int]] = []
    for threshold in sorted({float(row["best_similarity"]) for row in rows}):
        above = [row for row in rows if float(row["best_similarity"]) >= threshold]
        relevant = sum(bool(row["relevant"]) for row in above)
        precision = relevant / len(above)
        if relevant >= min_relevant_examples and precision >= target_precision:
            candidates.append((threshold, precision, relevant, len(above)))
    if candidates:
        threshold, precision, relevant, reviewed_above = min(candidates, key=lambda row: row[0])
        # Avoid encoding meaningless floating-point noise in a reviewed config.
        threshold = round(max(threshold, 0.0), 3)
        source = "reviewed_boundary"
    else:
        threshold = float(default_threshold)
        precision = 0.0
        relevant = 0
        reviewed_above = 0
        source = "default_insufficient_evidence"
    return threshold, {
        "source": source,
        "reviewed_examples": len(rows),
        "reviewed_relevant": sum(bool(row["relevant"]) for row in rows),
        "reviewed_precision_above_threshold": precision,
        "reviewed_relevant_above_threshold": relevant,
        "reviewed_examples_above_threshold": reviewed_above,
    }
