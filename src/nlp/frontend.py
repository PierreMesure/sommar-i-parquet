"""Build compact, static frontend data from topic-model outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def related_episode_rows(
    episode_embeddings: np.ndarray,
    episode_ids: Sequence[int],
    *,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Return each episode's nearest cosine neighbours in embedding space."""
    vectors = np.asarray(episode_embeddings, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(episode_ids):
        raise ValueError("Episode IDs and the embedding matrix must have equal length.")
    if not len(vectors):
        return []

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    neighbour_count = min(max(0, top_k), max(0, len(episode_ids) - 1))
    rows: list[dict[str, Any]] = []
    for index, episode_id in enumerate(episode_ids):
        if not neighbour_count:
            continue
        candidates = np.argpartition(similarities[index], -neighbour_count)[-neighbour_count:]
        candidates = candidates[np.argsort(similarities[index, candidates])[::-1]]
        for rank, candidate in enumerate(candidates, start=1):
            rows.append(
                {
                    "sr_episode_id": int(episode_id),
                    "related_episode_id": int(episode_ids[int(candidate)]),
                    "similarity": float(similarities[index, int(candidate)]),
                    "rank": rank,
                }
            )
    return rows


def topic_frontend_payload(
    topics: Sequence[dict[str, Any]],
    episode_topics: Sequence[dict[str, Any]],
    episode_map: Sequence[dict[str, Any]],
    related_episodes: Sequence[dict[str, Any]],
    *,
    max_topics_per_episode: int = 5,
) -> dict[str, Any]:
    """Create the compact topic catalogue and per-episode analysis records."""
    topic_records: dict[str, dict[str, Any]] = {}
    for topic in topics:
        topic_id = int(topic["topic_id"])
        if topic_id == -1 or topic.get("is_outlier") or topic.get("is_low_quality"):
            continue
        keywords = [str(value) for value in topic.get("keywords") or [] if value]
        label = str(topic.get("label") or "").strip()
        if not label or label.startswith(f"{topic_id}_"):
            label = " · ".join(keywords[:3]) or f"Ämne {topic_id}"
        record = {
            "label": label,
            "keywords": keywords[:8],
            "episodes": int(topic.get("episode_count") or 0),
            "chunks": int(topic.get("chunk_count") or 0),
        }
        parent = str(topic.get("parent_label") or "").strip()
        if parent:
            record["parent"] = parent
        topic_records[str(topic_id)] = record

    per_episode: dict[int, dict[str, Any]] = {}
    for row in episode_map:
        episode_id = int(row["sr_episode_id"])
        per_episode[episode_id] = {
            "x": round(float(row["x"]), 5),
            "y": round(float(row["y"]), 5),
            "dominant": int(row["dominant_topic_id"]),
            "topics": [],
            "related": [],
        }

    topic_rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in episode_topics:
        episode_id = int(row["sr_episode_id"])
        topic_id = int(row["topic_id"])
        if topic_id == -1 or str(topic_id) not in topic_records:
            continue
        topic_rows_by_episode.setdefault(episode_id, []).append(row)
    for episode_id, rows in topic_rows_by_episode.items():
        if episode_id not in per_episode:
            continue
        ranked = sorted(rows, key=lambda row: (-float(row["share"]), int(row["topic_id"])))
        per_episode[episode_id]["topics"] = [
            [int(row["topic_id"]), round(float(row["share"]), 4)]
            for row in ranked[:max_topics_per_episode]
        ]
        if (
            str(per_episode[episode_id]["dominant"]) not in topic_records
            and ranked
        ):
            per_episode[episode_id]["dominant"] = int(ranked[0]["topic_id"])

    for row in related_episodes:
        episode_id = int(row["sr_episode_id"])
        if episode_id not in per_episode:
            continue
        per_episode[episode_id]["related"].append(
            [int(row["related_episode_id"]), round(float(row["similarity"]), 4)]
        )

    # String keys compact predictably in JSON and avoid sparse arrays for SR IDs.
    return {
        "version": 1,
        "topics": topic_records,
        "episodes": {str(key): value for key, value in sorted(per_episode.items())},
    }


def write_topic_frontend_json(
    topics: Sequence[dict[str, Any]],
    episode_topics: Sequence[dict[str, Any]],
    episode_map: Sequence[dict[str, Any]],
    related_episodes: Sequence[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Write frontend topic data without whitespace to keep transfer small."""
    payload = topic_frontend_payload(
        topics,
        episode_topics,
        episode_map,
        related_episodes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return output_path


def curated_topic_frontend_payload(
    topics: Sequence[dict[str, Any]],
    episode_topics: Sequence[dict[str, Any]],
    episode_map: Sequence[dict[str, Any]],
    related_episodes: Sequence[dict[str, Any]],
    *,
    max_topics_per_episode: int = 5,
) -> dict[str, Any]:
    """Create a compact payload for stable, string-valued curated topic IDs.
    """
    topic_records = {
        str(topic["topic_id"]): {
            key: value
            for key, value in {
                "label": str(topic["label"]),
                "parent": str(topic.get("parent") or ""),
                "description": str(topic.get("description") or ""),
                "episodes": int(topic.get("episodes") or 0),
                "chunks": int(topic.get("chunks") or 0),
            }.items()
            if value != ""
        }
        for topic in topics
    }
    per_episode: dict[int, dict[str, Any]] = {
        int(row["sr_episode_id"]): {
            "x": round(float(row["x"]), 5),
            "y": round(float(row["y"]), 5),
            "dominant": None,
            "topics": [],
            "related": [],
        }
        for row in episode_map
    }
    topic_rows_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_topics:
        topic_id = str(row["topic_id"])
        episode_id = int(row["sr_episode_id"])
        if topic_id in topic_records and episode_id in per_episode:
            topic_rows_by_episode[episode_id].append(row)
    filtered_topic_rows: dict[int, list[dict[str, Any]]] = {}
    for episode_id, rows in topic_rows_by_episode.items():
        def topic_share(row: dict[str, Any]) -> float:
            return float(row.get("coverage") or row.get("share") or 0)

        ranked = sorted(rows, key=lambda row: (-topic_share(row), str(row["topic_id"])))
        filtered_topic_rows[episode_id] = ranked
        per_episode[episode_id]["topics"] = [
            [str(row["topic_id"]), round(topic_share(row), 4)]
            for row in ranked[:max_topics_per_episode]
        ]
        if ranked:
            per_episode[episode_id]["dominant"] = str(ranked[0]["topic_id"])
    # Keep filter tooltips/counts consistent with the associations actually
    # presented to users, rather than the unfiltered analytical totals.
    topic_episode_counts: dict[str, set[int]] = defaultdict(set)
    for episode_id, rows in filtered_topic_rows.items():
        for row in rows:
            topic_episode_counts[str(row["topic_id"])].add(episode_id)
    for topic_id, record in topic_records.items():
        record["episodes"] = len(topic_episode_counts.get(topic_id, set()))
    for row in related_episodes:
        episode_id = int(row["sr_episode_id"])
        if episode_id in per_episode:
            per_episode[episode_id]["related"].append(
                [int(row["related_episode_id"]), round(float(row["similarity"]), 4)]
            )
    return {
        "version": 2,
        "topics": topic_records,
        "episodes": {str(key): value for key, value in sorted(per_episode.items())},
    }


def write_curated_topic_frontend_json(
    topics: Sequence[dict[str, Any]],
    episode_topics: Sequence[dict[str, Any]],
    episode_map: Sequence[dict[str, Any]],
    related_episodes: Sequence[dict[str, Any]],
    output_path: Path,
) -> Path:
    payload = curated_topic_frontend_payload(
        topics,
        episode_topics,
        episode_map,
        related_episodes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return output_path
