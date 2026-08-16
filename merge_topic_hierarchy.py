"""Build the reviewed topic hierarchy from the 500-cluster BERTopic run.

No model inference or semantic reassignment happens here. Every accepted
chunk keeps its raw BERTopic membership, which is deterministically merged
through the editorial configuration in :mod:`src.nlp.manual_hierarchy`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.nlp.manual_hierarchy import (
    EXCLUDED,
    MANUAL_EPISODE_TOPICS,
    SPECIFIC_TOPICS,
    TOPIC_EVIDENCE_RULES,
    validate,
)
from src.nlp.frontend import write_curated_topic_frontend_json
from src.nlp.topics import write_parquet_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/nlp/k500"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/nlp/k500/hierarchy")
    )
    return parser.parse_args()


def evidence_allows(topic_id: str, rows: list[dict]) -> bool:
    """Require concentrated lexical evidence for topics known to be noisy.

    Raw clusters are useful candidates, but a single chunk can mention a
    sport in passing.  These rules operate on each episode's own candidate
    chunks and deliberately contain no person names.
    """
    rule = TOPIC_EVIDENCE_RULES.get(topic_id)
    if not rule:
        return True
    text = " ".join(str(row.get("text") or "").lower() for row in rows)
    terms = [str(term).lower() for term in rule.get("terms", [])]
    distinct_terms = sum(term in text for term in terms)
    return (
        len(rows) >= int(rule.get("min_chunks", 1))
        and distinct_terms >= int(rule.get("min_distinct_terms", 1))
    )


def main() -> None:
    args = parse_args()
    validate()
    raw_topics = {
        int(row["topic_id"]): row
        for row in pq.read_table(args.input_dir / "topics_labeled.parquet").to_pylist()
    }
    chunk_rows = pq.read_table(args.input_dir / "chunk_topics.parquet").to_pylist()

    raw_to_specific = {
        int(raw_id): topic_id
        for topic_id, topic in SPECIFIC_TOPICS.items()
        for raw_id in topic["raw"]
    }
    broad_ids = {
        label: f"broad-{index:02d}"
        for index, label in enumerate(
            sorted({str(topic["broad"]) for topic in SPECIFIC_TOPICS.values()}),
            start=1,
        )
    }

    candidate_chunks: list[dict] = []
    episode_word_totals: dict[int, int] = defaultdict(int)
    for chunk in chunk_rows:
        episode_id = int(chunk["sr_episode_id"])
        words = int(chunk.get("word_count") or 0)
        episode_word_totals[episode_id] += words
        specific_id = raw_to_specific.get(int(chunk["topic_id"]))
        if specific_id is None:
            continue
        topic = SPECIFIC_TOPICS[specific_id]
        candidate_chunks.append(
            {
                **chunk,
                "raw_topic_id": int(chunk["topic_id"]),
                "topic_id": specific_id,
                "topic_label": str(topic["label"]),
                "broad_topic_id": broad_ids[str(topic["broad"])],
                "broad_topic_label": str(topic["broad"]),
            }
        )
    candidate_groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in candidate_chunks:
        candidate_groups[(int(row["sr_episode_id"]), str(row["topic_id"]))].append(row)
    allowed = {
        key for key, rows in candidate_groups.items() if evidence_allows(key[1], rows)
    }
    specific_chunks = [
        row
        for row in candidate_chunks
        if (int(row["sr_episode_id"]), str(row["topic_id"])) in allowed
    ]

    topic_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {"chunks": 0, "episodes": set(), "words": 0}
    )
    episode_stats: dict[tuple[int, str], dict[str, object]] = defaultdict(
        lambda: {"chunks": 0, "words": 0, "source_title": "", "representative": None}
    )
    for chunk in specific_chunks:
        episode_id = int(chunk["sr_episode_id"])
        specific_id = str(chunk["topic_id"])
        words = int(chunk.get("word_count") or 0)
        stats = topic_stats[specific_id]
        stats["chunks"] = int(stats["chunks"]) + 1
        stats["words"] = int(stats["words"]) + words
        episodes = stats["episodes"]
        assert isinstance(episodes, set)
        episodes.add(episode_id)
        episode = episode_stats[(episode_id, specific_id)]
        episode["chunks"] = int(episode["chunks"]) + 1
        episode["words"] = int(episode["words"]) + words
        episode["source_title"] = str(chunk.get("source_title") or "")
        representative = episode["representative"]
        if representative is None or words > int(representative.get("word_count") or 0):
            episode["representative"] = chunk

    # Add only the two transcript-reviewed historical exceptions.  These do
    # not alter raw chunk membership and are marked explicitly below.
    for episode_id, assignments in MANUAL_EPISODE_TOPICS.items():
        for topic_id in assignments:
            if topic_id not in SPECIFIC_TOPICS:
                raise ValueError(f"Unknown manual topic {topic_id!r}")
            stats = topic_stats[topic_id]
            episodes = stats["episodes"]
            assert isinstance(episodes, set)
            episodes.add(int(episode_id))

    specific_rows = []
    raw_mapping_rows = []
    for topic_id, topic in SPECIFIC_TOPICS.items():
        raw_ids = [int(value) for value in topic["raw"]]
        stats = topic_stats[topic_id]
        specific_rows.append(
            {
                "topic_id": topic_id,
                "label": str(topic["label"]),
                "broad_topic_id": broad_ids[str(topic["broad"])],
                "broad_topic_label": str(topic["broad"]),
                "raw_topic_ids": raw_ids,
                "chunk_count": int(stats["chunks"]),
                "episode_count": len(stats["episodes"]),
                "word_count": int(stats["words"]),
            }
        )
        raw_mapping_rows.extend(
            {
                "raw_topic_id": raw_id,
                "topic_id": topic_id,
                "topic_label": str(topic["label"]),
            }
            for raw_id in raw_ids
        )

    episode_rows = []
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for (episode_id, topic_id), stats in episode_stats.items():
        representative = stats["representative"]
        assert isinstance(representative, dict)
        words = int(stats["words"])
        row = {
            "sr_episode_id": episode_id,
            "topic_id": topic_id,
            "topic_label": str(SPECIFIC_TOPICS[topic_id]["label"]),
            "broad_topic_id": broad_ids[str(SPECIFIC_TOPICS[topic_id]["broad"])],
            "broad_topic_label": str(SPECIFIC_TOPICS[topic_id]["broad"]),
            "chunk_count": int(stats["chunks"]),
            "word_count": words,
            "share": words / episode_word_totals[episode_id],
            "representative_chunk_id": str(representative["chunk_id"]),
            "representative_start_seconds": float(representative["start_seconds"]),
            "representative_excerpt": str(representative["text"])[:500],
            "source_title": str(stats["source_title"]),
            "assignment_source": "raw_cluster",
        }
        by_episode[episode_id].append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda row: (-float(row["share"]), str(row["topic_id"])))
        for rank, row in enumerate(rows, start=1):
            episode_rows.append({**row, "rank": rank})

    # The manual rows have no chunk-level evidence by design.  Append them to
    # the same episode table with a visible provenance flag, then recompute
    # ranks so the strongest reviewed subject remains first.
    for episode_id, assignments in MANUAL_EPISODE_TOPICS.items():
        for topic_id, share in assignments.items():
            topic = SPECIFIC_TOPICS[topic_id]
            episode_rows.append(
                {
                    "sr_episode_id": int(episode_id),
                    "topic_id": topic_id,
                    "topic_label": str(topic["label"]),
                    "broad_topic_id": broad_ids[str(topic["broad"])],
                    "broad_topic_label": str(topic["broad"]),
                    "chunk_count": 0,
                    "word_count": 0,
                    "share": float(share),
                    "representative_chunk_id": None,
                    "representative_start_seconds": None,
                    "representative_excerpt": "Manuell tillagt efter genomläsning av hela transkriptet.",
                    "source_title": "",
                    "rank": 0,
                    "assignment_source": "manual_transcript_review",
                }
            )
    ranked_by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in episode_rows:
        ranked_by_episode[int(row["sr_episode_id"])].append(row)
    episode_rows = []
    for rows in ranked_by_episode.values():
        rows.sort(key=lambda row: (-float(row["share"]), str(row["topic_id"])))
        episode_rows.extend({**row, "rank": rank} for rank, row in enumerate(rows, start=1))

    broad_rows = []
    for label, broad_id in broad_ids.items():
        children = [
            topic_id
            for topic_id, topic in SPECIFIC_TOPICS.items()
            if topic["broad"] == label
        ]
        broad_rows.append(
            {
                "broad_topic_id": broad_id,
                "label": label,
                "specific_topic_ids": children,
                "episode_count": len(
                    {
                        int(row["sr_episode_id"])
                        for row in episode_rows
                        if row["broad_topic_id"] == broad_id
                    }
                ),
            }
        )

    broad_episode_stats: dict[tuple[int, str], dict[str, object]] = defaultdict(
        lambda: {"chunks": 0, "words": 0, "source_title": ""}
    )
    for row in episode_rows:
        key = (int(row["sr_episode_id"]), str(row["broad_topic_id"]))
        stats = broad_episode_stats[key]
        stats["chunks"] = int(stats["chunks"]) + int(row["chunk_count"])
        stats["words"] = int(stats["words"]) + int(row["word_count"])
        stats["source_title"] = str(row["source_title"])
    broad_episode_rows = [
        {
            "sr_episode_id": episode_id,
            "broad_topic_id": broad_id,
            "broad_topic_label": next(
                row["label"] for row in broad_rows if row["broad_topic_id"] == broad_id
            ),
            "chunk_count": int(stats["chunks"]),
            "word_count": int(stats["words"]),
            "share": int(stats["words"]) / episode_word_totals[episode_id],
            "source_title": str(stats["source_title"]),
        }
        for (episode_id, broad_id), stats in broad_episode_stats.items()
    ]

    excluded_rows = [
        {
            "raw_topic_id": raw_id,
            "reason": reason,
            "llm_label": str(raw_topics[raw_id].get("llm_label") or ""),
            "keywords": raw_topics[raw_id].get("keywords") or [],
            "chunk_count": int(raw_topics[raw_id].get("chunk_count") or 0),
            "episode_count": int(raw_topics[raw_id].get("episode_count") or 0),
        }
        for reason, raw_ids in EXCLUDED.items()
        for raw_id in raw_ids
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_rows(specific_rows, args.output_dir / "topics.parquet")
    write_parquet_rows(broad_rows, args.output_dir / "broad_topics.parquet")
    write_parquet_rows(raw_mapping_rows, args.output_dir / "raw_to_specific.parquet")
    write_parquet_rows(excluded_rows, args.output_dir / "excluded_raw_topics.parquet")
    write_parquet_rows(specific_chunks, args.output_dir / "chunk_topics.parquet")
    write_parquet_rows(episode_rows, args.output_dir / "episode_topics.parquet")
    write_parquet_rows(broad_episode_rows, args.output_dir / "episode_broad_topics.parquet")
    frontend_topics = [
        {
            "topic_id": row["topic_id"],
            "label": row["label"],
            "parent": row["broad_topic_label"],
            "episodes": row["episode_count"],
            "chunks": row["chunk_count"],
        }
        for row in specific_rows
    ]
    write_curated_topic_frontend_json(
        frontend_topics,
        episode_rows,
        pq.read_table(args.input_dir / "episode_map.parquet").to_pylist(),
        pq.read_table(args.input_dir / "related_episodes.parquet").to_pylist(),
        args.output_dir / "topics.json",
    )

    covered = len({int(row["sr_episode_id"]) for row in episode_rows})
    total = len(episode_word_totals)
    print(
        f"Wrote {len(specific_rows)} specific topics in {len(broad_rows)} broad themes; "
        f"{covered}/{total} episodes retain at least one topic"
    )


if __name__ == "__main__":
    main()
