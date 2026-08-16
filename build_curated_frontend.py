"""Build the static frontend topic payload from curated topic assignments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import pyarrow.parquet as pq

from src.nlp.curated import topic_slug
from src.nlp.frontend import write_curated_topic_frontend_json


PARENT_LABELS = {
    "Relationships, life events, and health": "Relationer, livshändelser och hälsa",
    "Culture, media, and creative work": "Kultur, medier och skapande",
    "Society, identity, education, and work": "Samhälle, identitet, utbildning och arbete",
    "International affairs and history": "Internationella frågor och historia",
    "Science, nature, food, religion, and traditions": "Vetenskap, natur, mat, religion och traditioner",
    "Sport, adventure, transport, and places": "Sport, äventyr, transport och platser",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/nlp/curated"))
    parser.add_argument("--raw-nlp-dir", type=Path, default=Path("data/nlp"))
    parser.add_argument("--proposal", type=Path, default=Path("SPECIFIC_TOPICS_PROPOSAL.md"))
    parser.add_argument("--output", type=Path, default=Path("data/nlp/topics.json"))
    return parser.parse_args()


def proposal_parents(path: Path) -> dict[str, str]:
    """Return provisional broad-theme labels for the proposal's leaf topics."""
    current_parent = ""
    parents: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
            current_parent = PARENT_LABELS.get(heading, "")
            continue
        if line.startswith("- **") and "** — " in line and "_(raw:" in line:
            label = line.split("**", 2)[1]
            parents[topic_slug(label)] = current_parent
    return parents


def main() -> None:
    args = parse_args()
    topics = pq.read_table(args.input_dir / "topics.parquet").to_pylist()
    episode_topics = pq.read_table(args.input_dir / "episode_topics.parquet").to_pylist()
    episode_map = pq.read_table(args.raw_nlp_dir / "episode_map.parquet").to_pylist()
    related = pq.read_table(args.raw_nlp_dir / "related_episodes.parquet").to_pylist()
    parents = proposal_parents(args.proposal)
    episode_counts: dict[str, set[int]] = defaultdict(set)
    chunk_counts: dict[str, int] = defaultdict(int)
    for row in episode_topics:
        topic_id = str(row["topic_id"])
        episode_counts[topic_id].add(int(row["sr_episode_id"]))
        chunk_counts[topic_id] += int(row.get("chunk_count") or 0)
    topic_records = []
    for topic in topics:
        topic_id = str(topic["topic_id"])
        topic_records.append(
            {
                "topic_id": topic_id,
                "label": str(topic["label"]),
                "parent": parents.get(topic_id, ""),
                "description": str(topic.get("description") or ""),
                "episodes": len(episode_counts.get(topic_id, set())),
                "chunks": chunk_counts.get(topic_id, 0),
            }
        )
    output = write_curated_topic_frontend_json(
        topic_records,
        episode_topics,
        episode_map,
        related,
        args.output,
    )
    print(f"Wrote curated frontend topics to {output}")


if __name__ == "__main__":
    main()
