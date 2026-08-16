"""Apply a reviewed hierarchy proposal to the active topic catalogue."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from src.nlp.frontend import write_topic_frontend_json
from src.nlp.topics import write_parquet_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/nlp"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topics = pq.read_table(args.input_dir / "topic_hierarchy_proposal.parquet").to_pylist()
    write_parquet_rows(topics, args.input_dir / "topics.parquet")
    write_topic_frontend_json(
        topics,
        pq.read_table(args.input_dir / "episode_topics.parquet").to_pylist(),
        pq.read_table(args.input_dir / "episode_map.parquet").to_pylist(),
        pq.read_table(args.input_dir / "related_episodes.parquet").to_pylist(),
        args.input_dir / "topics.json",
    )
    print("Applied topic hierarchy to topics.parquet and topics.json")


if __name__ == "__main__":
    main()
