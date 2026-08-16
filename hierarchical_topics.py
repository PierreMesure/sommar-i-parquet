"""Build an inspectable BERTopic hierarchy from an existing topic model.

This is a post-hoc tree over existing topic clusters. It does not re-cluster
transcript chunks or choose a new number of initial topics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from bertopic import BERTopic

from src.nlp.topics import write_parquet_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/nlp"))
    parser.add_argument("--output", type=Path, default=Path("data/nlp/topic_hierarchy.parquet"))
    parser.add_argument(
        "--representation",
        choices=("ctfidf", "embeddings"),
        default="ctfidf",
        help="Distance space for merging existing topics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = pq.read_table(args.input_dir / "chunk_topics.parquet").to_pylist()
    labels_path = args.input_dir / "topics_labeled.parquet"
    label_rows = pq.read_table(labels_path).to_pylist() if labels_path.exists() else []
    labels = {str(row["topic_id"]): row.get("llm_label") or row["label"] for row in label_rows}

    model = BERTopic.load(args.input_dir / "bertopic_model")
    hierarchy = model.hierarchical_topics(
        [str(row["text"]) for row in chunks],
        use_ctfidf=args.representation == "ctfidf",
    )
    rows = hierarchy.to_dict(orient="records")
    for row in rows:
        row["leaf_topic_labels"] = [
            labels.get(str(topic_id), str(topic_id)) for topic_id in row["Topics"]
        ]
    write_parquet_rows(rows, args.output)
    print(f"Wrote {len(rows)} hierarchy merges to {args.output}")


if __name__ == "__main__":
    main()
