"""Fine-tune an existing BERTopic model's representations with OpenAI.

Cluster membership is read from ``chunk_topics.parquet`` and never changed.
BERTopic itself recalculates c-TF-IDF, selects representative chunk texts, and
passes the selected/truncated evidence to its OpenAI representation class.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from bertopic import BERTopic

from src.nlp.frontend import write_topic_frontend_json
from src.nlp.topics import build_openai_representation, write_parquet_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/nlp"))
    parser.add_argument("--output", type=Path, default=Path("data/nlp/topics_labeled.parquet"))
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--document-tokens", type=int, default=120)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--offset", type=int, default=0, help="Topic offset for resumable batches.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum usable topics to label in this invocation (all by default).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace topics.parquet and topics.json with the labelled topic catalogue.",
    )
    parser.add_argument(
        "--topic-ids",
        type=int,
        nargs="+",
        help="Explicit usable topic IDs to relabel; overrides --offset/--limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    chunk_rows = pq.read_table(args.input_dir / "chunk_topics.parquet").to_pylist()
    topic_rows = pq.read_table(args.input_dir / "topics.parquet").to_pylist()
    model = BERTopic.load(args.input_dir / "bertopic_model")
    # Outliers and explicitly low-quality clusters remain available in the raw
    # model, but should not consume editorial labelling calls or enter the UI.
    topic_ids = [
        int(row["topic_id"])
        for row in topic_rows
        if int(row["topic_id"]) != -1 and not row.get("is_low_quality")
    ]
    selected_ids = topic_ids[args.offset:]
    if args.topic_ids is not None:
        requested = set(args.topic_ids)
        unknown = requested - set(topic_ids)
        if unknown:
            raise ValueError(f"Unknown or excluded topic IDs: {sorted(unknown)}")
        selected_ids = [topic_id for topic_id in topic_ids if topic_id in requested]
    elif args.limit is not None:
        selected_ids = selected_ids[:args.limit]
    if not selected_ids:
        logging.info("No topics selected at offset %d", args.offset)
        return
    raw_topics = model.get_topics()
    selected_topics = {topic_id: raw_topics[topic_id] for topic_id in selected_ids}
    documents = pd.DataFrame(
        {
            "Document": [str(row["text"]) for row in chunk_rows],
            "Topic": [int(row["topic_id"]) for row in chunk_rows],
            "ID": range(len(chunk_rows)),
            "Image": None,
        }
    )
    # This is the same BERTopic representation method used by update_topics,
    # but we pass a subset of the existing c-TF-IDF topic mapping. It lets the
    # job resume under short-lived runners without changing topic assignments.
    labels = build_openai_representation(
        model=args.model,
        document_tokens=args.document_tokens,
        delay_seconds=args.delay_seconds,
    )["Main"].extract_topics(model, documents, model.c_tf_idf_, selected_topics)

    existing = {}
    if args.output.exists() and args.offset:
        existing = {int(row["topic_id"]): row for row in pq.read_table(args.output).to_pylist()}
    for row in topic_rows:
        topic_id = int(row["topic_id"])
        if topic_id not in labels:
            continue
        llm_label = str(labels[topic_id][0][0])
        existing[topic_id] = {
            **row,
            "cluster_label": row["label"],
            "llm_label": llm_label,
            "label": llm_label,
            "llm_model": args.model,
            "review_status": "unreviewed",
        }
    rows = [existing.get(int(row["topic_id"]), row) for row in topic_rows]
    write_parquet_rows(rows, args.output)
    if args.apply:
        write_parquet_rows(rows, args.input_dir / "topics.parquet")
        write_topic_frontend_json(
            rows,
            pq.read_table(args.input_dir / "episode_topics.parquet").to_pylist(),
            pq.read_table(args.input_dir / "episode_map.parquet").to_pylist(),
            pq.read_table(args.input_dir / "related_episodes.parquet").to_pylist(),
            args.input_dir / "topics.json",
        )
    logging.info(
        "Wrote labels for %d topics (%d–%d) to %s",
        len(selected_ids), selected_ids[0], selected_ids[-1], args.output,
    )


if __name__ == "__main__":
    main()
