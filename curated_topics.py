"""Assign the editorial topic catalogue to semantic transcript chunks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.nlp.curated import (
    aggregate_episode_topics,
    apply_topic_guidance,
    assign_chunk_topics,
    parse_curated_topic_proposal,
    sample_topic_calibration_candidates,
)
from src.nlp.embeddings import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingEncoder,
    load_matching_cached_embeddings,
)
from src.nlp.topics import write_parquet_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=Path("SPECIFIC_TOPICS_PROPOSAL.md"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/nlp"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/nlp/curated"))
    parser.add_argument(
        "--guidance",
        type=Path,
        default=Path("data/nlp/curated/topic_guidance.json"),
        help="Optional Swedish definitions and positive/negative examples.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-backend",
        choices=("mlx", "sentence-transformers"),
        default=DEFAULT_EMBEDDING_BACKEND,
    )
    parser.add_argument("--embedding-max-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mlx-cache-limit-mb", type=int, default=512)
    parser.add_argument("--min-similarity", type=float, default=0.51)
    parser.add_argument(
        "--topic-thresholds",
        type=Path,
        default=Path("data/nlp/curated/topic_thresholds.json"),
        help="Optional reviewed per-topic minimum similarities.",
    )
    parser.add_argument("--min-winner-margin", type=float, default=0.01)
    parser.add_argument(
        "--accept-ambiguous-above",
        type=float,
        default=0.56,
        help="Accept a strong best match even when its runner-up is close.",
    )
    parser.add_argument("--supporting-min-similarity", type=float, default=0.495)
    parser.add_argument("--supporting-min-winner-margin", type=float, default=0.008)
    parser.add_argument(
        "--min-negative-margin",
        type=float,
        help="Optional topic-minus-negative veto; disabled by default.",
    )
    parser.add_argument(
        "--secondary-min-similarity",
        type=float,
        help="Enable second chunk labels with this stricter similarity floor.",
    )
    parser.add_argument("--secondary-max-score-gap", type=float, default=0.015)
    parser.add_argument("--min-coverage", type=float, default=0.03)
    parser.add_argument("--min-evidence-words", type=int, default=140)
    parser.add_argument("--single-chunk-min-similarity", type=float, default=0.515)
    parser.add_argument("--min-supporting-chunks", type=int, default=2)
    parser.add_argument("--supporting-min-coverage", type=float, default=0.08)
    parser.add_argument("--supporting-min-evidence-words", type=int, default=250)
    parser.add_argument(
        "--cached-chunks-only",
        action="store_true",
        help="Calibrate on matching cached direct chunk vectors without embedding missing chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    topics = parse_curated_topic_proposal(args.proposal)
    if args.guidance.exists():
        topics = apply_topic_guidance(topics, args.guidance)
        logging.info("Loaded reviewed topic guidance from %s", args.guidance)
    chunks = pq.read_table(args.input_dir / "chunks.parquet").to_pylist()
    cache_root = args.input_dir / "embedding_cache"

    if args.cached_chunks_only:
        chunks, chunk_embeddings = load_matching_cached_embeddings(
            chunks,
            id_key="chunk_id",
            cache_root=cache_root,
            cache_name="chunks",
            backend=args.embedding_backend,
            model_name=args.embedding_model,
            prompt_name=None,
            max_length=args.embedding_max_tokens,
        )
        if not chunks:
            raise ValueError("No current chunks have matching cached direct embeddings")
        logging.info("Using %d cached current chunks for calibration", len(chunks))
    else:
        document_encoder = EmbeddingEncoder(
            model_name=args.embedding_model,
            backend=args.embedding_backend,
            prompt_name=None,
            batch_size=args.batch_size,
            max_length=args.embedding_max_tokens,
            mlx_cache_limit_mb=args.mlx_cache_limit_mb,
        )
        chunk_embeddings = document_encoder.encode_cached(
            chunks,
            id_key="chunk_id",
            text_key="text",
            cache_root=cache_root,
            cache_name="chunks",
            batch_size=args.batch_size,
        )

    topic_records = [topic.to_dict() for topic in topics]
    query_encoder: EmbeddingEncoder | None = None
    cached_topic_records, topic_embeddings = load_matching_cached_embeddings(
        topic_records,
        id_key="topic_id",
        cache_root=args.output_dir / "embedding_cache",
        cache_name="topics",
        backend=args.embedding_backend,
        model_name=args.embedding_model,
        prompt_name="sts_query",
        max_length=args.embedding_max_tokens,
    )
    if len(cached_topic_records) != len(topic_records):
        query_encoder = EmbeddingEncoder(
            model_name=args.embedding_model,
            backend=args.embedding_backend,
            prompt_name="sts_query",
            batch_size=max(args.batch_size, 8),
            max_length=args.embedding_max_tokens,
            mlx_cache_limit_mb=args.mlx_cache_limit_mb,
        )
        topic_embeddings = query_encoder.encode_cached(
            topic_records,
            id_key="topic_id",
            text_key="query_text",
            cache_root=args.output_dir / "embedding_cache",
            cache_name="topics",
            batch_size=max(args.batch_size, 8),
        )
    else:
        logging.info("Reused all %d cached curated topic embeddings", len(topic_records))
    negative_records = [
        {
            "example_id": f"{topic.topic_id}:negative:{index}",
            "query_text": f"Texten handlar huvudsakligen om {example}.",
        }
        for topic in topics
        for index, example in enumerate(topic.negative_examples)
    ]
    if any(len(topic.negative_examples) != 2 for topic in topics):
        raise ValueError("Every curated topic must have exactly two negative examples")
    cached_negative_records, negative_embeddings = load_matching_cached_embeddings(
        negative_records,
        id_key="example_id",
        cache_root=args.output_dir / "embedding_cache",
        cache_name="negative_examples",
        backend=args.embedding_backend,
        model_name=args.embedding_model,
        prompt_name="sts_query",
        max_length=args.embedding_max_tokens,
    )
    if len(cached_negative_records) != len(negative_records):
        if query_encoder is None:
            query_encoder = EmbeddingEncoder(
                model_name=args.embedding_model,
                backend=args.embedding_backend,
                prompt_name="sts_query",
                batch_size=max(args.batch_size, 8),
                max_length=args.embedding_max_tokens,
                mlx_cache_limit_mb=args.mlx_cache_limit_mb,
            )
        negative_embeddings = query_encoder.encode_cached(
            negative_records,
            id_key="example_id",
            text_key="query_text",
            cache_root=args.output_dir / "embedding_cache",
            cache_name="negative_examples",
            batch_size=max(args.batch_size, 8),
        )
    else:
        logging.info("Reused all %d cached negative-example embeddings", len(negative_records))
    negative_embeddings = negative_embeddings.reshape(len(topics), 2, -1)
    topic_thresholds: dict[str, float] = {}
    if args.topic_thresholds.exists():
        threshold_payload = json.loads(args.topic_thresholds.read_text(encoding="utf-8"))
        topic_thresholds = {
            str(topic_id): float(value)
            for topic_id, value in threshold_payload.get("topic_min_similarities", {}).items()
        }
        logging.info("Loaded %d reviewed topic thresholds", len(topic_thresholds))
    chunk_assignments, chunk_decisions = assign_chunk_topics(
        chunks,
        chunk_embeddings,
        topics,
        topic_embeddings,
        min_similarity=args.min_similarity,
        topic_min_similarities=topic_thresholds,
        min_winner_margin=args.min_winner_margin,
        accept_ambiguous_above=args.accept_ambiguous_above,
        supporting_min_similarity=args.supporting_min_similarity,
        supporting_min_winner_margin=args.supporting_min_winner_margin,
        negative_embeddings=negative_embeddings,
        min_negative_margin=args.min_negative_margin,
        secondary_min_similarity=args.secondary_min_similarity,
        secondary_max_score_gap=args.secondary_max_score_gap,
    )
    episode_topics = aggregate_episode_topics(
        chunks,
        chunk_assignments,
        min_coverage=args.min_coverage,
        min_evidence_words=args.min_evidence_words,
        single_chunk_min_similarity=args.single_chunk_min_similarity,
        min_supporting_chunks=args.min_supporting_chunks,
        supporting_min_coverage=args.supporting_min_coverage,
        supporting_min_evidence_words=args.supporting_min_evidence_words,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_rows(topic_records, args.output_dir / "topics.parquet")
    write_parquet_rows(chunk_assignments, args.output_dir / "chunk_topics.parquet")
    write_parquet_rows(chunk_decisions, args.output_dir / "chunk_topic_decisions.parquet")
    write_parquet_rows(
        sample_topic_calibration_candidates(chunk_decisions),
        args.output_dir / "topic_calibration_samples.parquet",
    )
    write_parquet_rows(episode_topics, args.output_dir / "episode_topics.parquet")
    np.save(args.output_dir / "topic_embeddings.npy", topic_embeddings.astype(np.float16))
    logging.info(
        "Wrote %d topics, %d/%d accepted chunks, and %d episode assignments to %s",
        len(topics),
        len(chunk_assignments),
        len(chunk_decisions),
        len(episode_topics),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
