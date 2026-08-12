"""Build semantic transcript chunks and an initial BERTopic model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.nlp.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROMPT,
    DEFAULT_EMBEDDING_BACKEND,
    EmbeddingEncoder,
)
from src.nlp.segment import segment_transcript
from src.nlp.topics import fit_topics, write_parquet_rows, write_topic_outputs
from src.nlp.transcripts import load_episode_metadata, load_transcripts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts-dir", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--episodes-path", type=Path, default=Path("data/episodes.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/nlp"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-prompt", default=DEFAULT_EMBEDDING_PROMPT)
    parser.add_argument(
        "--embedding-backend",
        choices=("mlx", "sentence-transformers"),
        default=DEFAULT_EMBEDDING_BACKEND,
    )
    parser.add_argument("--model-cache-dir", type=Path, default=Path("data/models/embeddings"))
    parser.add_argument("--device", default=None, help="Sentence Transformers device, e.g. cuda or mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=4,
        help="Smaller embedding batch for the much longer final chunks.",
    )
    parser.add_argument(
        "--chunk-embedding-strategy",
        choices=("unit-mean", "direct"),
        default="unit-mean",
        help="Use the mean of small unit embeddings (memory-safe) or re-embed full chunks.",
    )
    parser.add_argument("--mlx-cache-limit-mb", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Only use this many transcript files")
    parser.add_argument("--unit-words", type=int, default=80)
    parser.add_argument("--preserve-pause-seconds", type=float, default=1.5)
    parser.add_argument("--strong-pause-seconds", type=float, default=5.0)
    parser.add_argument("--semantic-window-units", type=int, default=3)
    parser.add_argument("--semantic-weight", type=float, default=0.82)
    parser.add_argument("--min-chunk-words", type=int, default=140)
    parser.add_argument("--target-chunk-words", type=int, default=450)
    parser.add_argument("--max-chunk-words", type=int, default=1800)
    parser.add_argument("--boundary-threshold", type=float, default=0.85)
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Write semantic chunks and boundary diagnostics without fitting BERTopic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    episodes = load_episode_metadata(args.episodes_path)
    transcripts = load_transcripts(
        args.transcripts_dir,
        unit_words=args.unit_words,
        preserve_pause_seconds=args.preserve_pause_seconds,
        limit=args.limit,
    )
    logging.info("Loaded %d complete transcripts", len(transcripts))
    if not transcripts:
        raise ValueError(f"No complete transcripts found in {args.transcripts_dir}")

    unit_rows = [unit.to_dict() for transcript in transcripts for unit in transcript.units]
    encoder = EmbeddingEncoder(
        model_name=args.embedding_model,
        model_cache_dir=args.model_cache_dir,
        device=args.device,
        batch_size=args.batch_size,
        prompt_name=args.embedding_prompt or None,
        backend=args.embedding_backend,
        max_length=args.embedding_max_tokens,
        mlx_cache_limit_mb=args.mlx_cache_limit_mb,
    )
    unit_embeddings = encoder.encode_cached(
        unit_rows,
        id_key="unit_id",
        text_key="text",
        cache_root=args.output_dir / "embedding_cache",
        cache_name="units",
        force=args.force_embeddings,
    )

    chunks: list[dict] = []
    chunk_embeddings: list = []
    boundary_rows: list[dict] = []
    offset = 0
    for transcript in transcripts:
        count = len(transcript.units)
        episode_chunks, episode_boundaries = segment_transcript(
            transcript,
            unit_embeddings[offset:offset + count],
            window_units=args.semantic_window_units,
            semantic_weight=args.semantic_weight,
            strong_pause_seconds=args.strong_pause_seconds,
            min_chunk_words=args.min_chunk_words,
            target_chunk_words=args.target_chunk_words,
            max_chunk_words=args.max_chunk_words,
            boundary_threshold=args.boundary_threshold,
        )
        offset += count
        metadata = episodes.get(transcript.sr_episode_id, {})
        shared = {
            "source_title": metadata.get("source_title") or transcript.source_title,
            "date": metadata.get("date"),
            "year": metadata.get("year"),
            "program_type": metadata.get("program_type"),
            "episode_speakers": metadata.get("episode_speakers") or [],
        }
        for chunk in episode_chunks:
            chunks.append({**chunk, **shared})
            if args.chunk_embedding_strategy == "unit-mean":
                unit_start = int(chunk["unit_start_index"])
                unit_end = int(chunk["unit_end_index"])
                start = offset - count + unit_start
                end = offset - count + unit_end
                weights = [unit.word_count for unit in transcript.units[unit_start:unit_end]]
                vector = np.average(
                    unit_embeddings[start:end],
                    axis=0,
                    weights=weights,
                )
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                chunk_embeddings.append(vector)
        boundary_rows.extend(episode_boundaries)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_rows(chunks, args.output_dir / "chunks.parquet")
    write_parquet_rows(boundary_rows, args.output_dir / "boundaries.parquet")
    logging.info(
        "Wrote %d semantic chunks and %d boundary diagnostics to %s",
        len(chunks),
        len(boundary_rows),
        args.output_dir,
    )
    if args.chunks_only:
        return

    if args.chunk_embedding_strategy == "direct":
        chunk_embeddings = encoder.encode_cached(
            chunks,
            id_key="chunk_id",
            text_key="text",
            cache_root=args.output_dir / "embedding_cache",
            cache_name="chunks",
            force=args.force_embeddings,
            batch_size=args.chunk_batch_size,
        )
    else:
        chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)
    model, topic_ids, probabilities = fit_topics(
        chunks,
        chunk_embeddings,
        random_state=args.random_state,
        min_cluster_size=args.min_cluster_size,
        min_df=args.min_df,
    )
    write_topic_outputs(
        output_dir=args.output_dir,
        chunks=chunks,
        embeddings=chunk_embeddings,
        model=model,
        topic_ids=topic_ids,
        probabilities=probabilities,
        random_state=args.random_state,
    )
    logging.info("Wrote topic model outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
