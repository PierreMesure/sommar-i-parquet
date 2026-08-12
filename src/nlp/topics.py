"""Fit BERTopic and write inspectable chunk- and episode-level outputs."""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DOMAIN_STOPWORDS = {
    "program",
    "programmet",
    "radio",
    "sveriges",
    "sommar",
    "sommarprat",
    "sommarpratare",
    "sommarvärd",
    "vinter",
    "vinterprat",
    "vintervärd",
    "lyssna",
    "lyssnar",
    "lyssnare",
    "musik",
    "låten",
    "låt",
    "p1",
}


def write_parquet_rows(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    """Write inferred Arrow rows with Zstandard compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path, compression="zstd")
    return path


def fit_topics(
    chunks: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    random_state: int = 42,
    min_cluster_size: int | None = None,
    min_df: int = 3,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit a reproducible BERTopic model over timestamped chunks."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from stopwordsiso import stopwords
    from umap import UMAP

    documents = [str(chunk["text"]) for chunk in chunks]
    if len(documents) < 10:
        raise ValueError("At least 10 chunks are required to fit a topic model.")
    cluster_size = min_cluster_size or max(12, min(80, round(len(documents) * 0.006)))
    neighbours = min(15, len(documents) - 1)
    logging.info(
        "Fitting BERTopic to %d chunks (min_cluster_size=%d)",
        len(documents),
        cluster_size,
    )

    vectorizer = CountVectorizer(
        stop_words=sorted(set(stopwords("sv")) | DOMAIN_STOPWORDS),
        # Tiny smoke tests may collapse to one topic document. The full corpus
        # still uses the requested minimum document frequency.
        min_df=1 if len(documents) < 100 else min_df,
        ngram_range=(1, 2),
    )
    umap_model = UMAP(
        n_neighbors=neighbours,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cluster_size,
        min_samples=max(5, cluster_size // 3),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    model = BERTopic(
        language="multilingual",
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        top_n_words=12,
        calculate_probabilities=False,
        verbose=True,
    )
    topic_ids, probabilities = model.fit_transform(documents, embeddings)
    return (
        model,
        np.asarray(topic_ids, dtype=np.int32),
        np.asarray(probabilities, dtype=np.float32),
    )


def _topic_rows(
    model: Any,
    chunks: Sequence[dict[str, Any]],
    topic_ids: np.ndarray,
) -> list[dict[str, Any]]:
    episodes_by_topic: dict[int, set[int]] = defaultdict(set)
    for chunk, topic_id in zip(chunks, topic_ids, strict=True):
        episodes_by_topic[int(topic_id)].add(int(chunk["sr_episode_id"]))

    information = {
        int(row["Topic"]): row
        for row in model.get_topic_info().to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    for topic_id, info in sorted(information.items()):
        weighted_terms = model.get_topic(topic_id) or []
        representative_documents = model.get_representative_docs(topic_id) or []
        rows.append(
            {
                "topic_id": topic_id,
                "label": info.get("Name"),
                "chunk_count": int(info.get("Count", 0)),
                "episode_count": len(episodes_by_topic[topic_id]),
                "keywords": [term for term, _ in weighted_terms],
                "keyword_scores": [float(score) for _, score in weighted_terms],
                "representative_excerpts": [str(text)[:600] for text in representative_documents[:3]],
                "is_outlier": topic_id == -1,
            }
        )
    return rows


def _episode_topic_rows(
    chunks: Sequence[dict[str, Any]],
    topic_ids: np.ndarray,
) -> list[dict[str, Any]]:
    totals: dict[int, int] = defaultdict(int)
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for chunk, topic_id_value in zip(chunks, topic_ids, strict=True):
        episode_id = int(chunk["sr_episode_id"])
        topic_id = int(topic_id_value)
        words = int(chunk["word_count"])
        totals[episode_id] += words
        key = (episode_id, topic_id)
        record = grouped.setdefault(
            key,
            {
                "sr_episode_id": episode_id,
                "topic_id": topic_id,
                "chunk_count": 0,
                "word_count": 0,
                "representative_chunk_id": chunk["chunk_id"],
                "representative_start_seconds": chunk["start_seconds"],
                "representative_excerpt": str(chunk["text"])[:500],
                "source_title": chunk.get("source_title"),
            },
        )
        record["chunk_count"] += 1
        record["word_count"] += words

    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in grouped.values():
        episode_id = int(record["sr_episode_id"])
        record["share"] = record["word_count"] / max(totals[episode_id], 1)
        by_episode[episode_id].append(record)

    rows: list[dict[str, Any]] = []
    for episode_id, records in by_episode.items():
        records.sort(key=lambda row: (-row["share"], row["topic_id"]))
        for rank, record in enumerate(records, start=1):
            rows.append({**record, "rank": rank})
    return rows


def _episode_map_rows(
    chunks: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    topic_ids: np.ndarray,
    *,
    random_state: int,
) -> tuple[list[dict[str, Any]], np.ndarray, list[int]]:
    from umap import UMAP

    positions: dict[int, list[int]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        positions[int(chunk["sr_episode_id"])].append(index)
    episode_ids = sorted(positions)
    episode_embeddings = np.stack(
        [embeddings[positions[episode_id]].mean(axis=0) for episode_id in episode_ids]
    ).astype(np.float32)
    norms = np.linalg.norm(episode_embeddings, axis=1, keepdims=True)
    episode_embeddings /= np.maximum(norms, 1e-12)

    if len(episode_ids) >= 3:
        coordinates = UMAP(
            n_neighbors=min(15, len(episode_ids) - 1),
            n_components=2,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
            low_memory=True,
        ).fit_transform(episode_embeddings)
    else:
        coordinates = np.column_stack(
            [np.arange(len(episode_ids), dtype=np.float32), np.zeros(len(episode_ids), dtype=np.float32)]
        )

    rows: list[dict[str, Any]] = []
    for row_index, episode_id in enumerate(episode_ids):
        indices = positions[episode_id]
        topic_counts: dict[int, int] = defaultdict(int)
        for chunk_index in indices:
            topic_counts[int(topic_ids[chunk_index])] += int(chunks[chunk_index]["word_count"])
        dominant_topic = max(topic_counts, key=topic_counts.get)
        rows.append(
            {
                "sr_episode_id": episode_id,
                "x": float(coordinates[row_index, 0]),
                "y": float(coordinates[row_index, 1]),
                "dominant_topic_id": dominant_topic,
                "source_title": chunks[indices[0]].get("source_title"),
                "year": chunks[indices[0]].get("year"),
                "program_type": chunks[indices[0]].get("program_type"),
            }
        )
    return rows, episode_embeddings, episode_ids


def write_topic_outputs(
    *,
    output_dir: Path,
    chunks: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    model: Any,
    topic_ids: np.ndarray,
    probabilities: np.ndarray,
    random_state: int = 42,
) -> None:
    """Persist model assignments, summaries, and an episode-level 2D map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_rows = [
        {
            **chunk,
            "topic_id": int(topic_id),
            "topic_probability": float(probability),
        }
        for chunk, topic_id, probability in zip(chunks, topic_ids, probabilities, strict=True)
    ]
    topics = _topic_rows(model, chunks, topic_ids)
    episode_topics = _episode_topic_rows(chunks, topic_ids)
    episode_map, episode_embeddings, episode_ids = _episode_map_rows(
        chunks,
        embeddings,
        topic_ids,
        random_state=random_state,
    )

    write_parquet_rows(chunk_rows, output_dir / "chunk_topics.parquet")
    write_parquet_rows(topics, output_dir / "topics.parquet")
    write_parquet_rows(episode_topics, output_dir / "episode_topics.parquet")
    write_parquet_rows(episode_map, output_dir / "episode_map.parquet")
    np.save(output_dir / "episode_embeddings.npy", episode_embeddings.astype(np.float16))
    (output_dir / "episode_embedding_ids.txt").write_text(
        "\n".join(str(episode_id) for episode_id in episode_ids) + "\n",
        encoding="utf-8",
    )
    model.save(output_dir / "bertopic_model", serialization="safetensors", save_ctfidf=True)

    topic_by_id = {int(topic["topic_id"]): topic for topic in topics}
    episode_titles: dict[int, str] = {
        int(chunk["sr_episode_id"]): str(chunk.get("source_title") or chunk["sr_episode_id"])
        for chunk in chunks
    }
    episodes_by_topic: dict[int, list[int]] = defaultdict(list)
    for row in episode_topics:
        if row["rank"] <= 3:
            episodes_by_topic[int(row["topic_id"])].append(int(row["sr_episode_id"]))
    lines = ["# Initial transcript topics", ""]
    ordered_topics = sorted(
        (topic for topic in topics if topic["topic_id"] != -1),
        key=lambda topic: -topic["chunk_count"],
    )
    for topic in ordered_topics:
        topic_id = int(topic["topic_id"])
        lines.append(f"## {topic_id}: {', '.join(topic['keywords'][:6])}")
        lines.append("")
        lines.append(
            f"{topic['chunk_count']} chunks across {topic['episode_count']} episodes."
        )
        examples = list(dict.fromkeys(episodes_by_topic[topic_id]))[:5]
        if examples:
            lines.append("")
            lines.append("Examples: " + "; ".join(episode_titles[item] for item in examples))
        representative = topic_by_id[topic_id]["representative_excerpts"]
        if representative:
            lines.append("")
            lines.append(f"> {representative[0]}")
        lines.append("")
    (output_dir / "topic_report.md").write_text("\n".join(lines), encoding="utf-8")
