"""Fit BERTopic and write inspectable chunk- and episode-level outputs."""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.nlp.frontend import related_episode_rows, write_topic_frontend_json


DOMAIN_STOPWORDS = {
    # Spoken-language scaffolding that dominates personal narratives without
    # distinguishing their subject. This only affects c-TF-IDF labels; the
    # embedding input remains untouched.
    "absolut",
    "alltså",
    "alltid",
    "bara",
    "börja",
    "började",
    "börjar",
    "dag",
    "egentligen",
    "faktiskt",
    "fick",
    "först",
    "förstås",
    "gång",
    "ganska",
    "gick",
    "gör",
    "göra",
    "hela",
    "helt",
    "idag",
    "ibland",
    "kanske",
    "känner",
    "kom",
    "kommer",
    "liksom",
    "länge",
    "många",
    "måste",
    "nog",
    "precis",
    "redan",
    "riktigt",
    "saker",
    "satt",
    "sedan",
    "sen",
    "ser",
    "sist",
    "sitter",
    "själv",
    "slags",
    "säger",
    "sätt",
    "tid",
    "tillbaka",
    "tog",
    "tror",
    "tycker",
    "tänker",
    "tänkte",
    "verkligen",
    "vet",
    "ville",
    "väldigt",
    "år",
    "ändå",
    # Programme framing and recurrent transitions.
    "program",
    "programmet",
    "producent",
    "sommari",
    "sommarprogram",
    "app",
    "play",
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

# Some old episode copies contain app/player and archival framing instead of
# programme content. HDBSCAN groups these very reliably, so flag such clusters
# rather than displaying them as an invented subject in the frontend.
ARCHIVE_BOILERPLATE_TERMS = {
    "sr",
    "sverigesradio",
    "podd",
    "lista",
    "tekniker",
    "hemsidan",
    "radioprogram",
}


def _is_low_quality_topic(
    weighted_terms: Sequence[tuple[str, float]],
    representative_documents: Sequence[str],
) -> bool:
    terms = {
        word.lower()
        for term, _ in weighted_terms[:8]
        for word in term.replace("_", " ").split()
    }
    # A single generic word such as "podd" is not enough. The archival
    # clusters consistently contain several of these terms, unlike real
    # programme subjects.
    if len(terms & ARCHIVE_BOILERPLATE_TERMS) >= 3:
        return True
    return False


def build_openai_representation(
    *,
    model: str,
    document_tokens: int = 120,
    delay_seconds: float = 0.25,
) -> dict[str, Any]:
    """Create BERTopic's documented c-TF-IDF-to-OpenAI representation step."""
    import os

    import tiktoken
    from bertopic.representation import OpenAI as OpenAIRepresentation
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for --llm-label-model.")
    # GPT-5 family models use the o200k vocabulary. Passing the tokenizer lets
    # BERTopic truncate each c-TF-IDF-selected document by actual token count.
    tokenizer = tiktoken.get_encoding("o200k_base")
    prompt = """Jag granskar ett automatiskt textkluster från svenska radioprogram.

Representativa utdrag:
[DOCUMENTS]

Nyckelord: [KEYWORDS]

Ge en kort svensk etikett på 2–6 ord. Hitta inte på detaljer. Använd inte ord
som "radioprat", "Sommarprat" eller "Vinterprat" för vanliga innehållsämnen:
de beskriver mediet och inte ämnet. Om underlaget mest är sångtext,
programframing eller trasig transkription, namnge i stället det mönstret
ärligt. Svara enbart i exakt detta format:
topic: <etikett>
"""
    llm_representation = OpenAIRepresentation(
        OpenAI(),
        model=model,
        prompt=prompt,
        system_prompt="Du är en noggrann svensk redaktör som namnger textkluster.",
        generator_kwargs={
            "reasoning_effort": "none",
            "temperature": 0,
            "max_completion_tokens": 32,
        },
        delay_in_seconds=delay_seconds,
        nr_docs=4,
        diversity=0.1,
        doc_length=document_tokens,
        tokenizer=tokenizer,
    )
    # BERTopic's generic OpenAI wrapper defaults to ``stop="\n"``. GPT-5.6
    # rejects the legacy stop parameter, and our explicit ``topic:`` format
    # already makes a stop sequence unnecessary.
    llm_representation.generator_kwargs.pop("stop", None)
    # Preserve raw c-TF-IDF keywords as an aspect alongside the generated label.
    return {"Main": llm_representation, "c-TF-IDF": None}


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
    clusterer: str = "kmeans",
    n_topics: int = 80,
    min_cluster_size: int | None = None,
    min_samples: int = 5,
    umap_neighbors: int = 15,
    umap_components: int = 5,
    min_df: int = 3,
    llm_label_model: str | None = None,
    llm_document_tokens: int = 120,
    llm_delay_seconds: float = 0.25,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit a reproducible BERTopic model over timestamped chunks.

    HDBSCAN is the discovery default: it finds dense semantic groups and keeps
    ambiguous chunks as outliers. K-means remains available for controlled,
    fixed-vocabulary experiments after the corpus has been reviewed.
    """
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import CountVectorizer
    from stopwordsiso import stopwords
    from umap import UMAP

    representation_model: Any = (
        build_openai_representation(
            model=llm_label_model,
            document_tokens=llm_document_tokens,
            delay_seconds=llm_delay_seconds,
        )
        if llm_label_model
        else None
    )

    documents = [str(chunk["text"]) for chunk in chunks]
    if len(documents) < 10:
        raise ValueError("At least 10 chunks are required to fit a topic model.")
    if clusterer not in {"kmeans", "hdbscan"}:
        raise ValueError(f"Unsupported clusterer: {clusterer}")
    if clusterer == "kmeans" and not 2 <= n_topics <= len(documents):
        raise ValueError(f"n_topics must be between 2 and {len(documents)}")

    cluster_size = min_cluster_size or 15

    vectorizer = CountVectorizer(
        stop_words=sorted(set(stopwords("sv")) | DOMAIN_STOPWORDS),
        # Tiny smoke tests may collapse to one topic document. The full corpus
        # still uses the requested minimum document frequency.
        min_df=1 if len(documents) < 100 else min_df,
        ngram_range=(1, 2),
    )
    if clusterer == "kmeans":
        # Unit-normalised embeddings make Euclidean K-means equivalent to
        # spherical/cosine K-means for assigning points to centroids. Keep the
        # original space: UMAP is useful for display, but distorts clusters.
        logging.info("Fitting BERTopic to %d chunks (spherical K-means, k=%d)", len(documents), n_topics)
        umap_model: Any = BaseDimensionalityReduction()
        cluster_model: Any = KMeans(
            n_clusters=n_topics,
            init="k-means++",
            n_init=10,
            random_state=random_state,
        )
    else:
        neighbours = min(umap_neighbors, len(documents) - 1)
        logging.info(
            "Fitting BERTopic to %d chunks (HDBSCAN min_cluster_size=%d)",
            len(documents),
            cluster_size,
        )
        umap_model = UMAP(
            n_neighbors=neighbours,
            n_components=umap_components,
            min_dist=0.0,
            metric="cosine",
            random_state=random_state,
            low_memory=True,
        )
        cluster_model = HDBSCAN(
            min_cluster_size=cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
    model = BERTopic(
        language="multilingual",
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(
            bm25_weighting=True,
            reduce_frequent_words=True,
        ),
        representation_model=representation_model,
        top_n_words=12,
        calculate_probabilities=False,
        verbose=True,
    )
    topic_ids, probabilities = model.fit_transform(documents, embeddings)
    if probabilities is None:
        # K-means has no density-based membership probability. Every assigned
        # chunk is retained; callers can use its topic share within an episode.
        probabilities = np.ones(len(topic_ids), dtype=np.float32)
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
        raw_topics = model.topic_aspects_.get("c-TF-IDF", {})
        weighted_terms = raw_topics.get(topic_id) or model.get_topic(topic_id) or []
        primary_representation = model.get_topic(topic_id) or []
        llm_label = (
            str(primary_representation[0][0])
            if raw_topics and primary_representation
            else None
        )
        representative_documents = model.get_representative_docs(topic_id) or []
        rows.append(
            {
                "topic_id": topic_id,
                "label": llm_label or info.get("Name"),
                "cluster_label": info.get("Name"),
                "llm_label": llm_label,
                "chunk_count": int(info.get("Count", 0)),
                "episode_count": len(episodes_by_topic[topic_id]),
                "keywords": [term for term, _ in weighted_terms],
                "keyword_scores": [float(score) for _, score in weighted_terms],
                "representative_excerpts": [str(text)[:600] for text in representative_documents[:3]],
                "is_outlier": topic_id == -1,
                "is_low_quality": topic_id != -1 and _is_low_quality_topic(
                    weighted_terms,
                    representative_documents,
                ),
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
        content_topic_counts = {
            topic_id: count for topic_id, count in topic_counts.items() if topic_id != -1
        }
        dominant_topic = max(content_topic_counts or topic_counts, key=(content_topic_counts or topic_counts).get)
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
    related_episodes = related_episode_rows(
        episode_embeddings,
        episode_ids,
        top_k=8,
    )

    write_parquet_rows(chunk_rows, output_dir / "chunk_topics.parquet")
    write_parquet_rows(topics, output_dir / "topics.parquet")
    write_parquet_rows(episode_topics, output_dir / "episode_topics.parquet")
    write_parquet_rows(episode_map, output_dir / "episode_map.parquet")
    write_parquet_rows(related_episodes, output_dir / "related_episodes.parquet")
    write_topic_frontend_json(
        topics,
        episode_topics,
        episode_map,
        related_episodes,
        output_dir / "topics.json",
    )
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
