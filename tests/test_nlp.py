import json
from pathlib import Path

import numpy as np
import pytest

from src.nlp.curated import (
    aggregate_episode_topics,
    assign_chunk_topics,
    estimate_topic_threshold,
    parse_curated_topic_proposal,
    sample_topic_calibration_candidates,
)
from src.nlp.manual_hierarchy import (
    EXCLUDED,
    SPECIFIC_TOPICS,
    validate as validate_manual_hierarchy,
)
from src.nlp.segment import score_boundaries, segment_transcript
from src.nlp.transcripts import EpisodeTranscript, TranscriptUnit, build_units, load_transcripts


def test_build_units_preserves_a_long_pause_as_a_candidate_boundary():
    units = build_units(
        episode_id=1,
        target_words=100,
        min_words=2,
        preserve_pause_seconds=1.5,
        segments=[
            {"start": 0.0, "end": 2.0, "text": "Första delen börjar här."},
            {"start": 2.1, "end": 4.0, "text": "Den fortsätter en stund."},
            {"start": 12.0, "end": 14.0, "text": "Ett nytt kapitel börjar."},
        ],
    )

    assert len(units) == 2
    assert units[1].pause_before_seconds == 8.0
    assert units[1].text == "Ett nytt kapitel börjar."


def test_loading_transcripts_excludes_detected_boilerplate_in_memory(tmp_path):
    path = tmp_path / "123.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0,
                        "end": 4,
                        "text": "Det här är poddversionen av Sommar i P1.",
                    },
                    {
                        "start": 20,
                        "end": 30,
                        "text": "Min berättelse börjar i Göteborg och fortsätter här.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    transcripts = load_transcripts(tmp_path, unit_words=10)

    assert len(transcripts) == 1
    assert transcripts[0].units[0].text == "Min berättelse börjar i Göteborg och fortsätter här."
    # The loader performs a non-destructive view; the source remains raw.
    assert "poddversionen" in path.read_text(encoding="utf-8")


def _transcript_with_six_units() -> EpisodeTranscript:
    units = tuple(
        TranscriptUnit(
            unit_id=f"1:u{index}",
            sr_episode_id=1,
            unit_index=index,
            start_seconds=float(index * 30 + (20 if index == 3 else 0)),
            end_seconds=float(index * 30 + (20 if index == 3 else 0) + 25),
            pause_before_seconds=20.0 if index == 3 else 0.2,
            text=("familj barndom " if index < 3 else "forskning rymden ") * 50,
            word_count=100,
            segment_count=2,
        )
        for index in range(6)
    )
    return EpisodeTranscript(
        sr_episode_id=1,
        source_title="Testavsnitt",
        transcript_path="transcript.json",
        transcript_engine="test",
        transcript_model="test-model",
        units=units,
    )


def test_semantic_chunking_selects_music_break_between_distinct_contexts():
    transcript = _transcript_with_six_units()
    embeddings = np.asarray(
        [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3,
        dtype=np.float32,
    )

    chunks, boundaries = segment_transcript(
        transcript,
        embeddings,
        min_chunk_words=150,
        target_chunk_words=300,
        max_chunk_words=500,
        boundary_threshold=0.85,
    )

    assert [chunk["word_count"] for chunk in chunks] == [300, 300]
    selected = [row for row in boundaries if row["selected"]]
    assert len(selected) == 1
    assert selected[0]["boundary_index"] == 3
    assert selected[0]["likely_music_break"] is True
    assert selected[0]["selection_reason"] == "music_break"


def test_boundary_scores_use_rolling_context_not_only_adjacent_units():
    transcript = _transcript_with_six_units()
    embeddings = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    boundaries = score_boundaries(transcript, embeddings, window_units=2)

    strongest = max(boundaries, key=lambda boundary: boundary.semantic_distance)
    assert strongest.boundary_index == 3
    assert strongest.semantic_distance > 0.5


def test_curated_topic_proposal_parser_stops_before_rejected_clusters(tmp_path: Path):
    proposal = tmp_path / "topics.md"
    proposal.write_text(
        "\n".join(
            [
                "# Topics",
                "- **Adoption och internationella adoptioner** — adopting and being adopted. _(raw: 3, 185)_",
                "## Raw clusters intentionally not promoted",
                "- **Generic personal narrative:** 0, 2.",
            ]
        ),
        encoding="utf-8",
    )

    topics = parse_curated_topic_proposal(proposal)

    assert len(topics) == 1
    assert topics[0].topic_id == "adoption-och-internationella-adoptioner"
    assert topics[0].raw_topic_ids == (3, 185)


def test_curated_assignment_uses_floor_gap_and_episode_coverage():
    proposal = Path("SPECIFIC_TOPICS_PROPOSAL.md")
    topics = parse_curated_topic_proposal(proposal)[:3]
    chunks = [
        {
            "chunk_id": "1:c0",
            "sr_episode_id": 1,
            "word_count": 200,
            "start_seconds": 0.0,
            "end_seconds": 60.0,
            "text": "Ett tydligt ämnesstycke",
        },
        {
            "chunk_id": "1:c1",
            "sr_episode_id": 1,
            "word_count": 800,
            "start_seconds": 60.0,
            "end_seconds": 300.0,
            "text": "Ett annat stycke",
        },
    ]
    chunk_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    topic_embeddings = np.asarray(
        [[1.0, 0.0], [0.96, 0.28], [-1.0, 0.0]],
        dtype=np.float32,
    )
    topic_embeddings /= np.linalg.norm(topic_embeddings, axis=1, keepdims=True)

    assignments, decisions = assign_chunk_topics(
        chunks,
        chunk_embeddings,
        topics,
        topic_embeddings,
        min_similarity=0.5,
        min_winner_margin=0.02,
        accept_ambiguous_above=None,
        supporting_min_similarity=None,
        secondary_min_similarity=0.5,
        secondary_max_score_gap=0.05,
    )
    episode_topics = aggregate_episode_topics(
        chunks,
        assignments,
        min_coverage=0.1,
        min_evidence_words=100,
    )

    assert [row["topic_id"] for row in assignments] == [
        topics[0].topic_id,
        topics[1].topic_id,
    ]
    assert decisions[0]["accepted"] is True
    assert decisions[1]["rejection_reason"] == "below_topic_threshold"
    assert len(episode_topics) == 2
    assert all(row["coverage"] == 0.2 for row in episode_topics)


def test_curated_assignment_rejects_an_ambiguous_winner():
    topics = parse_curated_topic_proposal(Path("SPECIFIC_TOPICS_PROPOSAL.md"))[:2]
    chunks = [
        {
            "chunk_id": "1:c0",
            "sr_episode_id": 1,
            "word_count": 200,
            "start_seconds": 0.0,
            "end_seconds": 60.0,
            "text": "Ett tvetydigt stycke",
        }
    ]
    chunk_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    topic_embeddings = np.asarray([[0.8, 0.6], [0.79, 0.613]], dtype=np.float32)
    topic_embeddings /= np.linalg.norm(topic_embeddings, axis=1, keepdims=True)

    assignments, decisions = assign_chunk_topics(
        chunks,
        chunk_embeddings,
        topics,
        topic_embeddings,
        min_similarity=0.5,
        min_winner_margin=0.02,
        accept_ambiguous_above=None,
        supporting_min_similarity=None,
    )

    assert assignments == []
    assert decisions[0]["accepted"] is False
    assert decisions[0]["rejection_reason"] == "ambiguous_winner"


def test_curated_assignment_rejects_a_negative_near_miss():
    topics = parse_curated_topic_proposal(Path("SPECIFIC_TOPICS_PROPOSAL.md"))[:2]
    chunks = [
        {
            "chunk_id": "1:c0",
            "sr_episode_id": 1,
            "word_count": 200,
            "start_seconds": 0.0,
            "end_seconds": 60.0,
            "text": "Ett närliggande men uttryckligen uteslutet stycke",
        }
    ]
    chunk_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    topic_embeddings = np.asarray([[0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
    negative_embeddings = np.asarray(
        [
            [[0.9, np.sqrt(0.19)], [0.1, np.sqrt(0.99)]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    assignments, decisions = assign_chunk_topics(
        chunks,
        chunk_embeddings,
        topics,
        topic_embeddings,
        min_similarity=0.5,
        min_winner_margin=0.05,
        accept_ambiguous_above=None,
        supporting_min_similarity=None,
        negative_embeddings=negative_embeddings,
        min_negative_margin=-0.02,
    )

    assert assignments == []
    assert decisions[0]["negative_similarity"] == pytest.approx(0.9)
    assert decisions[0]["rejection_reason"] == "negative_near_miss"


def test_curated_calibration_sampling_and_threshold_estimation():
    decisions = [
        {
            "chunk_id": f"1:c{index}",
            "best_topic_id": "topic-a",
            "best_similarity": score,
        }
        for index, score in enumerate([0.49, 0.495, 0.51, 0.515, 0.54, 0.55])
    ]
    sampled = sample_topic_calibration_candidates(decisions, per_score_band=1)

    assert {row["chunk_id"] for row in sampled} == {"1:c0", "1:c2", "1:c4"}
    threshold, diagnostics = estimate_topic_threshold(
        [
            {"best_similarity": 0.49, "relevant": False},
            {"best_similarity": 0.51, "relevant": False},
            {"best_similarity": 0.54, "relevant": True},
            {"best_similarity": 0.55, "relevant": True},
        ],
        default_threshold=0.53,
        target_precision=0.9,
    )

    assert threshold == 0.54
    assert diagnostics["source"] == "reviewed_boundary"


def test_repeated_supporting_chunks_can_establish_an_episode_topic():
    topic_id = "topic-a"
    chunks = [
        {"sr_episode_id": 1, "word_count": 300},
        {"sr_episode_id": 1, "word_count": 300},
        {"sr_episode_id": 1, "word_count": 400},
    ]
    assignments = [
        {
            "sr_episode_id": 1,
            "topic_id": topic_id,
            "topic_label": "Topic A",
            "word_count": 300,
            "similarity": 0.51,
            "evidence_tier": "supporting",
            "excerpt": "Första belägget",
        },
        {
            "sr_episode_id": 1,
            "topic_id": topic_id,
            "topic_label": "Topic A",
            "word_count": 300,
            "similarity": 0.52,
            "evidence_tier": "supporting",
            "excerpt": "Andra belägget",
        },
    ]

    episode_topics = aggregate_episode_topics(chunks, assignments)

    assert len(episode_topics) == 1
    assert episode_topics[0]["strong_chunk_count"] == 0
    assert episode_topics[0]["coverage"] == 0.6


def test_a_borderline_single_chunk_cannot_establish_an_episode_topic():
    chunks = [
        {"sr_episode_id": 1, "word_count": 300},
        {"sr_episode_id": 1, "word_count": 700},
    ]
    assignments = [
        {
            "sr_episode_id": 1,
            "topic_id": "topic-a",
            "topic_label": "Topic A",
            "word_count": 300,
            "similarity": 0.51,
            "evidence_tier": "strong",
            "excerpt": "Ett ensamt svagt belägg",
        }
    ]

    assert aggregate_episode_topics(chunks, assignments) == []


def test_manual_hierarchy_covers_k500_exactly_once():
    validate_manual_hierarchy()
    accepted = [
        raw_id for topic in SPECIFIC_TOPICS.values() for raw_id in topic["raw"]
    ]
    excluded = [raw_id for values in EXCLUDED.values() for raw_id in values]

    assert sorted(accepted + excluded) == list(range(500))
    assert len(accepted + excluded) == len(set(accepted + excluded))
    assert all(topic["label"] and topic["broad"] for topic in SPECIFIC_TOPICS.values())
