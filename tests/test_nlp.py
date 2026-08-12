import numpy as np

from src.nlp.segment import score_boundaries, segment_transcript
from src.nlp.transcripts import EpisodeTranscript, TranscriptUnit, build_units


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
