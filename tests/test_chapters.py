from src.nlp.chapters import (
    ChapterBoundary,
    chapter_segment_ranges,
    resolve_boundaries,
)


def test_resolve_boundaries_snaps_a_quoted_anchor_to_a_segment_end():
    segments = [
        {"start": 0, "end": 3, "text": "Jag växte upp i Kiruna med min familj."},
        {"start": 3, "end": 7, "text": "Det var där jag lärde mig att spela fotboll."},
        {"start": 8, "end": 12, "text": "Senare började jag arbeta som lärare i Stockholm."},
        {"start": 12, "end": 16, "text": "I skolan mötte jag många nyfikna elever."},
    ]
    boundaries = resolve_boundaries(
        [ChapterBoundary(after_words="där jag lärde mig att spela fotboll")],
        segments,
        min_segments_per_chapter=1,
    )
    assert len(boundaries) == 1
    assert boundaries[0].segment_index == 1
    assert boundaries[0].score > 0.9
    assert chapter_segment_ranges(len(segments), boundaries) == [(0, 1), (2, 3)]
