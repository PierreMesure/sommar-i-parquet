from src.whisper.quality import strip_artifacts, strip_introductions, transcript_artifacts


def test_detects_repeated_word_and_character_loops():
    transcript = {"segments": [
        {"text": "kanske kanske kanske kanske kanske kanske kanske kanske kanske kanske kanske kanske kanske kanske kanske kanske kanske kanske kanske kanske"},
        {"text": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    ]}

    assert [row["segment_index"] for row in transcript_artifacts(transcript)] == [0, 1]


def test_strips_known_sr_introductions_and_retains_them_for_audit():
    transcript = {"segments": [
        {"text": "Det här programmet är Sommar i P1."},
        {"text": "Jag växte upp i Göteborg."},
        {"text": "Musiken har förkortats av upphovsrättsliga skäl."},
    ]}

    removed = strip_introductions(transcript)

    assert len(removed) == 2
    assert [segment["text"] for segment in transcript["segments"]] == ["Jag växte upp i Göteborg."]
    assert transcript["sommar_i_parquet"]["removed_introduction_segments"] == removed


def test_strips_standard_podcast_intro_and_preserves_artifact_metadata():
    transcript = {"segments": [
        {"text": "Det här är poddversionen av Sommar i P1."},
        {"text": "Av upphovsrättsliga skäl är musiken förkortad."},
        {"text": "Dags för Sommar i P1 med Ada Lovelace."},
        {"text": "kanske " * 20},
    ]}

    strip_introductions(transcript)
    artifacts = transcript_artifacts(transcript)
    removed = strip_artifacts(transcript, artifacts)

    assert transcript["segments"] == []
    assert len(transcript["sommar_i_parquet"]["removed_introduction_segments"]) == 3
    assert removed[0]["reasons"] == ["dominant_token_loop"]


def test_does_not_strip_matching_boilerplate_after_the_opening():
    transcript = {"segments": [
        {"start": 45, "text": "Det här är poddversionen av Sommar i P1."},
    ]}

    assert strip_introductions(transcript) == []
    assert len(transcript["segments"]) == 1
