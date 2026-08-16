from src.whisper.quality import (
    strip_artifacts,
    strip_boilerplate,
    strip_recommended_episode_extract,
    transcript_artifacts,
    transcript_boilerplate,
)


def test_detects_repeated_word_and_character_loops():
    transcript = {
        "segments": [
            {"text": "kanske " * 20},
            {"text": "a" * 120},
        ]
    }

    assert [row["segment_index"] for row in transcript_artifacts(transcript)] == [0, 1]


def test_strips_observed_sr_opening_variants_and_retains_evidence():
    transcript = {
        "segments": [
            {
                "start": 0.4,
                "end": 4.9,
                "text": "Det här är poddversionen av Vinter i P1 med Sissela Kyle.",
            },
            {
                "start": 5.0,
                "end": 9.0,
                "text": "Musiken är förkortad av upphovsrättsliga skäl.",
            },
            {
                "start": 10.0,
                "end": 15.0,
                "text": "Alla sommarprogram finns i vår app Sveriges Radio Play.",
            },
            {"start": 25.0, "end": 31.0, "text": "Min berättelse börjar i Göteborg."},
        ]
    }

    removed = strip_boilerplate(transcript)

    assert len(removed) == 3
    assert [segment["text"] for segment in transcript["segments"]] == [
        "Min berättelse börjar i Göteborg."
    ]
    assert transcript["sommar_i_parquet"]["removed_introduction_segments"] == [
        item["segment"] for item in removed
    ]
    assert transcript["sommar_i_parquet"]["removed_introduction_segment_audit"][0][
        "reasons"
    ] == ["programme_identification"]


def test_detects_closing_music_list_notice_but_not_body_reference():
    transcript = {
        "segments": [
            {
                "start": 100.0,
                "end": 105.0,
                "text": "Jag hittade en lista över musiken på biblioteket.",
            },
            {
                "start": 950.0,
                "end": 960.0,
                "text": "Hela musiklistan hittar du på Sveriges Radios hemsida.",
            },
            {"start": 970.0, "end": 1000.0, "text": "Tack för att du lyssnade."},
        ]
    }

    matches = transcript_boilerplate(transcript)

    assert [match["segment_index"] for match in matches] == [1]


def test_detects_old_web_promotions_and_production_credits_only_at_end():
    transcript = {
        "segments": [
            {"start": 0, "end": 5, "text": "Producenten berättar om sitt arbete."},
            {
                "start": 900,
                "end": 906,
                "text": "Ladda ner Sommars specialdesignade webbradioapparat via sr.se sommar.",
            },
            {
                "start": 907,
                "end": 912,
                "text": "Det var Sommar i P1 med Ada Lovelace.",
            },
            {"start": 913, "end": 917, "text": "Producent Boel Adler."},
        ]
    }

    matches = transcript_boilerplate(transcript)

    assert [match["segment_index"] for match in matches] == [1, 2, 3]


def test_strips_artifact_and_preserves_detection_reason():
    transcript = {
        "segments": [
            {"text": "En riktig mening."},
            {"text": "kanske " * 20},
        ]
    }

    removed = strip_artifacts(transcript, transcript_artifacts(transcript))

    assert [segment["text"] for segment in transcript["segments"]] == ["En riktig mening."]
    assert removed[0]["reasons"] == ["dominant_token_loop"]


def test_does_not_strip_matching_opening_boilerplate_late_in_programme():
    transcript = {
        "segments": [
            {"start": 0, "end": 1, "text": "Hej."},
            {
                "start": 100,
                "end": 105,
                "text": "Det här är poddversionen av Sommar i P1.",
            },
            {"start": 1000, "end": 1001, "text": "Slut."},
        ]
    }

    assert transcript_boilerplate(transcript) == []


def test_removes_production_credit_and_everything_after_it_with_words():
    transcript = {
        "segments": [
            {"start": 0, "end": 10, "text": "Själva programmets sista ord."},
            {"start": 11, "end": 14, "text": "Producent Anna Andersson och tekniker Bo Berg."},
            {"start": 15, "end": 30, "text": "En annan sommarvärd berättade sedan."},
        ],
        "word_segments": [
            {"word": "Själva", "start": 0, "end": 1},
            {"word": "Producent", "start": 11, "end": 12},
            {"word": "En", "start": 15, "end": 16},
        ],
    }

    removed = strip_recommended_episode_extract(transcript)

    assert [segment["text"] for segment in removed] == [
        "Producent Anna Andersson och tekniker Bo Berg.",
        "En annan sommarvärd berättade sedan.",
    ]
    assert [segment["text"] for segment in transcript["segments"]] == [
        "Själva programmets sista ord."
    ]
    assert [word["word"] for word in transcript["word_segments"]] == ["Själva"]
    assert transcript["sommar_i_parquet"]["removed_recommended_episode_extract_audit"][0][
        "removed_segment_count"
    ] == 2


def test_recognises_comma_delimited_production_credit():
    transcript = {
        "segments": [
            {"start": 0, "end": 10, "text": "Programmet är slut."},
            {"start": 11, "end": 14, "text": "Producent, Effie Karabuda."},
            {"start": 15, "end": 30, "text": "En rekommenderad trailer."},
        ]
    }

    strip_recommended_episode_extract(transcript)

    assert [segment["text"] for segment in transcript["segments"]] == [
        "Programmet är slut."
    ]


def test_uses_final_sommarpratade_cue_when_credit_is_missing():
    transcript = {
        "segments": [
            {"start": 0, "end": 100, "text": "Det egna programmets sista del."},
            {"start": 101, "end": 105, "text": "En annan som sommarpratade 2020."},
            {"start": 106, "end": 150, "text": "Det rekommenderade programmets utdrag."},
        ]
    }

    removed = strip_recommended_episode_extract(transcript)

    assert [segment["text"] for segment in removed] == [
        "En annan som sommarpratade 2020.",
        "Det rekommenderade programmets utdrag.",
    ]
    assert transcript["sommar_i_parquet"]["removed_recommended_episode_extract_audit"][0][
        "boundary_source"
    ] == "trailing_recommendation_cue"
