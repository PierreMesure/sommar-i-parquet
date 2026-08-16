from src.sr.parse import (
    _speaker_from_title,
    exclusion_reason,
    parse_episode,
    parse_episodes,
    parse_speakers,
)


def test_winter_subtitle_is_not_part_of_the_speaker_name() -> None:
    assert _speaker_from_title("Tage Danielsson - På Vintergatan 1971", 1971) == "Tage Danielsson"


def test_sommarprat_suffix_is_not_part_of_the_speaker_name() -> None:
    assert _speaker_from_title("Bo Landin Sommarprat\u00a01977", 1977) == "Bo Landin"


def test_hyphen_before_year_is_not_part_of_the_speaker_name() -> None:
    assert _speaker_from_title("Monica Borrfors -1988", 1988) == "Monica Borrfors"


def test_parse_episode() -> None:
    raw = {
        "id": 562584,
        "title": "Kalle Moraeus 2015",
        "description": "MUSIKER, ARTIST, PROGRAMLEDARE.",
        "url": "https://www.sverigesradio.se/avsnitt/562584",
        "publishdateutc": "/Date(1435143600000)/",
        "downloadpodfile": {
            "duration": 2997,
            "url": "https://example.test/kalle.mp3",
        },
    }

    assert parse_episode(raw) == {
        "sr_episode_id": 562584,
        "sr_audio_id": None,
        "source_title": "Kalle Moraeus 2015",
        "speaker": "Kalle Moraeus",
        "date": "2015-06-24",
        "year": 2015,
        "program_type": "Sommar",
        "episode_url": "https://www.sverigesradio.se/avsnitt/562584",
        "mp3_url": "https://example.test/kalle.mp3",
        "length_seconds": 2997,
        "audio_file_size_bytes": None,
        "image_url": None,
        "image_credit": None,
        "short_summary": "MUSIKER, ARTIST, PROGRAMLEDARE.",
        "is_listeners_host": False,
    }


def test_listeners_host_detected_from_description_or_title() -> None:
    from_desc = {
        "id": 2181521,
        "title": "Eva Armini 2023",
        "description": "Lyssnarnas Sommarvärd om att bli bortadopterad...",
        "url": "https://example.test/episode",
        "publishdateutc": "2023-07-12T12:00:00Z",
        "downloadpodfile": {"duration": 3600, "url": "https://example.test/audio.mp3"},
    }
    from_title = {
        "id": 12345,
        "title": "Emilia Lind (Lyssnarnas Sommarvärd 2016)",
        "description": "Student.",
        "url": "https://example.test/episode",
        "publishdateutc": "2016-07-20T12:00:00Z",
        "downloadpodfile": {"duration": 3600, "url": "https://example.test/audio.mp3"},
    }
    from_override = {
        "id": 921565,  # Tommy Ivarsson, 2017
        "title": "Tommy Ivarsson",
        "description": "Om vad som händer när man lever i den värsta av mardrömmar...",
        "url": "https://example.test/episode",
        "publishdateutc": "2017-07-19T12:00:00Z",
        "downloadpodfile": {"duration": 3600, "url": "https://example.test/audio.mp3"},
    }

    assert parse_episode(from_desc)["is_listeners_host"] is True
    assert parse_episode(from_title)["is_listeners_host"] is True
    assert parse_episode(from_override)["is_listeners_host"] is True


def test_parse_iso_date_and_missing_audio() -> None:
    raw = {
        "id": 1,
        "title": "Example Speaker",
        "description": None,
        "url": "https://example.test/episode",
        "publishdateutc": "2025-12-25T12:00:00Z",
    }

    parsed = parse_episode(raw)

    assert parsed["date"] == "2025-12-25"
    assert parsed["program_type"] == "Vinter"
    assert parsed["mp3_url"] is None
    assert parsed["length_seconds"] is None
    assert exclusion_reason(parsed) == "missing_audio"


def test_january_vinter_episode_uses_the_previous_season_year() -> None:
    raw = {
        "id": 2,
        "title": "Example Speaker",
        "description": None,
        "url": "https://example.test/episode",
        "publishdateutc": "2026-01-01T12:00:00Z",
        "downloadpodfile": {"duration": 3600, "url": "https://example.test/audio.mp3"},
    }

    parsed = parse_episode(raw)

    assert parsed["date"] == "2026-01-01"
    assert parsed["year"] == 2025
    assert parsed["program_type"] == "Vinter"


def test_speaker_rows_split_co_credited_people_and_known_exceptions() -> None:
    speakers = parse_speakers(
        [
            {"sr_episode_id": 1, "speaker": "Jenny och Susanna Kallur"},
            {"sr_episode_id": 2, "speaker": "Niklas Natt och Dag"},
            {
                "sr_episode_id": 3,
                "speaker": "IJustWantToBeCool (Victor Beer, Emil Beer, Joel Adolphson)",
            },
        ]
    )

    assert speakers == [
        {"sr_episode_id": 1, "speaker_index": 1, "speaker_appearance_id": "1:1", "speaker": "Jenny Kallur", "wikidata_id": None},
        {"sr_episode_id": 1, "speaker_index": 2, "speaker_appearance_id": "1:2", "speaker": "Susanna Kallur", "wikidata_id": None},
        {"sr_episode_id": 2, "speaker_index": 1, "speaker_appearance_id": "2:1", "speaker": "Niklas Natt och Dag", "wikidata_id": None},
        {"sr_episode_id": 3, "speaker_index": 1, "speaker_appearance_id": "3:1", "speaker": "Victor Beer", "wikidata_id": None},
        {"sr_episode_id": 3, "speaker_index": 2, "speaker_appearance_id": "3:2", "speaker": "Emil Beer", "wikidata_id": None},
        {"sr_episode_id": 3, "speaker_index": 3, "speaker_appearance_id": "3:3", "speaker": "Joel Adolphson", "wikidata_id": None},
    ]


def test_september_archive_episode_is_sommar() -> None:
    raw = {
        "id": 425083,
        "title": "Marie Selander 1974",
        "description": "Sångerska, kompositör och sångpedagog",
        "url": "https://example.test/episode",
        "publishdateutc": "1974-09-03T12:00:00Z",
        "downloadpodfile": {
            "duration": 1549,
            "url": "https://example.test/marie.mp3",
        },
    }

    parsed = parse_episode(raw)

    assert parsed["date"] == "1974-09-03"
    assert parsed["program_type"] == "Sommar"
    assert exclusion_reason(parsed) is None


def test_archive_reissue_recovers_original_date_and_speaker() -> None:
    raw = {
        "id": 1619205,
        "title": "Sommar i P1 med Sven Wollter",
        "description": "Sommar i P1 med Sven Wollter från den 8 juli 1981",
        "url": "https://example.test/episode",
        "publishdateutc": "2020-11-13T15:00:00Z",
        "downloadpodfile": {
            "duration": 2219,
            "url": "https://example.test/wollter.mp3",
        },
    }

    parsed = parse_episode(raw)

    assert parsed["speaker"] == "Sven Wollter"
    assert parsed["date"] == "1981-07-08"
    assert parsed["year"] == 1981
    assert parsed["program_type"] == "Sommar"


def test_winter_date_wins_over_previous_sommar_host_description() -> None:
    raw = {
        "id": 3,
        "title": "Example Speaker - Vinter 2018",
        "description": "Konstnär. Sommarvärd 2017.",
        "url": "https://example.test/episode",
        "publishdateutc": "2018-12-27T12:00:00Z",
        "downloadpodfile": {
            "duration": 3600,
            "url": "https://example.test/winter.mp3",
        },
    }

    parsed = parse_episode(raw)

    assert parsed["speaker"] == "Example Speaker"
    assert parsed["program_type"] == "Vinter"


def test_programme_suffix_and_talk_title_are_removed_from_speaker() -> None:
    raw = {
        "id": 30,
        "title": "Olof Wretling – Kvinnorna i mitt liv – Vinterprat",
        "description": "Komiker.",
        "url": "https://example.test/episode",
        "publishdateutc": "2019-12-29T12:00:00Z",
        "downloadpodfile": {"duration": 3600, "url": "https://example.test/winter.mp3"},
    }

    parsed = parse_episode(raw)

    assert parsed["speaker"] == "Olof Wretling"


def test_trailing_lifespan_is_removed_from_speaker() -> None:
    raw = {
        "id": 33,
        "title": "Sven-David Sandström 1942 – 2019",
        "description": "Kompositör.",
        "url": "https://example.test/episode",
        "publishdateutc": "2019-07-10T12:00:00Z",
        "downloadpodfile": {"duration": 3600, "url": "https://example.test/audio.mp3"},
    }

    assert parse_episode(raw)["speaker"] == "Sven-David Sandström"


def test_programme_suffix_without_dash_is_removed_except_ingrid_sommar() -> None:
    staffan = {
        "id": 31,
        "title": "Staffan Olsson Vinter 2010",
        "description": "",
        "url": "https://example.test/episode",
        "publishdateutc": "2010-12-28T12:00:00Z",
        "downloadpodfile": {"duration": 3600, "url": "https://example.test/winter.mp3"},
    }
    ingrid = {
        **staffan,
        "id": 32,
        "title": "Ingrid Sommar 2005",
        "publishdateutc": "2005-07-01T12:00:00Z",
    }

    assert parse_episode(staffan)["speaker"] == "Staffan Olsson"
    assert parse_episode(ingrid)["speaker"] == "Ingrid Sommar"


def test_season_and_listener_host_suffixes_are_removed() -> None:
    examples = (
        ("Martina Haag - Vinter 2013/14", "Martina Haag"),
        ("Lars Lerin - Vinter 2015 (jan)", "Lars Lerin"),
        ("Herman Geijer - Lyssnarnas Sommarvärd", "Herman Geijer"),
        ("Emilia Lind (Lyssnarnas Sommarvärd 2016)", "Emilia Lind"),
    )
    for episode_id, (title, speaker) in enumerate(examples, start=40):
        raw = {
            "id": episode_id,
            "title": title,
            "description": "",
            "url": "https://example.test/episode",
            "publishdateutc": "2015-12-28T12:00:00Z",
            "downloadpodfile": {"duration": 3600, "url": "https://example.test/audio.mp3"},
        }
        assert parse_episode(raw)["speaker"] == speaker


def test_rebroadcast_title_is_normalized_but_kept() -> None:
    raw = {
        "id": 4,
        "title": "Henrik Dorsin (Repris från 2012)",
        "description": "Skådespelare och komiker.",
        "url": "https://example.test/episode",
        "publishdateutc": "2017-08-11T12:00:00Z",
        "downloadpodfile": {
            "duration": 5102,
            "url": "https://example.test/rebroadcast.mp3",
        },
    }

    parsed = parse_episode(raw)

    assert parsed["speaker"] == "Henrik Dorsin"
    assert exclusion_reason(parsed) is None


def test_english_version_is_filtered() -> None:
    raw = {
        "id": 5,
        "title": 'Felix "PewDiePie" Kjellberg (in English)',
        "description": "English edition.",
        "url": "https://example.test/episode",
        "publishdateutc": "2014-08-09T12:00:00Z",
        "downloadpodfile": {
            "duration": 3600,
            "url": "https://example.test/english.mp3",
        },
    }

    parsed = parse_episode(raw)

    assert exclusion_reason(parsed) == "alternate_language"


def test_specials_and_short_items_are_filtered() -> None:
    raw = [
        {
            "id": 1,
            "title": "TRAILER: Sommar i P1",
            "description": "Trailer",
            "url": "https://example.test/trailer",
            "publishdateutc": "2020-06-11T12:00:00Z",
            "downloadpodfile": {
                "duration": 42,
                "url": "https://example.test/trailer.mp3",
            },
        },
        {
            "id": 2,
            "title": "Årets Sommarvärdar presenteras",
            "description": "Presskonferens",
            "url": "https://example.test/announcement",
            "publishdateutc": "2013-06-01T12:00:00Z",
            "downloadpodfile": {
                "duration": 3600,
                "url": "https://example.test/announcement.mp3",
            },
        },
    ]

    assert parse_episodes(raw) == []
    assert len(parse_episodes(raw, include_specials=True)) == 2
