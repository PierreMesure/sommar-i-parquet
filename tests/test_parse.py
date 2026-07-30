from src.sr.parse import parse_episode


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
        "speaker": "Kalle Moraeus",
        "date": "2015-06-24",
        "year": 2015,
        "program_type": "Sommar",
        "episode_url": "https://www.sverigesradio.se/avsnitt/562584",
        "mp3_url": "https://example.test/kalle.mp3",
        "length_minutes": 49.95,
        "short_summary": "MUSIKER, ARTIST, PROGRAMLEDARE.",
    }


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
    assert parsed["length_minutes"] is None

