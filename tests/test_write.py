import json

from src.utils.write import write_frontend_json


def test_write_frontend_json_compacts_episode_and_speaker_data(tmp_path):
    episodes = [
        {
            "sr_episode_id": 1,
            "date": "2000-07-01",
            "program_type": "Sommar",
            "length_seconds": 1831,
            "image_url": "https://static-cdn.sr.se/images/2071/example.jpg",
            "short_summary": "En beskrivning.",
            "episode_speakers": ["Q1"],
            "speaker_ages": [184],
        },
        {
            "sr_episode_id": 2,
            "date": "2001-07-01",
            "program_type": "Sommar",
            "length_seconds": 1800,
            "image_url": None,
            "short_summary": "En till beskrivning.",
            "episode_speakers": ["Q1"],
            "speaker_ages": [185],
        },
    ]
    speakers = [
        {
            "wikidata_id": "Q1",
            "speaker": "Ada Lovelace",
            "sr_names": ["Augusta Ada King", "Ada Lovelace"],
            "episode_count": 2,
            "ages_at_episodes": [184, 185],
            "wikipedia_url": "https://sv.wikipedia.org/wiki/Ada_Lovelace",
            "gender": "kvinna",
            "birth_date": "1815-12-10",
            "death_date": "1852-11-27",
            "citizenships": ["Storbritannien"],
            "occupations": ["matematiker"],
        },
    ]

    output = write_frontend_json(episodes, speakers, tmp_path / "episodes.json")

    assert json.loads(output.read_text()) == {
        "speakers": {
            "Q1": {
                "name": "Ada Lovelace",
                "count": 2,
                "aliases": ["Augusta Ada King"],
                "wiki": "https://sv.wikipedia.org/wiki/Ada_Lovelace",
                "gender": "kvinna",
                "born": "1815-12-10",
                "died": "1852-11-27",
                "citizenships": ["Storbritannien"],
                "occupations": ["matematiker"],
                "ages": [184, 185],
            }
        },
        "episodes": [
            {
                "id": 1,
                "date": "2000-07-01",
                "type": "Sommar",
                "minutes": 31,
                "image": "/images/2071/example.jpg",
                "description": "En beskrivning.",
                "speakers": ["Q1"],
                "ages": [184],
                "initials": "AL",
            },
            {
                "id": 2,
                "date": "2001-07-01",
                "type": "Sommar",
                "minutes": 30,
                "image": None,
                "description": "En till beskrivning.",
                "speakers": ["Q1"],
                "ages": [185],
                "initials": "AL",
            },
        ]
    }
