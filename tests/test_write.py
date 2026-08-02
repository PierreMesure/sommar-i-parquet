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
        },
        {
            "sr_episode_id": 2,
            "date": "2001-07-01",
            "program_type": "Sommar",
            "length_seconds": 1800,
            "image_url": None,
            "short_summary": "En till beskrivning.",
        },
    ]
    speakers = [
        {"sr_episode_id": 1, "speaker_index": 1, "speaker": "Ada Lovelace"},
        {"sr_episode_id": 2, "speaker_index": 1, "speaker": "Ada Lovelace"},
    ]

    output = write_frontend_json(episodes, speakers, tmp_path / "episodes.json")

    assert json.loads(output.read_text()) == {
        "episodes": [
            {
                "id": 1,
                "date": "2000-07-01",
                "type": "Sommar",
                "minutes": 31,
                "image": "/images/2071/example.jpg",
                "description": "En beskrivning.",
                "speakers": ["Ada Lovelace"],
                "initials": "AL",
                "returning": True,
                "group": "Ada Lovelace",
            },
            {
                "id": 2,
                "date": "2001-07-01",
                "type": "Sommar",
                "minutes": 30,
                "image": None,
                "description": "En till beskrivning.",
                "speakers": ["Ada Lovelace"],
                "initials": "AL",
                "returning": True,
                "group": "Ada Lovelace",
            },
        ]
    }
