from src.sr.parse import parse_music_playlists


def test_parse_music_sorts_tracks_and_identifies_theme_song() -> None:
    playlists = [
        {
            "sr_episode_id": 2815529,
            "publishdateutc": "/Date(1785063600000)/",
            "songs": [
                {
                    "title": "Second",
                    "artist": "Artist B",
                    "starttimeutc": "/Date(1785063725000)/",
                    "stoptimeutc": "/Date(1785063865000)/",
                },
                {
                    "title": "Sommar Sommar Sommar",
                    "artist": "SR",
                    "composer": "Sten Carlberg",
                    "albumname": "Sommar",
                    "recordlabel": "",
                    "starttimeutc": "/Date(1785063600000)/",
                    "stoptimeutc": "/Date(1785063617000)/",
                },
            ],
        }
    ]

    tracks = parse_music_playlists(playlists)

    assert [track["title"] for track in tracks] == [
        "Sommar Sommar Sommar",
        "Second",
    ]
    assert tracks[0]["track_number"] == 1
    assert tracks[0]["is_theme_song"] is True
    assert tracks[0]["record_label"] is None
    assert tracks[1]["track_number"] == 2
    assert tracks[1]["is_theme_song"] is False


def test_parse_music_deduplicates_and_keeps_richest_metadata() -> None:
    playlists = [
        {
            "sr_episode_id": 1,
            "publishdateutc": "2015-06-24T11:00:00Z",
            "songs": [
                {
                    "title": "Signatur Sommar Sommar Sommar",
                    "artist": "Radiosymfonikerna",
                    "starttimeutc": "/Date(1435143600000)/",
                    "stoptimeutc": "/Date(1435143617000)/",
                },
                {
                    "title": "Signatur Sommar Sommar Sommar",
                    "artist": "Radiosymfonikerna",
                    "albumname": "Sommar Sommar Sommar",
                    "starttimeutc": "/Date(1435143600000)/",
                    "stoptimeutc": "/Date(1435143617000)/",
                },
                {
                    "title": "Vintergatan",
                    "artist": "SR",
                    "starttimeutc": "/Date(1435143620000)/",
                    "stoptimeutc": "/Date(1435143637000)/",
                },
            ],
        }
    ]

    tracks = parse_music_playlists(playlists)

    assert len(tracks) == 2
    assert tracks[0]["album"] == "Sommar Sommar Sommar"
    assert tracks[0]["is_theme_song"] is True
    assert tracks[1]["is_theme_song"] is True
