from src.wd.parse import enrich_speakers_with_wikidata


def test_season_participant_match_has_priority() -> None:
    episodes = [{"sr_episode_id": 1, "date": "2020-07-01"}]
    season_participants = [
        {
            "speaker": {"value": "http://www.wikidata.org/entity/Q10"},
            "speakerLabel": {"value": "Alex Example"},
            "date": {"value": "2020-07-01T00:00:00Z"},
        }
    ]

    result = enrich_speakers_with_wikidata(
        [{"sr_episode_id": 1, "speaker": "Alex Example"}],
        episodes=episodes,
        season_participants=season_participants,
    )

    assert result[0]["wikidata_id"] == "Q10"


def test_conflicting_season_participants_are_left_unmatched() -> None:
    season_participants = [
        {
            "speaker": {"value": f"http://www.wikidata.org/entity/{qid}"},
            "speakerLabel": {"value": "Alex Example"},
            "date": {"value": "2020-07-01T00:00:00Z"},
        }
        for qid in ("Q10", "Q11")
    ]

    result = enrich_speakers_with_wikidata(
        [{"sr_episode_id": 1, "speaker": "Alex Example"}],
        episodes=[{"sr_episode_id": 1, "date": "2020-07-01"}],
        season_participants=season_participants,
    )

    assert result[0]["wikidata_id"] is None


def test_multiple_speakers_are_matched_independently_by_label() -> None:
    episodes = [{"sr_episode_id": 1, "date": "2023-07-01"}]
    speakers = [
        {"sr_episode_id": 1, "speaker": "Hooja"},
        {"sr_episode_id": 1, "speaker": "Mårdis"},
    ]
    season_participants = [
        {
            "speaker": {"value": "http://www.wikidata.org/entity/Q100"},
            "speakerLabel": {"value": "Hooja"},
            "date": {"value": "2023-07-01T00:00:00Z"},
        },
        {
            "speaker": {"value": "http://www.wikidata.org/entity/Q101"},
            "speakerLabel": {"value": "Mårdis"},
            "date": {"value": "2023-07-01T00:00:00Z"},
        },
    ]

    result = enrich_speakers_with_wikidata(
        speakers,
        episodes=episodes,
        season_participants=season_participants,
    )

    assert [row["wikidata_id"] for row in result] == ["Q100", "Q101"]


def test_date_only_match_is_not_copied_to_all_speakers() -> None:
    speakers = [
        {"sr_episode_id": 1, "speaker": "First Example"},
        {"sr_episode_id": 1, "speaker": "Second Example"},
    ]
    result = enrich_speakers_with_wikidata(
        speakers,
        episodes=[{"sr_episode_id": 1, "date": "2020-07-01"}],
        season_participants=[
            {
                "speaker": {"value": "http://www.wikidata.org/entity/Q10"},
                "date": {"value": "2020-07-01T00:00:00Z"},
            }
        ],
    )

    assert [row["wikidata_id"] for row in result] == [None, None]


def test_speaker_label_override_matches_anonymous_sr_credit() -> None:
    result = enrich_speakers_with_wikidata(
        [{"sr_episode_id": 1, "speaker": "Stefan, pappa till Ebba"}],
        episodes=[{"sr_episode_id": 1, "date": "2018-07-10"}],
        season_participants=[
            {
                "speaker": {"value": "http://www.wikidata.org/entity/Q100"},
                "speakerLabel": {"value": "Stefan Åkerlund"},
                "date": {"value": "2018-07-10T00:00:00Z"},
            }
        ],
    )

    assert result[0]["wikidata_id"] == "Q100"


def test_non_qid_participant_placeholder_is_ignored() -> None:
    result = enrich_speakers_with_wikidata(
        [{"sr_episode_id": 1, "speaker": "Name variant"}],
        episodes=[{"sr_episode_id": 1, "date": "2009-08-13"}],
        season_participants=[
            {
                "speaker": {
                    "value": "http://www.wikidata.org/.well-known/genid/abc"
                },
                "date": {"value": "2009-08-13T00:00:00Z"},
            },
            {
                "speaker": {"value": "http://www.wikidata.org/entity/Q100"},
                "speakerLabel": {"value": "Different Wikidata label"},
                "date": {"value": "2009-08-13T00:00:00Z"},
            },
        ],
    )

    assert result[0]["wikidata_id"] == "Q100"
