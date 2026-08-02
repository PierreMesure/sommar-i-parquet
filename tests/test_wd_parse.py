from src.wd.parse import enrich_episodes_with_wikidata


def test_season_participant_match_has_priority() -> None:
    episodes = [{"speaker": "Alex Example", "date": "2020-07-01"}]
    season_participants = [
        {
            "speaker": {"value": "http://www.wikidata.org/entity/Q10"},
            "speakerLabel": {"value": "Alex Example"},
            "date": {"value": "2020-07-01T00:00:00Z"},
        }
    ]

    result = enrich_episodes_with_wikidata(
        episodes,
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

    result = enrich_episodes_with_wikidata(
        [{"speaker": "Alex Example", "date": "2020-07-01"}],
        season_participants=season_participants,
    )

    assert result[0]["wikidata_id"] is None
