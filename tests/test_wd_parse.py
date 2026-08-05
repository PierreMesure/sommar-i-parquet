from src.wd.parse import (
    attach_episode_speakers,
    attach_episode_speaker_ages,
    age_at_date,
    build_speaker_records,
    enrich_speakers_with_wikidata,
    parse_speaker_metadata,
)


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


def test_stage_names_match_wikidata_items_labelled_with_real_names() -> None:
    result = enrich_speakers_with_wikidata(
        [
            {"sr_episode_id": 1, "speaker": "Hooja"},
            {"sr_episode_id": 1, "speaker": "Mårdis"},
        ],
        episodes=[{"sr_episode_id": 1, "date": "2023-08-12"}],
        season_participants=[
            {
                "speaker": {"value": "http://www.wikidata.org/entity/Q100"},
                "speakerLabel": {"value": "Joakim Lithner"},
                "date": {"value": "2023-08-12T00:00:00Z"},
            },
            {
                "speaker": {"value": "http://www.wikidata.org/entity/Q101"},
                "speakerLabel": {"value": "Markus Mattsby"},
                "date": {"value": "2023-08-12T00:00:00Z"},
            },
        ],
    )

    assert [speaker["wikidata_id"] for speaker in result] == ["Q100", "Q101"]


def test_emil_beer_uses_verified_qid_override() -> None:
    result = enrich_speakers_with_wikidata(
        [{"sr_episode_id": 1, "speaker": "Emil Beer"}],
        episodes=[{"sr_episode_id": 1, "date": "2016-08-02"}],
        season_participants=[],
    )

    assert result[0]["wikidata_id"] == "Q113960293"


def test_normalized_episode_and_speaker_records() -> None:
    appearances = [
        {
            "sr_episode_id": 1,
            "speaker_index": 1,
            "speaker": "Ada King",
            "wikidata_id": "Q1",
            "date": "2000-07-01",
        },
        {
            "sr_episode_id": 2,
            "speaker_index": 1,
            "speaker": "Ada Lovelace",
            "wikidata_id": "Q1",
            "date": "2001-07-01",
        },
    ]
    episodes = attach_episode_speakers(
        [{"sr_episode_id": 1}, {"sr_episode_id": 2}], appearances
    )
    speakers = build_speaker_records(
        appearances,
        [
            {
                "wikidata_id": "Q1",
                "gender": "kvinna",
                "occupations": ["matematiker"],
            }
        ],
        episodes=[
            {"sr_episode_id": 1, "date": "2000-07-01"},
            {"sr_episode_id": 2, "date": "2001-07-01"},
        ],
    )

    assert episodes == [
        {"sr_episode_id": 1, "episode_speakers": ["Q1"]},
        {"sr_episode_id": 2, "episode_speakers": ["Q1"]},
    ]
    assert speakers[0]["speaker"] == "Ada Lovelace"
    assert speakers[0]["sr_names"] == ["Ada King", "Ada Lovelace"]
    assert speakers[0]["episode_count"] == 2
    assert speakers[0]["gender"] == "kvinna"
    assert speakers[0]["occupations"] == ["matematiker"]


def test_age_is_conservative_for_partial_birth_dates() -> None:
    assert age_at_date("1980", "2020-01-01") == 39
    assert age_at_date("1980-06", "2020-06-30") == 39
    assert age_at_date("1980-06-15", "2020-06-15") == 40
    assert age_at_date("1980", "2020-01-01", "2010") is None


def test_episode_speaker_ages_follow_speaker_order() -> None:
    result = attach_episode_speaker_ages(
        [{"sr_episode_id": 1, "date": "2020-07-01", "episode_speakers": ["Q1", "Q2"]}],
        [
            {"sr_episode_id": 1, "speaker_index": 2, "wikidata_id": "Q2"},
            {"sr_episode_id": 1, "speaker_index": 1, "wikidata_id": "Q1"},
        ],
        [
            {"wikidata_id": "Q1", "birth_date": "1980-08-01"},
            {"wikidata_id": "Q2", "birth_date": "1990"},
        ],
    )
    assert result[0]["speaker_ages"] == [39, 29]


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


def test_parse_speaker_metadata_prefers_swedish_labels_and_wikipedia() -> None:
    result = parse_speaker_metadata(
        [
            _binding(
                speaker="http://www.wikidata.org/entity/Q1",
                speakerLabel="Ada Lovelace",
                speakerDescription="brittisk matematiker",
                svArticle="https://sv.wikipedia.org/wiki/Ada_Lovelace",
                gender="http://www.wikidata.org/entity/Q2",
                genderLabel="kvinna",
                birthDate="1815-12-10T00:00:00Z",
                birthPrecision="11",
                deathDate="1852-01-01T00:00:00Z",
                deathPrecision="9",
                citizenship="http://www.wikidata.org/entity/Q3",
                citizenshipLabel="United Kingdom",
                occupation="http://www.wikidata.org/entity/Q4",
                occupationLabel="matematiker",
            ),
            _binding(
                speaker="http://www.wikidata.org/entity/Q1",
                occupation="http://www.wikidata.org/entity/Q6",
                occupationLabel="programmerare",
            ),
        ]
    )

    assert result == [
        {
            "wikidata_id": "Q1",
            "wikidata_label": "Ada Lovelace",
            "wikidata_description": "brittisk matematiker",
            "wikipedia_url": "https://sv.wikipedia.org/wiki/Ada_Lovelace",
            "gender": "kvinna",
            "gender_id": "Q2",
            "birth_date": "1815-12-10",
            "death_date": "1852",
            "citizenships": ["United Kingdom"],
            "citizenship_ids": ["Q3"],
            "occupations": ["matematiker", "programmerare"],
            "occupation_ids": ["Q4", "Q6"],
        }
    ]


def _binding(**values: str) -> dict:
    return {name: {"value": value} for name, value in values.items()}
