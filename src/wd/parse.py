"""Match SR speaker appearances to dated Wikidata participants."""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable
import re
from typing import Any


SPEAKER_LABEL_OVERRIDES = {
    # SR presented him anonymously in the episode metadata, while the
    # corresponding Wikidata item uses his full name.
    "Stefan, pappa till Ebba": "Stefan Åkerlund",
    # Sveriges Radio credits these Hooja members by stage name, whereas their
    # Wikidata items are correctly labelled with their real names.
    "Hooja": "Joakim Lithner",
    "Mårdis": "Markus Mattsby",
    # She has since changed her name; the SR archive retains her former name.
    "Johanna Almer": "Johanna Andersson",
}
QID_RE = re.compile(r"^Q[1-9][0-9]*$")


def _binding_value(binding: dict[str, Any], name: str) -> str | None:
    value = binding.get(name, {}).get("value")
    return value if isinstance(value, str) else None


def _qid(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.rsplit("/", 1)[-1]
    return candidate if QID_RE.fullmatch(candidate) else None


def _date(value: str | None, precision: str | None) -> str | None:
    if not value or not precision:
        return None
    normalized = value.lstrip("+").split("T", 1)[0]
    if precision.isdigit() and int(precision) >= 11:
        return normalized
    if precision == "10":
        return normalized[:7]
    if precision == "9":
        return normalized[:4]
    return None


def parse_speaker_metadata(bindings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge ungrouped Wikidata SPARQL rows into one record per speaker."""
    records: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        qid = _qid(_binding_value(binding, "speaker"))
        if not qid:
            continue
        record = records.setdefault(
            qid,
            {
                "wikidata_id": qid,
                "wikipedia_url": None,
                "gender": None,
                "gender_id": None,
                "birth_date": _date(
                    _binding_value(binding, "birthDate"),
                    _binding_value(binding, "birthPrecision"),
                ),
                "death_date": _date(
                    _binding_value(binding, "deathDate"),
                    _binding_value(binding, "deathPrecision"),
                ),
                "citizenships": {},
                "occupations": {},
            },
        )
        record["wikipedia_url"] = (
            _binding_value(binding, "svArticle")
            or record["wikipedia_url"]
            or _binding_value(binding, "enArticle")
        )
        gender_id = _qid(_binding_value(binding, "gender"))
        if gender_id and record["gender_id"] is None:
            record["gender_id"] = gender_id
            record["gender"] = _binding_value(binding, "genderLabel") or gender_id
        for value_name, label_name, key in (
            ("citizenship", "citizenshipLabel", "citizenships"),
            ("occupation", "occupationLabel", "occupations"),
        ):
            value_id = _qid(_binding_value(binding, value_name))
            if value_id:
                record[key][value_id] = _binding_value(binding, label_name) or value_id

    return [
        {
            **{key: value for key, value in record.items() if key not in {"citizenships", "occupations"}},
            "citizenships": list(record["citizenships"].values()),
            "citizenship_ids": list(record["citizenships"]),
            "occupations": list(record["occupations"].values()),
            "occupation_ids": list(record["occupations"]),
        }
        for _, record in sorted(records.items())
    ]


def _normalise_name(value: str) -> str:
    """Normalize labels enough for conservative exact name matching."""
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.casefold().replace("’", "'").split())


def _participant_matches(
    season_participants: Iterable[dict[str, Any]],
) -> dict[str, list[tuple[str, str | None]]]:
    """Group unique (Q-ID, label) pairs by broadcast date."""
    matches: dict[str, dict[str, str | None]] = {}
    for binding in season_participants:
        try:
            date = binding["date"]["value"][:10]
            qid = binding["speaker"]["value"].rsplit("/", 1)[-1]
        except (KeyError, TypeError):
            continue
        # A Wikidata statement can use a ``some value`` placeholder. WDQS
        # represents that as a generated node rather than a Q-item; it must
        # not make an otherwise unambiguous broadcast date look ambiguous.
        if not date or not QID_RE.fullmatch(qid):
            continue
        label = (binding.get("speakerLabel") or {}).get("value")
        matches.setdefault(date, {}).setdefault(qid, label)
        if label and matches[date][qid] is None:
            matches[date][qid] = label
    return {date: list(qids.items()) for date, qids in matches.items()}


def enrich_speakers_with_wikidata(
    speakers: Iterable[dict[str, Any]],
    *,
    episodes: Iterable[dict[str, Any]],
    season_participants: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add a Q-ID independently to each speaker appearance.

    A unique participant on a date is accepted as a conservative date-only
    match. When several participants share a date, the Wikidata label must
    exactly match the SR speaker name after normalization.
    """
    speaker_rows = list(speakers)
    speaker_counts = Counter(row.get("sr_episode_id") for row in speaker_rows)
    episode_dates = {
        episode["sr_episode_id"]: episode["date"] for episode in episodes
    }
    participants_by_date = _participant_matches(season_participants)
    enriched: list[dict[str, Any]] = []
    for speaker in speaker_rows:
        row = dict(speaker)
        date = episode_dates.get(row.get("sr_episode_id"))
        candidates = participants_by_date.get(date, []) if date else []
        candidate_ids = {qid for qid, _ in candidates}
        speaker_name = str(row["speaker"])
        accepted_labels = {
            _normalise_name(speaker_name),
            _normalise_name(SPEAKER_LABEL_OVERRIDES.get(speaker_name, speaker_name)),
        }
        matched_ids = {
            qid
            for qid, label in candidates
            if label and _normalise_name(label) in accepted_labels
        }
        if len(matched_ids) == 1:
            row["wikidata_id"] = next(iter(matched_ids))
        elif len(candidate_ids) == 1 and speaker_counts[row.get("sr_episode_id")] == 1:
            row["wikidata_id"] = next(iter(candidate_ids))
        else:
            row["wikidata_id"] = None
        enriched.append(row)
    return enriched
