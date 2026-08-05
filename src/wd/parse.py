"""Match SR speaker appearances to dated Wikidata participants."""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date
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
SPEAKER_QID_OVERRIDES = {
    # The 2016 season currently has no participant statement for Emil, while
    # the two other IJustWantToBeCool members do. His item is unambiguous.
    "Emil Beer": "Q113960293",
}
QID_RE = re.compile(r"^Q[1-9][0-9]*$")

SPEAKER_METADATA_DEFAULTS: dict[str, Any] = {
    "wikidata_label": None,
    "wikidata_description": None,
    "wikipedia_url": None,
    "gender": None,
    "gender_id": None,
    "birth_date": None,
    "death_date": None,
    "citizenships": [],
    "citizenship_ids": [],
    "occupations": [],
    "occupation_ids": [],
}


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


def age_at_date(
    birth_date: str | None,
    episode_date: str | None,
    death_date: str | None = None,
) -> int | None:
    """Return a conservative integer age for a broadcast date.

    Wikidata dates may only have year or month precision. In those cases we
    round down when the birthday could not yet have occurred.
    """
    if not birth_date or not episode_date:
        return None
    try:
        episode = date.fromisoformat(episode_date[:10])
        if death_date:
            death_parts = death_date.split("-")
            death_year = int(death_parts[0])
            if len(death_parts) == 1 and episode.year > death_year:
                return None
            if len(death_parts) == 2 and (episode.year, episode.month) > (
                death_year,
                int(death_parts[1]),
            ):
                return None
            if len(death_parts) >= 3 and episode > date(
                death_year, int(death_parts[1]), int(death_parts[2])
            ):
                return None
        parts = birth_date.split("-")
        birth_year = int(parts[0])
        if len(parts) == 1:
            age = episode.year - birth_year - 1
        elif len(parts) == 2:
            birth_month = int(parts[1])
            age = episode.year - birth_year - (episode.month <= birth_month)
        else:
            birthday = date(episode.year, int(parts[1]), int(parts[2]))
            age = episode.year - birth_year - (episode < birthday)
    except (TypeError, ValueError):
        return None
    return age if age >= 0 else None


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
                "wikidata_label": _binding_value(binding, "speakerLabel"),
                "wikidata_description": _binding_value(
                    binding, "speakerDescription"
                ),
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
        record["wikidata_label"] = (
            record["wikidata_label"]
            or _binding_value(binding, "speakerLabel")
        )
        record["wikidata_description"] = (
            record["wikidata_description"]
            or _binding_value(binding, "speakerDescription")
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


def attach_episode_speakers(
    episodes: Iterable[dict[str, Any]],
    speaker_appearances: Iterable[dict[str, Any]],
    *,
    require_qids: bool = True,
) -> list[dict[str, Any]]:
    """Attach ordered, unique speaker Q-IDs to canonical episode records."""
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for appearance in speaker_appearances:
        by_episode[int(appearance["sr_episode_id"])].append(appearance)
    for appearances in by_episode.values():
        appearances.sort(key=lambda row: int(row["speaker_index"]))

    records: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = int(episode["sr_episode_id"])
        appearances = by_episode.get(episode_id, [])
        missing = [row["speaker"] for row in appearances if not row.get("wikidata_id")]
        if missing and require_qids:
            raise ValueError(
                f"Episode {episode_id} has speakers without a Wikidata ID: "
                f"{', '.join(missing)}"
            )
        qids = list(
            dict.fromkeys(
                str(row["wikidata_id"])
                for row in appearances
                if row.get("wikidata_id")
            )
        )
        records.append({**episode, "episode_speakers": qids})
    return records


def attach_episode_speaker_ages(
    episodes: Iterable[dict[str, Any]],
    speaker_appearances: Iterable[dict[str, Any]],
    speaker_metadata: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach ages aligned with each episode's ordered speaker Q-IDs."""
    birth_dates = {
        str(record["wikidata_id"]): (
            record.get("birth_date"),
            record.get("death_date"),
        )
        for record in speaker_metadata
        if record.get("wikidata_id")
    }
    records: list[dict[str, Any]] = []
    for episode in episodes:
        def speaker_age(qid: str) -> int | None:
            birth_date, death_date = birth_dates.get(str(qid), (None, None))
            return age_at_date(birth_date, episode.get("date"), death_date)

        ages = [
            speaker_age(qid)
            for qid in episode.get("episode_speakers", [])
        ]
        records.append({**episode, "speaker_ages": ages})
    return records


def build_speaker_records(
    speaker_appearances: Iterable[dict[str, Any]],
    speaker_metadata: Iterable[dict[str, Any]],
    episodes: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Build one normalized speaker record per Wikidata item."""
    metadata_by_id = {
        str(record["wikidata_id"]): record for record in speaker_metadata
    }
    episode_dates = {
        int(episode["sr_episode_id"]): episode.get("date")
        for episode in episodes
    }
    grouped: dict[str, dict[str, Any]] = {}
    for appearance in speaker_appearances:
        qid = appearance.get("wikidata_id")
        if not qid:
            continue
        group = grouped.setdefault(
            str(qid),
            {"sr_names": [], "episode_ids": set(), "episode_ages": {}},
        )
        name = str(appearance["speaker"])
        if name not in group["sr_names"]:
            group["sr_names"].append(name)
        group["episode_ids"].add(int(appearance["sr_episode_id"]))
        episode_id = int(appearance["sr_episode_id"])
        episode_date = episode_dates.get(episode_id) or appearance.get("date")
        metadata = metadata_by_id.get(str(qid), {})
        group["episode_ages"][episode_id] = age_at_date(
            metadata.get("birth_date"),
            episode_date,
            metadata.get("death_date"),
        )

    records: list[dict[str, Any]] = []
    for qid, group in grouped.items():
        metadata = metadata_by_id.get(qid, {})
        records.append(
            {
                "wikidata_id": qid,
                # Appearances are chronological, so the latest SR credit is
                # the most useful default for people whose names changed.
                "speaker": group["sr_names"][-1],
                "sr_names": group["sr_names"],
                "episode_count": len(group["episode_ids"]),
                "episode_ids": sorted(group["episode_ids"]),
                "ages_at_episodes": [
                    group["episode_ages"].get(episode_id)
                    for episode_id in sorted(group["episode_ids"])
                ],
                **{
                    key: metadata.get(key, default.copy() if isinstance(default, list) else default)
                    for key, default in SPEAKER_METADATA_DEFAULTS.items()
                },
            }
        )
    return sorted(records, key=lambda record: record["wikidata_id"])


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
        if speaker_name in SPEAKER_QID_OVERRIDES:
            row["wikidata_id"] = SPEAKER_QID_OVERRIDES[speaker_name]
            enriched.append(row)
            continue
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
