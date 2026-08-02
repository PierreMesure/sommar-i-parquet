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
}
QID_RE = re.compile(r"^Q[1-9][0-9]*$")


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
        speaker_label = SPEAKER_LABEL_OVERRIDES.get(
            str(row["speaker"]), str(row["speaker"])
        )
        matched_ids = {
            qid
            for qid, label in candidates
            if label and _normalise_name(label) == _normalise_name(speaker_label)
        }
        if len(matched_ids) == 1:
            row["wikidata_id"] = next(iter(matched_ids))
        elif len(candidate_ids) == 1 and speaker_counts[row.get("sr_episode_id")] == 1:
            row["wikidata_id"] = next(iter(candidate_ids))
        else:
            row["wikidata_id"] = None
        enriched.append(row)
    return enriched
