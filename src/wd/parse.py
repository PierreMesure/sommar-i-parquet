"""Match SR speakers to dated Wikidata season-participant statements."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


def enrich_episodes_with_wikidata(
    episodes: Iterable[dict[str, Any]],
    *,
    season_participants: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add QIDs when both sources have exactly one participant on a date."""
    season_matches_by_date: dict[str, set[str]] = {}
    for binding in season_participants:
        try:
            date = binding["date"]["value"][:10]
            qid = binding["speaker"]["value"].rsplit("/", 1)[-1]
            if qid:
                season_matches_by_date.setdefault(date, set()).add(qid)
        except KeyError:
            continue

    episode_rows = list(episodes)
    episode_dates = Counter(episode["date"] for episode in episode_rows)
    enriched: list[dict[str, Any]] = []
    for episode in episode_rows:
        row = dict(episode)
        season_ids = season_matches_by_date.get(row["date"], set())
        if episode_dates[row["date"]] == 1 and len(season_ids) == 1:
            row["wikidata_id"] = next(iter(season_ids))
            enriched.append(row)
            continue

        row["wikidata_id"] = None
        enriched.append(row)
    return enriched
