"""Download and cache the Wikidata data used to enrich episode records."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = (
    "sommar-i-parquet/0.1 "
    "(https://github.com/PierreMesure/sommar-i-parquet)"
)
RATE_LIMIT_RETRY_SECONDS = 65
LOGGER = logging.getLogger(__name__)
SEASON_PARTICIPANTS_QUERY = """
SELECT (GROUP_CONCAT(DISTINCT CONCAT(
  SUBSTR(STR(?date), 1, 10), "=",
  STRAFTER(STR(?speaker), "entity/")
); separator="|") AS ?date_qids)
WHERE {
  ?season wdt:P179 wd:Q7560435;
          p:P710 ?statement.
  ?statement ps:P710 ?speaker;
             pq:P585 ?date.
}
"""


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _retry_after_seconds(response: httpx.Response) -> int:
    """Use Wikidata's requested wait, whether seconds or an HTTP date."""
    retry_after = response.headers.get("Retry-After", "")
    try:
        return max(1, int(retry_after))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
            return max(1, round((retry_at - datetime.now(UTC)).total_seconds()))
        except (TypeError, ValueError):
            return RATE_LIMIT_RETRY_SECONDS


def download_season_participants(
    *,
    cache_dir: str | Path = "data/cache/wikidata",
) -> list[dict[str, Any]]:
    """Fetch qualified participant/date records from Wikidata season items."""
    path = Path(cache_dir) / "season_participants.json"
    cached = _load_json(path)
    if cached is not None:
        return cached

    try:
        with httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/sparql-results+json",
                "Accept-Encoding": "gzip, deflate",
            },
        ) as client:
            for attempt in range(2):
                response = client.get(
                    WIKIDATA_SPARQL_URL,
                    params={"query": SEASON_PARTICIPANTS_QUERY, "format": "json"},
                )
                if response.status_code != httpx.codes.TOO_MANY_REQUESTS:
                    response.raise_for_status()
                    value = response.json()["results"]["bindings"][0]["date_qids"][
                        "value"
                    ]
                    bindings = [
                        {
                            "speaker": {
                                "value": "http://www.wikidata.org/entity/" + qid
                            },
                            "date": {"value": date},
                        }
                        for pair in value.split("|")
                        for date, qid in [pair.split("=", maxsplit=1)]
                        if qid
                    ]
                    break
                if attempt == 0:
                    retry_seconds = _retry_after_seconds(response)
                    LOGGER.warning(
                        "Wikidata's query service rate-limited this request; retrying "
                        "once in %d seconds.",
                        retry_seconds,
                    )
                    time.sleep(retry_seconds)
                    continue
                response.raise_for_status()
    except httpx.HTTPError as error:
        LOGGER.warning(
            "Could not download Wikidata season participants (%s); "
            "continuing without Wikidata enrichment.",
            error,
        )
        return []
    _write_json(path, bindings)
    return bindings
