"""Download and cache the Wikidata data used to enrich episode records."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = (
    "sommar-i-parquet/0.1 "
    "(https://github.com/PierreMesure/sommar-i-parquet)"
)
RATE_LIMIT_RETRY_SECONDS = 65
SPEAKER_METADATA_BATCH_SIZE = 100
LOGGER = logging.getLogger(__name__)
QID_RE = re.compile(r"^Q[1-9][0-9]*$")
SEASON_PARTICIPANTS_QUERY = """
SELECT ?speaker ?speakerLabel ?date
WHERE {
  ?season wdt:P179 wd:Q7560435;
          p:P710 ?statement.
  ?statement ps:P710 ?speaker;
             pq:P585 ?date.
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "sv,en".
  }
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
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch qualified participant/date records from Wikidata season items."""
    path = Path(cache_dir) / "season_participants.json"
    cached = _load_json(path)
    if cached is not None and not force_refresh:
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
                    bindings = response.json()["results"]["bindings"]
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


SPEAKER_METADATA_QUERY = """
PREFIX schema: <http://schema.org/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT DISTINCT ?speaker ?svArticle ?enArticle
       ?gender ?genderLabel
       ?birthDate ?birthPrecision ?deathDate ?deathPrecision
       ?citizenship ?citizenshipLabel
       ?occupation ?occupationLabel
WHERE {
  VALUES ?speaker { %s }
  OPTIONAL { ?speaker wdt:P21 ?gender. }
  OPTIONAL {
    ?speaker p:P569/psv:P569 ?birthValue.
    ?birthValue wikibase:timeValue ?birthDate;
                wikibase:timePrecision ?birthPrecision.
  }
  OPTIONAL {
    ?speaker p:P570/psv:P570 ?deathValue.
    ?deathValue wikibase:timeValue ?deathDate;
                wikibase:timePrecision ?deathPrecision.
  }
  OPTIONAL { ?speaker wdt:P27 ?citizenship. }
  OPTIONAL { ?speaker wdt:P106 ?occupation. }
  OPTIONAL {
    ?svArticle schema:about ?speaker;
               schema:isPartOf <https://sv.wikipedia.org/>.
  }
  OPTIONAL {
    ?enArticle schema:about ?speaker;
               schema:isPartOf <https://en.wikipedia.org/>.
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "sv,en". }
}
"""


def _download_speaker_metadata_bindings(qids: list[str]) -> list[dict[str, Any]]:
    values = " ".join(f"wd:{qid}" for qid in qids)
    query = SPEAKER_METADATA_QUERY % values
    with httpx.Client(
        timeout=30.0,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
            "Accept-Encoding": "gzip, deflate",
        },
    ) as client:
        for attempt in range(2):
            response = client.post(
                WIKIDATA_SPARQL_URL,
                params={"format": "json"},
                content=query.encode("utf-8"),
                headers={"Content-Type": "application/sparql-query"},
            )
            if response.status_code != httpx.codes.TOO_MANY_REQUESTS:
                response.raise_for_status()
                return response.json()["results"]["bindings"]
            if attempt == 0:
                retry_seconds = _retry_after_seconds(response)
                LOGGER.warning(
                    "Wikidata's query service rate-limited metadata enrichment; "
                    "retrying once in %d seconds.",
                    retry_seconds,
                )
                time.sleep(retry_seconds)
                continue
            response.raise_for_status()
    return []


def download_speaker_metadata(
    qids: list[str],
    *,
    cache_dir: str | Path = "data/cache/wikidata",
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Download cached, ungrouped SPARQL metadata rows for matched speakers."""
    path = Path(cache_dir) / "speaker_metadata_sparql.json"
    cached = _load_json(path) or {"qids": [], "bindings": []}
    requested_qids = sorted({qid for qid in qids if QID_RE.fullmatch(qid)})
    cached_qids = set() if force_refresh else set(cached.get("qids", []))
    bindings = [] if force_refresh else list(cached.get("bindings", []))
    missing_qids = [qid for qid in requested_qids if qid not in cached_qids]
    if not missing_qids:
        return bindings

    for start in range(0, len(missing_qids), SPEAKER_METADATA_BATCH_SIZE):
        batch = missing_qids[start : start + SPEAKER_METADATA_BATCH_SIZE]
        try:
            bindings.extend(_download_speaker_metadata_bindings(batch))
        except httpx.HTTPError as error:
            LOGGER.warning("Could not download Wikidata speaker metadata (%s).", error)
            break
        cached_qids.update(batch)
        _write_json(
            path,
            {"qids": sorted(cached_qids), "bindings": bindings},
        )
        if start + SPEAKER_METADATA_BATCH_SIZE < len(missing_qids):
            time.sleep(0.2)
    return bindings
