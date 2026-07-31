"""Download episode metadata from Sveriges Radio's open API."""

from __future__ import annotations

import logging
import json
from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

EPISODES_URL = "https://api.sr.se/api/v2/episodes/index"
MUSIC_URL = "https://api.sr.se/api/v2/playlists/getplaylistbyepisodeid"
PROGRAM_ID = 2071
USER_AGENT = "sommar-i-parquet/0.1"


def download_episodes(
    *,
    program_id: int = PROGRAM_ID,
    page_size: int = 100,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Download all episode-list pages for an SR program.

    ``max_pages`` is intended for fast development and test runs. When omitted,
    pagination continues until the API's reported final page.
    """
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive")

    transport = httpx.HTTPTransport(retries=3)
    episodes: list[dict[str, Any]] = []
    page = 1

    with httpx.Client(
        transport=transport,
        timeout=30.0,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        follow_redirects=True,
    ) as client:
        while True:
            logging.info("Downloading SR episode page %d", page)
            response = client.get(
                EPISODES_URL,
                params={
                    "programid": program_id,
                    "format": "json",
                    "size": page_size,
                    "page": page,
                },
            )
            response.raise_for_status()
            payload = response.json()

            page_episodes = payload.get("episodes", [])
            if not isinstance(page_episodes, list):
                raise TypeError("SR response field 'episodes' is not a list")
            episodes.extend(page_episodes)

            pagination = payload.get("pagination", {})
            total_pages = int(pagination.get("totalpages", page))
            if page >= total_pages:
                break
            if max_pages is not None and page >= max_pages:
                break
            page += 1

    return episodes


def download_music_playlists(
    episodes: Sequence[dict[str, Any]],
    *,
    episode_ids: Collection[int],
    cache_dir: str | Path = "data/cache/music",
    max_workers: int = 4,
    max_episodes: int | None = None,
) -> list[dict[str, Any]]:
    """Download official SR playlists, caching one JSON response per episode."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")

    selected = [episode for episode in episodes if int(episode["id"]) in episode_ids]
    if max_episodes is not None:
        selected = selected[:max_episodes]

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []

    for episode in selected:
        episode_id = int(episode["id"])
        path = cache_path / f"{episode_id}.json"
        if path.exists():
            results[episode_id] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(episode)

    logging.info(
        "Music playlists: %d cached, %d to download",
        len(results),
        len(missing),
    )
    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(
        transport=transport,
        timeout=30.0,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        follow_redirects=True,
    ) as client:

        def fetch(episode: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            episode_id = int(episode["id"])
            response = client.get(
                MUSIC_URL,
                params={"id": episode_id, "format": "json", "size": 100},
            )
            response.raise_for_status()
            return episode_id, response.json()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch, episode): episode for episode in missing}
            for completed, future in enumerate(as_completed(futures), start=1):
                episode_id, payload = future.result()
                results[episode_id] = payload
                (cache_path / f"{episode_id}.json").write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
                if completed % 25 == 0 or completed == len(missing):
                    logging.info(
                        "Downloaded %d/%d music playlists",
                        completed,
                        len(missing),
                    )

    return [
        {
            "sr_episode_id": int(episode["id"]),
            "publishdateutc": episode["publishdateutc"],
            "songs": results[int(episode["id"])].get("song", []),
        }
        for episode in selected
    ]
