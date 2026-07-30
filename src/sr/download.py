"""Download episode metadata from Sveriges Radio's open API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

EPISODES_URL = "https://api.sr.se/api/v2/episodes/index"
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
                raise ValueError("SR response field 'episodes' is not a list")
            episodes.extend(page_episodes)

            pagination = payload.get("pagination", {})
            total_pages = int(pagination.get("totalpages", page))
            if page >= total_pages:
                break
            if max_pages is not None and page >= max_pages:
                break
            page += 1

    return episodes

