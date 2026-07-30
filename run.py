"""Download the SR archive and write the MVP dataset as Parquet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.sr.download import download_episodes
from src.sr.parse import parse_episodes
from src.utils.write import write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Sommar & Vinter i P1 episodes from SR."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/episodes.parquet"),
        help="Output Parquet file (default: data/episodes.parquet)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Episodes requested per API page (default: 100)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Only fetch this many pages; useful for development.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw_episodes = download_episodes(
        page_size=args.page_size,
        max_pages=args.max_pages,
    )
    episodes = parse_episodes(raw_episodes)
    output = write_parquet(episodes, args.output)
    logging.info("Wrote %d episodes to %s", len(episodes), output)


if __name__ == "__main__":
    main()

