"""Download the SR archive and write the MVP dataset as Parquet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.sr.download import download_episodes, download_music_playlists
from src.sr.parse import parse_episodes, parse_music_playlists
from src.utils.write import MUSIC_SCHEMA, write_parquet


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
    parser.add_argument(
        "--include-specials",
        action="store_true",
        help="Keep trailers, announcements, short items, and other non-host episodes.",
    )
    parser.add_argument(
        "--music-output",
        type=Path,
        default=Path("data/music.parquet"),
        help="Output music Parquet file (default: data/music.parquet)",
    )
    parser.add_argument(
        "--music-from-year",
        type=int,
        default=2011,
        help="Earliest year to query for music metadata (default: 2011)",
    )
    parser.add_argument(
        "--music-workers",
        type=int,
        default=4,
        help="Concurrent SR music requests (default: 4)",
    )
    parser.add_argument(
        "--max-music-episodes",
        type=int,
        default=None,
        help="Only fetch this many music playlists; useful for development.",
    )
    parser.add_argument(
        "--skip-music",
        action="store_true",
        help="Only build the episode table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw_episodes = download_episodes(
        page_size=args.page_size,
        max_pages=args.max_pages,
    )
    episodes = parse_episodes(
        raw_episodes,
        include_specials=args.include_specials,
    )
    output = write_parquet(episodes, args.output)
    logging.info("Wrote %d episodes to %s", len(episodes), output)

    if not args.skip_music:
        music_episode_ids = {
            episode["sr_episode_id"]
            for episode in episodes
            if episode["year"] >= args.music_from_year
        }
        raw_playlists = download_music_playlists(
            raw_episodes,
            episode_ids=music_episode_ids,
            max_workers=args.music_workers,
            max_episodes=args.max_music_episodes,
        )
        music = parse_music_playlists(raw_playlists)
        music_output = write_parquet(
            music,
            args.music_output,
            schema=MUSIC_SCHEMA,
        )
        logging.info("Wrote %d song plays to %s", len(music), music_output)


if __name__ == "__main__":
    main()
