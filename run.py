"""Download the SR archive and write the MVP dataset as Parquet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.sr.download import download_episodes, download_music_playlists
from src.sr.parse import (
    episode_metadata,
    parse_episodes,
    parse_music_playlists,
    parse_speakers,
)
from src.utils.write import (
    MUSIC_SCHEMA,
    SPEAKER_APPEARANCE_SCHEMA,
    SPEAKER_SCHEMA,
    write_frontend_json,
    write_parquet,
)
from src.wd.download import download_season_participants, download_speaker_metadata
from src.wd.parse import (
    attach_episode_speakers,
    attach_episode_speaker_ages,
    build_speaker_records,
    enrich_speakers_with_wikidata,
    parse_speaker_metadata,
)


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
        "--speakers-output",
        type=Path,
        default=Path("data/speakers.parquet"),
        help="Normalized speaker Parquet output (default: data/speakers.parquet)",
    )
    parser.add_argument(
        "--speaker-appearances-output",
        type=Path,
        default=Path("data/speaker_appearances.parquet"),
        help="Enriched browseable speaker-appearance output",
    )
    parser.add_argument(
        "--frontend-output",
        type=Path,
        default=Path("data/episodes.json"),
        help="Compact JSON output for the static frontend (default: data/episodes.json)",
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
    parser.add_argument(
        "--skip-wikidata",
        action="store_true",
        help="Only build SR-derived data, without Wikidata enrichment.",
    )
    parser.add_argument(
        "--refresh-wikidata",
        action="store_true",
        help="Refresh the cached Wikidata participant data.",
    )
    parser.add_argument(
        "--allow-missing-wikidata",
        action="store_true",
        help="Keep speaker appearances that cannot yet be matched to Wikidata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    raw_episodes = download_episodes(
        page_size=args.page_size,
        max_pages=args.max_pages,
    )
    episodes = parse_episodes(
        raw_episodes,
        include_specials=args.include_specials,
    )
    speaker_appearances = parse_speakers(episodes)
    if not args.skip_wikidata:
        season_participants = download_season_participants(
            force_refresh=args.refresh_wikidata,
        )
        speaker_appearances = enrich_speakers_with_wikidata(
            speaker_appearances,
            episodes=episodes,
            season_participants=season_participants,
        )

    metadata = (
        download_speaker_metadata(
            [
                speaker["wikidata_id"]
                for speaker in speaker_appearances
                if speaker["wikidata_id"]
            ],
            force_refresh=args.refresh_wikidata,
        )
        if not args.skip_wikidata
        else []
    )
    speaker_metadata = parse_speaker_metadata(metadata)
    episode_rows = attach_episode_speakers(
        episode_metadata(episodes),
        speaker_appearances,
        require_qids=not args.skip_wikidata and not args.allow_missing_wikidata,
    )
    episode_rows = attach_episode_speaker_ages(
        episode_rows,
        speaker_appearances,
        speaker_metadata,
    )
    speakers = build_speaker_records(
        speaker_appearances,
        speaker_metadata,
        episodes=episode_rows,
    )

    output = write_parquet(episode_rows, args.output)
    logging.info("Wrote %d episodes to %s", len(episodes), output)
    speakers_output = write_parquet(
        speakers,
        args.speakers_output,
        schema=SPEAKER_SCHEMA,
    )
    logging.info("Wrote %d speakers to %s", len(speakers), speakers_output)

    episodes_by_id = {episode["sr_episode_id"]: episode for episode in episode_rows}
    enriched_appearances = [
        {**episodes_by_id[speaker["sr_episode_id"]], **speaker}
        for speaker in speaker_appearances
    ]
    appearances_output = write_parquet(
        enriched_appearances,
        args.speaker_appearances_output,
        schema=SPEAKER_APPEARANCE_SCHEMA,
    )
    logging.info(
        "Wrote %d enriched speaker appearances to %s",
        len(enriched_appearances),
        appearances_output,
    )
    frontend_output = write_frontend_json(episode_rows, speakers, args.frontend_output)
    logging.info("Wrote %d frontend episodes to %s", len(episodes), frontend_output)

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
