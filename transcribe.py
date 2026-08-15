"""Download and locally transcribe one Sommar/Vinter i P1 episode."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.whisper.download import download_file
from src.whisper.quality import quarantine_transcript, strip_artifacts, transcript_artifacts
from src.whisper.transcribe import transcribe_audio_whisperx


KB_WHISPER_MODEL = "KBLab/kb-whisper-large"
DEFAULT_MODEL_DIR = Path("data/models/kb-whisper-large-ct2")
DEFAULT_ALIGNMENT_MODEL_DIR = Path("data/models/whisperx-alignment")
DEFAULT_TRANSCRIPTS_DIR = Path("data/transcripts")


def episode_from_parquet(episode_id: int, episodes_path: Path) -> dict[str, Any]:
    """Look up a single SR episode, requiring an audio download URL."""
    for episode in pq.read_table(episodes_path).to_pylist():
        if episode["sr_episode_id"] == episode_id:
            if not episode.get("mp3_url"):
                raise ValueError(f"Episode {episode_id} has no MP3 URL.")
            return episode
    raise ValueError(f"Episode {episode_id} was not found in {episodes_path}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_id", type=int, help="SR episode ID from data/episodes.parquet")
    parser.add_argument("--episodes-path", type=Path, default=Path("data/episodes.parquet"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRANSCRIPTS_DIR)
    alignment_group = parser.add_mutually_exclusive_group()
    alignment_group.add_argument(
        "--force-align-words", dest="force_align_words", action="store_true", default=True,
        help="Run wav2vec2 forced alignment for word timestamps (the default).",
    )
    alignment_group.add_argument(
        "--no-align-words", dest="force_align_words", action="store_false",
        help="Skip forced alignment and omit word timestamps.",
    )
    parser.add_argument("--model", default=KB_WHISPER_MODEL)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-transcribe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = episode_from_parquet(args.episode_id, args.episodes_path)
    audio_path = args.output_dir / "audio" / f"{args.episode_id}.mp3"
    transcript_path = args.output_dir / f"{args.episode_id}.json"

    if transcript_path.exists() and not args.force_transcribe:
        logging.info("Transcript already exists: %s", transcript_path)
        return

    if not 1 <= args.batch_size <= 16:
        raise ValueError("--batch-size must be between 1 and 16 to avoid batched-decoding artifacts.")
    logging.info("Downloading %s", episode["source_title"])
    download_file(episode["mp3_url"], audio_path, overwrite=args.force_download)
    logging.info("Transcribing %s", episode["source_title"])
    transcript = transcribe_audio_whisperx(
        audio_path=audio_path,
        output_path=transcript_path,
        model=args.model,
        model_dir=args.model_dir,
        alignment_model_dir=DEFAULT_ALIGNMENT_MODEL_DIR,
        batch_size=args.batch_size,
        force_align_words=args.force_align_words,
    )
    transcript["sommar_i_parquet"].update(
        {
            "episode_id": episode["sr_episode_id"],
            "episode_url": episode["episode_url"],
            "mp3_url": episode["mp3_url"],
            "source_title": episode["source_title"],
        }
    )
    import json

    artifacts = transcript_artifacts(transcript)
    retrying = any((transcript_path.parent / "faulty").glob(f"{args.episode_id}-*.json"))
    if artifacts:
        transcript["sommar_i_parquet"]["artifacts"] = artifacts
        if retrying:
            strip_artifacts(transcript, artifacts)

    with transcript_path.open("w", encoding="utf-8") as output:
        json.dump(transcript, output, ensure_ascii=False, indent=2)
        output.write("\n")
    if artifacts and not retrying:
        faulty_path = quarantine_transcript(transcript_path, artifacts)
        logging.warning("Archived faulty transcript at %s", faulty_path)
        return
    if artifacts:
        logging.warning("Retried transcript still contained artifacts; stripped and retained them in metadata")
    logging.info("Wrote timestamped transcript: %s", transcript_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
