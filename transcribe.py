"""Download and locally transcribe one Sommar/Vinter i P1 episode."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.whisper.download import download_file, download_huggingface_model
from src.whisper.transcribe import transcribe_audio_whispermlx


MLX_MODEL_REPO = "jegeblad/kb-whisper-large-mlx-q4"
DEFAULT_MLX_MODEL_PATH = Path("data/models/kb-whisper-large-mlx-q4")
DEFAULT_ALIGNMENT_MODEL_DIR = Path("data/models/whispermlx-alignment")
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
    parser.add_argument(
        "--force-align-words",
        action="store_true",
        help="With WhisperMLX, run wav2vec2 forced alignment for word timestamps.",
    )
    parser.add_argument("--model-path", type=Path)
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

    model_path = args.model_path or DEFAULT_MLX_MODEL_PATH
    model_ready_path = model_path / "weights.safetensors"
    if not model_ready_path.exists():
        logging.info("Downloading MLX KB-Whisper-large model to %s", model_path)
        download_huggingface_model(MLX_MODEL_REPO, model_path)
    logging.info("Downloading %s", episode["source_title"])
    download_file(episode["mp3_url"], audio_path, overwrite=args.force_download)
    logging.info("Transcribing %s", episode["source_title"])
    transcript = transcribe_audio_whispermlx(
        audio_path=audio_path,
        output_path=transcript_path,
        model_path=model_path,
        alignment_model_dir=DEFAULT_ALIGNMENT_MODEL_DIR,
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

    with transcript_path.open("w", encoding="utf-8") as output:
        json.dump(transcript, output, ensure_ascii=False, indent=2)
        output.write("\n")
    logging.info("Wrote timestamped transcript: %s", transcript_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
