"""Sequentially transcribe every retained Sommar/Vinter episode.

Existing transcripts are skipped by ``transcribe.py``, so this command is safe
to interrupt and resume.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from src.whisper.download import download_file, download_huggingface_model
from src.whisper.transcribe import WhisperMLXSession
from transcribe import (
    DEFAULT_ALIGNMENT_MODEL_DIR,
    DEFAULT_MLX_MODEL_PATH,
    DEFAULT_TRANSCRIPTS_DIR,
    MLX_MODEL_REPO,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-path", type=Path, default=Path("data/episodes.parquet"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = pq.read_table(args.episodes_path).to_pylist()
    total = len(episodes)
    model_path = DEFAULT_MLX_MODEL_PATH
    if not (model_path / "weights.safetensors").exists():
        download_huggingface_model(MLX_MODEL_REPO, model_path)
    session = WhisperMLXSession(
        model_path=model_path,
        alignment_model_dir=DEFAULT_ALIGNMENT_MODEL_DIR,
        force_align_words=False,
    )

    for index, episode in enumerate(episodes, start=1):
        episode_id = episode["sr_episode_id"]
        print(f"[{index}/{total}] Episode {episode_id}", flush=True)
        output_path = DEFAULT_TRANSCRIPTS_DIR / f"{episode_id}.json"
        if output_path.exists():
            print(f"INFO Transcript already exists: {output_path}", flush=True)
            continue
        audio_path = DEFAULT_TRANSCRIPTS_DIR / "audio" / f"{episode_id}.mp3"
        print(f"INFO Downloading {episode.get('source_title', episode_id)}", flush=True)
        download_file(episode["mp3_url"], audio_path)
        print(f"INFO Transcribing {episode.get('source_title', episode_id)}", flush=True)
        transcript = session.transcribe(audio_path)
        transcript["sommar_i_parquet"] = {
            "audio_path": str(audio_path),
            "engine": "whispermlx (reused MLX ASR + Silero VAD)",
            "model_path": str(model_path),
            "episode_id": episode_id,
            "episode_url": episode.get("episode_url"),
            "mp3_url": episode.get("mp3_url"),
            "source_title": episode.get("source_title"),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with output_path.open("w", encoding="utf-8") as output:
            json.dump(transcript, output, ensure_ascii=False, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
