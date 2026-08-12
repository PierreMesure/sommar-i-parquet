"""Sequentially transcribe every retained Sommar/Vinter episode.

Existing transcripts are skipped by ``transcribe.py``, so this command is safe
to interrupt and resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from src.whisper.download import download_file
from src.whisper.transcribe import WhisperXSession
from transcribe import (
    DEFAULT_ALIGNMENT_MODEL_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_TRANSCRIPTS_DIR,
    KB_WHISPER_MODEL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-path", type=Path, default=Path("data/episodes.parquet"))
    parser.add_argument(
        "--episodes-per-worker",
        type=int,
        default=10,
        help="Restart the WhisperX worker after this many episodes (default: 10).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    # These options are used by the parent process to launch bounded workers.
    parser.add_argument("--worker-start", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-end", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_worker(episodes: list[dict], start: int, end: int, batch_size: int) -> None:
    """Transcribe one bounded batch in a process that can be discarded."""
    session = WhisperXSession(
        model=KB_WHISPER_MODEL,
        model_dir=DEFAULT_MODEL_DIR,
        alignment_model_dir=DEFAULT_ALIGNMENT_MODEL_DIR,
        batch_size=batch_size,
    )

    total = len(episodes)
    for index, episode in enumerate(episodes[start:end], start=start + 1):
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
            "engine": "whisperx (reused KB-Whisper-large FP16 + Silero VAD + wav2vec2 alignment)",
            "model": KB_WHISPER_MODEL,
            "model_dir": str(DEFAULT_MODEL_DIR),
            "alignment_model_dir": str(DEFAULT_ALIGNMENT_MODEL_DIR),
            "batch_size": batch_size,
            "force_align_words": True,
            "episode_id": episode_id,
            "episode_url": episode.get("episode_url"),
            "mp3_url": episode.get("mp3_url"),
            "source_title": episode.get("source_title"),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output:
            json.dump(transcript, output, ensure_ascii=False, indent=2)
            output.write("\n")


def main() -> None:
    args = parse_args()
    if args.episodes_per_worker < 1:
        raise ValueError("--episodes-per-worker must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    episodes = pq.read_table(args.episodes_path).to_pylist()
    total = len(episodes)
    if args.worker_start is not None or args.worker_end is not None:
        if args.worker_start is None or args.worker_end is None:
            raise ValueError("Both --worker-start and --worker-end are required together.")
        run_worker(episodes, args.worker_start, args.worker_end, args.batch_size)
        return

    script_path = Path(__file__).resolve()
    for start in range(0, total, args.episodes_per_worker):
        end = min(start + args.episodes_per_worker, total)
        batch_number = start // args.episodes_per_worker + 1
        batch_count = (total + args.episodes_per_worker - 1) // args.episodes_per_worker
        if all(
            (DEFAULT_TRANSCRIPTS_DIR / f"{episode['sr_episode_id']}.json").exists()
            for episode in episodes[start:end]
        ):
            logging.info("Skipping worker %s/%s; all transcripts already exist", batch_number, batch_count)
            continue
        logging.info(
            "Starting worker %s/%s for episodes %s-%s",
            batch_number,
            batch_count,
            start + 1,
            end,
        )
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--episodes-path",
                str(args.episodes_path),
                "--episodes-per-worker",
                str(args.episodes_per_worker),
                "--batch-size",
                str(args.batch_size),
                "--worker-start",
                str(start),
                "--worker-end",
                str(end),
            ],
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            check=True,
        )
        logging.info("Worker %s/%s finished; its MLX memory has been released", batch_number, batch_count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
