"""Audit transcript JSON files and optionally remove high-confidence noise."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from src.whisper.quality import (
    strip_artifacts,
    strip_boilerplate,
    strip_recommended_episode_extract,
    transcript_artifacts,
    transcript_boilerplate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts-dir", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Remove detected segments in place and retain them in transcript metadata.",
    )
    return parser.parse_args()


def _write_transcript(path: Path, transcript: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.transcripts_dir
    report_path = args.report or root / "audit-report.json"
    files: list[dict] = []
    unreadable: list[str] = []
    changed_files = 0
    transcript_files = 0

    for path in sorted(root.glob("*.json")):
        if path == report_path:
            continue
        try:
            transcript = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            unreadable.append(str(path))
            continue
        if not isinstance(transcript, dict) or "segments" not in transcript:
            continue
        transcript_files += 1
        boilerplate = transcript_boilerplate(transcript)
        artifacts = transcript_artifacts(transcript)
        recommended_extract = []
        if args.apply:
            recommended_extract = strip_recommended_episode_extract(transcript)
            # Recalculate after the complete trailing extract has gone.
            boilerplate = transcript_boilerplate(transcript)
            artifacts = transcript_artifacts(transcript)
        if boilerplate or artifacts or recommended_extract:
            files.append(
                {
                    "episode_id": int(path.stem) if path.stem.isdigit() else None,
                    "path": str(path),
                    "boilerplate": boilerplate,
                    "artifacts": artifacts,
                    "recommended_extract_segment_count": len(recommended_extract),
                }
            )
        if args.apply and (boilerplate or artifacts or recommended_extract):
            strip_boilerplate(transcript)
            # Recalculate indices after boilerplate segments have been removed.
            strip_artifacts(transcript, transcript_artifacts(transcript))
            _write_transcript(path, transcript)
            changed_files += 1

    boilerplate_reasons = Counter(
        reason
        for item in files
        for match in item["boilerplate"]
        for reason in match["reasons"]
    )
    artifact_reasons = Counter(
        reason
        for item in files
        for match in item["artifacts"]
        for reason in match["reasons"]
    )
    report = {
        "transcript_files": transcript_files,
        "flagged_files": len(files),
        "boilerplate_segments": sum(len(item["boilerplate"]) for item in files),
        "artifact_segments": sum(len(item["artifacts"]) for item in files),
        "boilerplate_reasons": dict(sorted(boilerplate_reasons.items())),
        "artifact_reasons": dict(sorted(artifact_reasons.items())),
        "changed_files": changed_files,
        "unreadable_files": unreadable,
        "episodes": [
            {"episode_id": item["episode_id"]}
            for item in files
            if item["episode_id"] is not None and item["artifacts"]
        ],
        "files": files,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "files"}))


if __name__ == "__main__":
    main()
