"""Audit transcript JSON files, quarantine artifacts, and restore healthy files."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from src.whisper.quality import transcript_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts-dir", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.transcripts_dir
    faulty_root = root / "faulty"
    healthy: dict[str, list[Path]] = defaultdict(list)
    faulty: list[tuple[Path, list[dict]]] = []
    unreadable: list[str] = []
    for path in root.rglob("*.json"):
        if faulty_root in path.parents:
            continue
        try:
            transcript = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            unreadable.append(str(path))
            continue
        if not isinstance(transcript, dict) or "segments" not in transcript:
            continue
        artifacts = transcript_artifacts(transcript)
        if artifacts:
            faulty.append((path, artifacts))
        else:
            healthy[path.stem].append(path)

    restored: list[str] = []
    if args.apply:
        for path, artifacts in faulty:
            relative = path.relative_to(root)
            destination = faulty_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(path, destination)
        for episode_id, candidates in healthy.items():
            destination = root / f"{episode_id}.json"
            if destination.exists():
                continue
            source = max(candidates, key=lambda path: path.stat().st_mtime)
            if source.exists():
                shutil.move(source, destination)
                restored.append(episode_id)

    report = {
        "healthy_episode_ids": len(healthy),
        "faulty_files": len(faulty),
        "faulty_segments": sum(len(artifacts) for _, artifacts in faulty),
        "unreadable_files": unreadable,
        "restored_to_canonical": len(restored),
        "faulty": [
            {"path": str(path), "artifacts": artifacts}
            for path, artifacts in faulty
        ],
    }
    report_path = root / "audit-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "faulty"}))


if __name__ == "__main__":
    main()
