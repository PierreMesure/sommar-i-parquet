"""Split one transcript into LLM-planned, timestamped chapters and label them."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.nlp.chapters import (
    ChapterLabels,
    ChapterPlan,
    chapter_segment_ranges,
    resolve_boundaries,
    transcript_text,
)


MODEL = "gpt-5.6-luna"
BOUNDARY_SYSTEM_PROMPT = """Du är redaktör för Sommar i P1 och Vinter i P1,
svenska radioprogram där en värd berättar personligt mellan musikinslag.
Läs hela utskriften och dela den i ett fåtal sammanhängande kapitel när ämne,
tidsperiod, berättelse eller perspektiv tydligt skiftar. Skapa inte kapitel vid
varje musikomnämnande. För varje övergång, återge 6–20 ord ordagrant från
alldeles före övergången. Hitta aldrig på ord och ta inte med programintro eller
programoutro som kapitelgränser."""
LABEL_SYSTEM_PROMPT = """Du är redaktör för Sommar i P1 och Vinter i P1.
Du får ett sammanhängande kapitel ur ett radioprogram. Skriv en kort svensk
rubrik, minst ett ämnesord eller en ämnesfras, och en saklig mening som
sammanfattar kapitlet. Hitta inte på detaljer och använd inte generiska etiketter
som 'radioprat' eller 'personliga minnen' när konkretare ämnen finns."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_id", type=int)
    parser.add_argument("--transcripts-dir", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/nlp/llm_chapters"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-input-characters", type=int, default=300_000)
    parser.add_argument("--minimum-anchor-score", type=float, default=0.64)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_response(client: OpenAI, *, model: str, system: str, user: str, schema: type[Any]) -> Any:
    """Call the Chat Completions structured-output helper without reasoning."""
    response = client.chat.completions.parse(
        model=model,
        reasoning_effort="none",
        temperature=0,
        max_completion_tokens=700,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
    )
    message = response.choices[0].message
    if message.refusal:
        raise RuntimeError(f"Model refused request: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("Model returned no structured response.")
    return message.parsed


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required in .env or the environment.")

    transcript_path = args.transcripts_dir / f"{args.episode_id}.json"
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = [segment for segment in payload.get("segments", []) if str(segment.get("text") or "").strip()]
    text = transcript_text(segments)
    if len(text) > args.max_input_characters:
        raise ValueError(
            f"Transcript has {len(text):,} characters, above --max-input-characters. "
            "Use a larger explicit limit or implement a multi-pass planner."
        )
    output_path = args.output_dir / f"{args.episode_id}.json"
    if output_path.exists() and not args.force:
        logging.info("Chapter analysis already exists: %s", output_path)
        return

    client = OpenAI()
    metadata = payload.get("sommar_i_parquet") or {}
    logging.info("Planning chapter boundaries for %s", metadata.get("source_title") or args.episode_id)
    plan = parse_response(
        client,
        model=args.model,
        system=BOUNDARY_SYSTEM_PROMPT,
        user="Här är hela utskriften:\n\n" + text,
        schema=ChapterPlan,
    )
    boundaries = resolve_boundaries(
        plan.boundaries,
        segments,
        minimum_score=args.minimum_anchor_score,
    )
    ranges = chapter_segment_ranges(len(segments), boundaries)
    chapters: list[dict[str, Any]] = []
    for index, (start_index, end_index) in enumerate(ranges, start=1):
        chapter_segments = segments[start_index:end_index + 1]
        chapter_text = transcript_text(chapter_segments)
        logging.info("Labelling chapter %d/%d", index, len(ranges))
        labels = parse_response(
            client,
            model=args.model,
            system=LABEL_SYSTEM_PROMPT,
            user="Kapitelutskrift:\n\n" + chapter_text,
            schema=ChapterLabels,
        )
        chapters.append(
            {
                "chapter_index": index,
                "start_seconds": float(chapter_segments[0]["start"]),
                "end_seconds": float(chapter_segments[-1]["end"]),
                "duration_seconds": float(chapter_segments[-1]["end"]) - float(chapter_segments[0]["start"]),
                "word_count": len(chapter_text.split()),
                "text": chapter_text,
                **labels.model_dump(),
            }
        )

    output = {
        "sr_episode_id": args.episode_id,
        "source_title": metadata.get("source_title"),
        "transcript_path": str(transcript_path),
        "transcript_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "model": args.model,
        "boundary_plan": plan.model_dump(),
        "resolved_boundaries": [boundary.__dict__ for boundary in boundaries],
        "chapters": chapters,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote %d labelled chapters to %s", len(chapters), output_path)


if __name__ == "__main__":
    main()
