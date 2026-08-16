"""Generate reviewable Swedish definitions and near-miss examples with Luna."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from src.nlp.curated import parse_curated_topic_proposal


class TopicGuidance(BaseModel):
    topic_id: str
    description_sv: str = Field(description="One precise Swedish scope sentence.")
    positive_examples: list[str] = Field(
        min_length=3,
        max_length=3,
        description="Three short Swedish examples of content that belongs.",
    )
    negative_examples: list[str] = Field(
        min_length=2,
        max_length=2,
        description="Two plausible Swedish near misses that must not be tagged.",
    )


class TopicGuidanceBatch(BaseModel):
    topics: list[TopicGuidance]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=Path("SPECIFIC_TOPICS_PROPOSAL.md"))
    parser.add_argument("--output", type=Path, default=Path("data/nlp/curated/topic_guidance.json"))
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--batch-size", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required.")
    source = parse_curated_topic_proposal(args.proposal)
    client = OpenAI()
    completed: dict[str, TopicGuidance] = {}
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        completed = {
            row["topic_id"]: TopicGuidance.model_validate(row)
            for row in previous.get("topics", [])
        }
    for start in range(0, len(source), args.batch_size):
        batch = [topic for topic in source[start:start + args.batch_size] if topic.topic_id not in completed]
        if not batch:
            continue
        source_rows = [
            {
                "topic_id": topic.topic_id,
                "label": topic.label,
                "editorial_scope": topic.description,
            }
            for topic in batch
        ]
        prompt = """Skriv sök- och granskningsunderlag för följande redaktionellt
valda ämnen i arkivet Sommar och Vinter i P1. Returnera varje topic_id exakt en
gång.

För varje ämne:
- skriv en exakt svensk omfattningsmening;
- skriv tre korta positiva exempel på vad ett transkriptstycke kan handla om;
- skriv två trovärdiga negativa närträffar: text som delar ord eller ligger nära
  ämnet men som INTE ska få etiketten;
- negativa exempel får inte vara absurda halmgubbar;
- använd aldrig generiska kategorier som personliga berättelser eller radioprat;
- skilj noga på närliggande ämnen, exempelvis väder/klimat,
  missbruk/dryckeskultur och adoption/familjeminnen.

Ämnen:
""" + json.dumps(source_rows, ensure_ascii=False, separators=(",", ":"))
        response = client.responses.parse(
            model=args.model,
            input=[
                {
                    "role": "system",
                    "content": "Du är en försiktig svensk taxonomiredaktör som definierar tydliga semantiska gränser.",
                },
                {"role": "user", "content": prompt},
            ],
            text_format=TopicGuidanceBatch,
            reasoning={"effort": "none"},
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError(f"No structured guidance returned for batch {start // args.batch_size + 1}")
        expected = {topic.topic_id for topic in batch}
        actual = [topic.topic_id for topic in parsed.topics]
        if set(actual) != expected or len(actual) != len(set(actual)):
            raise ValueError(f"Invalid topic IDs in batch {start // args.batch_size + 1}")
        completed.update({topic.topic_id: topic for topic in parsed.topics})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "topics": [completed[topic.topic_id].model_dump() for topic in source if topic.topic_id in completed],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"Prepared {len(completed)}/{len(source)} topic definitions")


if __name__ == "__main__":
    main()
