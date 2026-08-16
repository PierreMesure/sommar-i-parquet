"""Review controlled topic boundaries with Luna and derive local thresholds."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from src.nlp.curated import (
    apply_topic_guidance,
    estimate_topic_threshold,
    parse_curated_topic_proposal,
    sample_topic_calibration_candidates,
)


class CandidateJudgment(BaseModel):
    chunk_id: str
    relevant: bool
    reason_sv: str = Field(description="Högst en kort svensk mening.")


class TopicJudgment(BaseModel):
    topic_id: str
    judgments: list[CandidateJudgment]


class CalibrationBatch(BaseModel):
    topics: list[TopicJudgment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=Path("SPECIFIC_TOPICS_PROPOSAL.md"))
    parser.add_argument(
        "--guidance",
        type=Path,
        default=Path("data/nlp/curated/topic_guidance.json"),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/nlp/curated/chunk_topic_decisions.parquet"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/nlp/curated/topic_calibration_reviews.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/nlp/curated/topic_thresholds.json"),
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--per-score-band", type=int, default=2)
    parser.add_argument("--default-threshold", type=float, default=0.53)
    parser.add_argument("--target-precision", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required")
    topics = apply_topic_guidance(
        parse_curated_topic_proposal(args.proposal),
        args.guidance,
    )
    topic_by_id = {topic.topic_id: topic for topic in topics}
    decisions = pq.read_table(args.decisions).to_pylist()
    sampled = sample_topic_calibration_candidates(
        decisions,
        per_score_band=args.per_score_band,
    )
    candidates_by_topic: dict[str, list[dict]] = defaultdict(list)
    for row in sampled:
        candidates_by_topic[str(row["best_topic_id"])].append(row)

    completed: dict[str, TopicJudgment] = {}
    if args.reviews.exists():
        previous = json.loads(args.reviews.read_text(encoding="utf-8"))
        completed = {
            row["topic_id"]: TopicJudgment.model_validate(row)
            for row in previous.get("topics", [])
        }
    pending_ids = [topic.topic_id for topic in topics if topic.topic_id not in completed]
    client = OpenAI()
    for start in range(0, len(pending_ids), args.batch_size):
        batch_ids = pending_ids[start:start + args.batch_size]
        batch_payload = []
        expected_candidates: dict[str, set[str]] = {}
        for topic_id in batch_ids:
            topic = topic_by_id[topic_id]
            candidates = candidates_by_topic.get(topic_id, [])
            expected_candidates[topic_id] = {str(row["chunk_id"]) for row in candidates}
            batch_payload.append(
                {
                    "topic_id": topic_id,
                    "label": topic.label,
                    "definition": topic.description,
                    "positive_examples": list(topic.positive_examples),
                    "negative_near_misses": list(topic.negative_examples),
                    "candidates": [
                        {
                            "chunk_id": row["chunk_id"],
                            "similarity": round(float(row["best_similarity"]), 4),
                            "runner_up_topic": row["runner_up_topic_label"],
                            "text": str(row["excerpt"])[:700],
                        }
                        for row in candidates
                    ],
                }
            )
        prompt = """Bedöm om varje transkriptstycke faktiskt handlar om det angivna
ämnet. Ett ord, namn eller en kort anspelning räcker inte: ämnet ska vara en
meningsfull del av stycket. Följ definitionen och uteslut negativa närträffar.
Ett stycke får vara relevant för flera ämnen, så bedöm endast det angivna ämnet.
Returnera varje topic_id och chunk_id exakt en gång.

Underlag:
""" + json.dumps(batch_payload, ensure_ascii=False, separators=(",", ":"))
        response = client.responses.parse(
            model=args.model,
            input=[
                {
                    "role": "system",
                    "content": "Du är en strikt svensk ämnesredaktör som prioriterar precision framför täckning.",
                },
                {"role": "user", "content": prompt},
            ],
            text_format=CalibrationBatch,
            reasoning={"effort": "none"},
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError(f"No structured calibration returned for batch {start // args.batch_size + 1}")
        actual_topics = {row.topic_id for row in parsed.topics}
        if actual_topics != set(batch_ids):
            raise ValueError(f"Calibration topic mismatch: expected {batch_ids}, got {sorted(actual_topics)}")
        for row in parsed.topics:
            actual_candidates = {judgment.chunk_id for judgment in row.judgments}
            if actual_candidates != expected_candidates[row.topic_id]:
                raise ValueError(f"Calibration candidate mismatch for {row.topic_id}")
            completed[row.topic_id] = row
        args.reviews.parent.mkdir(parents=True, exist_ok=True)
        args.reviews.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "topics": [completed[topic.topic_id].model_dump() for topic in topics if topic.topic_id in completed],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"Reviewed {len(completed)}/{len(topics)} topic boundaries")

    review_lookup = {
        topic_id: {judgment.chunk_id: judgment for judgment in review.judgments}
        for topic_id, review in completed.items()
    }
    thresholds: dict[str, float] = {}
    diagnostics: dict[str, dict] = {}
    for topic in topics:
        reviewed_rows = []
        for candidate in candidates_by_topic.get(topic.topic_id, []):
            judgment = review_lookup[topic.topic_id][str(candidate["chunk_id"])]
            reviewed_rows.append({**candidate, "relevant": judgment.relevant})
        threshold, stats = estimate_topic_threshold(
            reviewed_rows,
            default_threshold=args.default_threshold,
            target_precision=args.target_precision,
        )
        thresholds[topic.topic_id] = threshold
        diagnostics[topic.topic_id] = {"label": topic.label, **stats}
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "default_threshold": args.default_threshold,
                "target_review_precision": args.target_precision,
                "topic_min_similarities": thresholds,
                "diagnostics": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(thresholds)} reviewed topic thresholds to {args.output}")


if __name__ == "__main__":
    main()
