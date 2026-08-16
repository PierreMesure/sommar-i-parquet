"""Use an LLM to propose a reviewable two-level topic taxonomy.

This is deliberately post-hoc: it never changes BERTopic assignments. It
groups the already-labelled leaf clusters into broad, human-readable themes
and keeps an explicit list of clusters that should stay out of the browsing
taxonomy (lyrics, programme framing, or incoherent residual clusters).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from src.nlp.topics import write_parquet_rows


class ParentTopic(BaseModel):
    """One browseable theme in the proposed taxonomy."""

    label: str = Field(description="A concise Swedish 2–5-word parent label.")
    description: str = Field(description="A one-sentence Swedish scope note.")


class HierarchyProposal(BaseModel):
    """The stable top level of a non-destructive editorial proposal."""

    parent_topics: list[ParentTopic] = Field(description="Aim for 12–18 broad themes.")
    editorial_notes: list[str] = Field(description="Short Swedish caveats for a human reviewer.")


class TopicAssignment(BaseModel):
    topic_id: int
    parent_label: str | None = Field(
        description="One exact parent label, or null when the topic should be excluded from browsing."
    )
    status: str = Field(description="Exactly 'candidate' or 'excluded'.")


class TopicAssignments(BaseModel):
    assignments: list[TopicAssignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/nlp/topics_labeled.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/nlp/topic_hierarchy_proposal.parquet"))
    parser.add_argument("--json-output", type=Path, default=Path("data/nlp/topic_hierarchy_proposal.json"))
    parser.add_argument("--model", default="gpt-5.6-luna")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required.")

    rows = [
        row
        for row in pq.read_table(args.input).to_pylist()
        if int(row["topic_id"]) != -1 and not row.get("is_low_quality")
    ]
    source = [
        {
            "id": int(row["topic_id"]),
            "label": str(row.get("llm_label") or row["label"]),
            "keywords": [str(term) for term in (row.get("keywords") or [])[:6]],
            "episodes": int(row.get("episode_count") or 0),
        }
        for row in rows
    ]
    taxonomy_prompt = """Du är redaktör för ett svenskt radioarkiv. Nedan finns
finare, automatiskt upptäckta ämneskluster från Sommar och Vinter i P1. Bygg
ett förslag till en redaktionellt begriplig hierarki för bläddring, inte en
klassifikation av programmens personer.

Regler:
- Skapa 12–18 breda överordnade ämnen på svenska.
- Använd inga påhittade fakta. Behåll små, tydliga ämnen som underämnen.
- Svara med den givna strukturen på svenska.

Kluster:
""" + json.dumps(source, ensure_ascii=False, separators=(",", ":"))

    client = OpenAI()
    response = client.responses.parse(
        model=args.model,
        input=[
            {"role": "system", "content": "Du skapar försiktiga, granskningsbara svenska ämnestaxonomier."},
            {"role": "user", "content": taxonomy_prompt},
        ],
        text_format=HierarchyProposal,
        reasoning={"effort": "none"},
    )
    proposal = response.output_parsed
    if proposal is None:
        raise RuntimeError("The model did not return a structured hierarchy proposal.")

    # First obtain the top-level vocabulary. Assigning 201 IDs in one answer
    # is unnecessarily error-prone, so classify small independent batches
    # against this fixed taxonomy and validate every batch exactly.
    parent_labels = [parent.label for parent in proposal.parent_topics]
    assignments: dict[int, TopicAssignment] = {}
    batch_size = 24
    for start in range(0, len(source), batch_size):
        batch = source[start:start + batch_size]
        assignment_prompt = """Fördela varje av följande finskaliga ämneskluster
till exakt en av de givna överordnade etiketterna. Om ett kluster mest består
av sångtext, programavslut/radioprat eller osammanhängande brus: välj null som
parent_label och status 'excluded'. I övriga fall väljer du en etikett exakt
som den är skriven och status 'candidate'. Returnera varje input-ID exakt en
gång.

Överordnade etiketter:
""" + json.dumps(parent_labels, ensure_ascii=False) + "\n\nKluster:\n" + json.dumps(
            batch, ensure_ascii=False, separators=(",", ":")
        )
        assignment_response = client.responses.parse(
            model=args.model,
            input=[
                {"role": "system", "content": "Du är en noggrann svensk redaktör som klassificerar ämneskluster."},
                {"role": "user", "content": assignment_prompt},
            ],
            text_format=TopicAssignments,
            reasoning={"effort": "none"},
        )
        parsed = assignment_response.output_parsed
        if parsed is None:
            raise RuntimeError(f"No structured assignments returned for batch {start // batch_size + 1}.")
        expected_batch = {entry["id"] for entry in batch}
        actual_batch = [assignment.topic_id for assignment in parsed.assignments]
        if set(actual_batch) != expected_batch or len(actual_batch) != len(set(actual_batch)):
            raise ValueError(f"Invalid assignments in batch {start // batch_size + 1}.")
        for assignment in parsed.assignments:
            if assignment.status not in {"candidate", "excluded"}:
                raise ValueError(f"Unknown status {assignment.status!r} for {assignment.topic_id}.")
            if assignment.parent_label is not None and assignment.parent_label not in parent_labels:
                raise ValueError(f"Unknown parent label {assignment.parent_label!r}.")
            if assignment.status == "excluded" and assignment.parent_label is not None:
                raise ValueError(f"Excluded topic {assignment.topic_id} has a parent.")
            if assignment.status == "candidate" and assignment.parent_label is None:
                raise ValueError(f"Candidate topic {assignment.topic_id} has no parent.")
            assignments[assignment.topic_id] = assignment

    output_rows = [
        {
            **row,
            "parent_label": assignments[int(row["topic_id"])].parent_label,
            "taxonomy_status": assignments[int(row["topic_id"])].status,
        }
        for row in rows
    ]
    write_parquet_rows(output_rows, args.output)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(
            {
                **proposal.model_dump(),
                "assignments": [assignment.model_dump() for assignment in assignments.values()],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(output_rows)} leaf-topic assignments to {args.output}")
    print(f"Wrote hierarchy proposal to {args.json_output}")


if __name__ == "__main__":
    main()
