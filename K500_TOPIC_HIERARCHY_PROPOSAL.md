# K500 topic hierarchy proposal

This proposal merges the 500 raw BERTopic clusters without reclassifying any
chunk. It is therefore deterministic: a chunk's original K-means cluster is
the only input to the merge.

## Editorial method

The Luna labels were used only as a reading aid. Decisions were made by also
reviewing the c-TF-IDF keywords, representative transcript passages, source
episodes, and the number of chunks and episodes in each cluster.

A cluster is promoted only when those signals make the same useful promise to
a visitor. Generic personal narrative, song-heavy or damaged transcription,
name-driven mixtures, and non-reusable one-off fragments are explicitly
withheld. Narrow subjects are retained when the underlying passages are
coherent—for example international adoption, the DDR doping system, game
development, and the Ungernrevolten 1956.

The complete auditable mapping lives in
`src/nlp/manual_hierarchy.py`. Its validator requires every raw ID from 0 to
499 to occur exactly once, either in a visible topic or in an exclusion class.

## Result

- 172 specific topics
- 14 broad themes
- 367 raw clusters retained
- 133 raw clusters withheld
- 1,975 of 1,978 episodes retain at least one topic before any later display
  threshold
- 12,007 episode–topic associations
- Median 6 specific topics per covered episode (mean 6.08)

The broad themes are:

- Hälsa och liv
- Internationellt och historia
- Kultur och medier
- Mat och livsstil
- Musik
- Natur och klimat
- Platser och resor
- Politik och ekonomi
- Relationer och familj
- Samhälle och identitet
- Tro och livsåskådning
- Utbildning och arbetsliv
- Vetenskap och teknik
- Äventyr och sport

## Intentionally uncovered episodes

Three episodes currently contain only excluded raw clusters:

- Amanda Ooms 1999
- Kristina Lugn 1994
- Olof Wretling – Håret – Vinterprat 2021

This is preferable to inventing a weak subject for them. Their map position and
related episodes remain available independently of topic assignment.

## Generated review data

Running `uv run python merge_topic_hierarchy.py` writes the reviewable outputs
under `data/nlp/k500/hierarchy/`:

- `topics.parquet`
- `broad_topics.parquet`
- `raw_to_specific.parquet`
- `excluded_raw_topics.parquet`
- `chunk_topics.parquet`
- `episode_topics.parquet`
- `episode_broad_topics.parquet`
- `topics.json`

No OpenAI request, embedding inference, nearest-neighbour assignment, or
zero-shot classification is involved in this build step.
