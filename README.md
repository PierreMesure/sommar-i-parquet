# Sommarpratkompassen

This repository contains script fetching information about the Swedish program *Sommar i P1* (and its younger sibling *Vinter i P1*). The program is also casually called *Sommarprat*.

You can read more about the program's history on [Wikipedia](https://sv.wikipedia.org/wiki/Sommar_i_P1). Its official page is on [Sveriges Radio's website](https://www.sverigesradio.se/sommar-i-p1).

## Context

*Sommarprat* are a bit of a cultural institution in Sweden and are usually listened by millions every year. Aware of the interest, Sveriges Radio keeps all the episodes online, at least the ones it kept a recording of.

Unfortunately, neither their app nor their website make it easy to browse such a large archive. It is hard to search, filter and find relevant episodes.

## Goal of the project

To find and gather structured information about the episodes of the programme and to publish them as open data so more people can find new episodes to listen and so more people can build innovative services to improve discovery.

With the advent of generative AI, people could ask what episodes they should listen to based on what they previously liked.

## Sources

I identified a few sources:

- Sveriges Radio, their official website has a [long list](https://www.sverigesradio.se/avsnitt/?programid=2071) going all the way back to 1960 and even private APIs that returns a [paginated list](https://www.sverigesradio.se/ajax/showmoreepisodelistitems?unitid=2071&page=0) as well as [episode metadata](https://web-api.sr.se/v1/player/ondemand?id=2815529&type=episode). They also have an unmaintained [open API](https://www.sverigesradio.se/artikel/this-is-swedish-radios-open-api).
- Wikipedia has a [page dedicated to the program](https://sv.wikipedia.org/wiki/Sommar_i_P1#Listor_över_sommar-_och_vintervärdar) as well as list pages for each decade ([1950](https://sv.wikipedia.org/wiki/Lista_över_sommarvärdar_under_1950-talet), [1960](https://sv.wikipedia.org/wiki/Lista_över_sommarvärdar_under_1960-talet), [1970](https://sv.wikipedia.org/wiki/Lista_över_sommarvärdar_under_1970-talet), ..., [2020](https://sv.wikipedia.org/wiki/Lista_över_sommarvärdar_under_2020-talet))
- Even better, Wikidata seems to have [the same data](https://w.wiki/SsC4). That opens a world of new possibilities since Wikidata's information is linked and structured, with a powerful API to fetch metadata related to each speaker.

## MVP

The first version downloads episode metadata from Sveriges Radio's open API and
writes a Zstandard-compressed Parquet file.

Install dependencies and fetch the complete archive:

```shell
uv sync
uv run python run.py
```

The default outputs are two normalized tables: one row per broadcast in
`data/episodes.parquet` and one row per Wikidata speaker in
`data/speakers.parquet`. A redundant join for browsing and debugging is written
to `data/speaker_appearances.parquet`. The same run also generates the compact
frontend dataset at `data/episodes.json`. For a quick development run:

```shell
uv run python run.py --max-pages 1 --output data/sample.parquet
```

The same run also writes one row per song play to `data/music.parquet`, using
SR's official episode-playlist API. Playlist responses are cached under
`data/cache/music`, so subsequent runs only fetch new episodes. Use
`--skip-music` to build only the episode table, or `--max-music-episodes 2` for
a small music development run.

By default, the parser removes trailers, podcast promotions, host-announcement
shows, anniversary and recap programmes, Q&As, alternate-language copies,
multi-guest specials, records without downloadable audio, and audio shorter
than 15 minutes. Use `--include-specials` to inspect the unfiltered SR archive.

The episode table contains one row per SR episode. Speaker relationships are
stored as an ordered list of Wikidata Q-IDs:

- `sr_episode_id`
- `sr_audio_id`
- `source_title`
- `date`
- `year`
- `program_type` (`Sommar`, `Vinter`, or null when the date is inconclusive)
- `episode_url`
- `mp3_url`
- `length_seconds`
- `audio_file_size_bytes`
- `image_url`
- `image_credit`
- `short_summary`
- `is_listeners_host`
- `episode_speakers`
- `speaker_ages` (ages aligned with `episode_speakers`)

The speaker table contains one row per Wikidata item, with:

- `wikidata_id`
- `speaker` (the most recent name used by SR)
- `sr_names`
- `episode_count`
- `episode_ids` and `ages_at_episodes`
- `wikidata_label`
- `wikidata_description`
- `wikipedia_url`
- `gender` and `gender_id`
- `birth_date` and `death_date`
- `citizenships` and `citizenship_ids`
- `occupations` and `occupation_ids`

Age values are calculated at the broadcast date. When Wikidata only provides a
birth year or month, the parser uses the lower possible age; dates after a
recorded death are treated as unmatched rather than producing an implausible
age.

`speaker_appearances.parquet` repeats the episode columns alongside these
appearance fields for convenient browsing. It is intentionally redundant; use
the episode and speaker tables as the canonical data.

Wikidata enrichment is stored once on the canonical speaker row. Episodes refer
to those rows through `episode_speakers`; this also represents programmes with
multiple hosts without duplicating episode metadata. A unique Wikidata
participant on a broadcast date is accepted as a conservative date-only match;
when several participants share a date, the Wikidata label must match the SR
speaker name.

Responses are cached under `data/cache/wikidata`. Pass `--refresh-wikidata`
to fetch the participant data again, or `--skip-wikidata` to omit this
optional enrichment.

The speaker metadata includes Swedish Wikipedia URLs where available (otherwise
English), gender, dates of birth and death, and labelled/Q-ID lists for
citizenship and occupation.

The music table contains:

- `sr_episode_id`
- `track_number`
- `title`
- `artist`
- `composer`
- `lyricist`
- `album`
- `record_label`
- `is_theme_song`

Run the tests with:

```shell
uv run pytest
```

## Static frontend

The `frontend/` directory contains a static Astro browser for the archive. Its
`public/episodes.json` and `public/topics.json` files are symbolic links to the
canonical datasets written under `data/`; run the relevant data pipelines
before starting the frontend.

Filters are divided into episode metadata (when), speaker metadata (who), and
episode content (what). The topic pipeline adds content filters, an episode map,
and precomputed related programmes. The archive remains usable when that
optional topic dataset has not been generated yet.

```shell
cd frontend
npm install
npm run dev
```

Create deployable static files with `npm run build`; the result is written to
`frontend/dist/`.

Development uses the local JSON symbolic links. Production builds fetch the
canonical datasets directly from the repository on GitHub.

## Local transcripts

`transcribe.py` downloads an episode MP3, then writes a timestamped JSON
transcript under `data/transcripts/`. By default it uses the MLX q4 conversion
of KBLab's Swedish KB-Whisper-large model, which runs directly on Apple
silicon's Metal GPU and includes word timestamps.

Transcribe a complete episode by its `sr_episode_id`:

```sh
uv run python transcribe.py 2550211
```

The first run downloads the q4 WhisperMLX model into `data/models/`; local
audio and transcripts are ignored by Git.

WhisperMLX performs VAD preprocessing and returns segment timestamps. For
word-level timestamps, enable forced alignment:

```sh
uv run python transcribe.py 2550211 --force-align-words
```

Without `--force-align-words`, WhisperMLX uses VAD but returns segment timestamps
only. Forced alignment downloads a Swedish wav2vec2 model on its first run. Speaker
diarization is intentionally not enabled yet: it requires a Hugging Face token
and accepting Pyannote's model agreement.

To transcribe the entire archive sequentially (and safely resume later), run:

```sh
uv run transcribe_all.py
```

New transcriptions remove only position-sensitive, high-confidence SR framing
(for example podcast-version, copyright, SR Play, and music-list notices). The
original removed segments remain under
`sommar_i_parquet.removed_introduction_segments`, with detection evidence under
`removed_introduction_segment_audit`. High-confidence ASR repetition loops are archived
for one retry; if they recur, the affected segments are retained in metadata and
removed from the usable transcript.

Audit the existing corpus without changing transcripts:

```sh
uv run python audit_transcripts.py
```

The report is written to `data/transcripts/audit-report.json`. After reviewing
it, apply the same reversible cleanup to existing transcripts with:

```sh
uv run python audit_transcripts.py --apply
```

To transcribe only episode IDs listed by an audit or retry manifest, pass a JSON
file containing an `episodes` array with `episode_id` values:

```sh
uv run python transcribe_all.py --manifest data/transcripts/retry-manifest.json
```

## Transcript topic modelling

`topic_model.py` turns complete transcript JSON files into semantically coherent,
timestamped chunks and fits an initial BERTopic model. By default it uses UMAP
and HDBSCAN to discover dense subjects without choosing a topic count or forcing
ambiguous chunks into a cluster. These are working groups to review, merge, and
name — not final editorial tags. Spherical K-means remains available for
fixed-vocabulary experiments with `--clusterer kmeans --n-topics 80`.

For embeddings it uses the
Apple-Silicon-native 4-bit MLX conversion of Harrier 270M
(`majentik/harrier-oss-v1-270m-MLX-4bit`) without an instruction prefix: Harrier
uses instructions on the query side, whereas transcript units are documents.
The unquantized Sentence Transformers backend remains available
with `--embedding-backend sentence-transformers`.

Chunk boundaries combine two signals:

- a change between rolling Harrier embeddings on either side of the boundary;
- the pause between transcript segments, used as evidence of an edited music
  break.

Pauses of at least 1.5 seconds are preserved as candidates and pauses of at
least 5 seconds receive a strong music-break score. Neither is an unconditional
cut: a dynamic program chooses the best boundaries for the full episode while
keeping chunks between the configured minimum and maximum lengths. All
candidate scores and decisions are written to `boundaries.parquet` so the
heuristic can be audited and retuned.

The target chunk size is 450 words, but the default maximum is 1,800 words so a
coherent chapter can remain intact. To avoid embedding those long texts a second
time, final chunk vectors are, by default, word-weighted means of the already
computed small-unit vectors. Use `--chunk-embedding-strategy direct` only when
testing full-chunk embeddings on a machine with ample memory.

Run the complete available transcript corpus with:

```sh
uv run python topic_model.py
```

Tune discovery density when reviewing it:

```sh
uv run python topic_model.py --min-cluster-size 12 --min-samples 3
uv run python topic_model.py --min-cluster-size 20 --min-samples 5
```

To generate readable Swedish labels for the discovered clusters, opt in to
BERTopic's OpenAI representation model. It selects four representative chunks
per cluster using c-TF-IDF, diversifies them, and truncates each to 120 actual
model tokens before sending them to the API. The raw c-TF-IDF keywords remain
in the output alongside the generated label for review.

```sh
uv run python topic_model.py --llm-label-model gpt-5.6-luna
```

This requires `OPENAI_API_KEY` in `.env`. The LLM is instructed to return only
a 2–6 word Swedish label and uses `reasoning_effort="none"`; it labels clusters
but does not alter their membership.

For a smaller experiment, or to inspect chaptering before fitting BERTopic:

```sh
uv run python topic_model.py --limit 10
uv run python topic_model.py --chunks-only
```

For the lowest local memory usage, reduce the unit batch size further:

```sh
uv run python topic_model.py --batch-size 4
```

Outputs are written under `data/nlp/`. They include semantic chunks, boundary
diagnostics, chunk and episode topic assignments, topic keywords and examples,
a two-dimensional episode map, top-eight cosine neighbours in
`related_episodes.parquet`, and the compact static frontend dataset
`topics.json`. Similarity and map coordinates come from the original episode
embedding space, not from browser-side calculations or distances on the 2D map.
Embedding caches are incremental: newly completed transcripts are added on the
next run without recomputing unchanged text.

After applying the reviewed topic assignments, rebuild the frontend topic
payload with:

```sh
uv run python build_curated_frontend.py
```

This keeps the curated string topic IDs, Swedish labels, provisional broad
themes, per-episode coverage, and the existing map/related-episode data in the
same compact `topics.json` consumed by the static site.

### Controlled editorial topics

The unsupervised clusters are discovery material. The reviewed leaf taxonomy in
`SPECIFIC_TOPICS_PROPOSAL.md` is assigned separately by `curated_topics.py`.
Harrier embeds transcript chunks as documents and the Swedish topic definitions
as `sts_query` queries, preserving the model's asymmetric retrieval setup.

```sh
uv run python curated_topics.py --batch-size 16 --mlx-cache-limit-mb 1024
```

Assignment is precision-first and deterministic once the embeddings and
thresholds are fixed:

- a strong chunk match must clear the topic floor and normally lead its
  runner-up by a minimum margin;
- a borderline match below 0.515 cannot establish an episode tag on its own;
- a very strong match can survive overlap between two related leaf topics;
- weaker supporting matches only establish an episode tag when they recur in
  multiple chunks and cover a meaningful portion of the episode;
- negative near-miss examples are embedded for auditing but are not automatic
  vetoes, because closely worded exclusions can also resemble valid passages;
- unmatched chunks remain unmatched instead of being forced into the taxonomy.

The controlled outputs live under `data/nlp/curated/`:

- `topics.parquet`: stable topic IDs, labels, definitions and examples;
- `chunk_topics.parquet`: accepted strong and supporting evidence;
- `chunk_topic_decisions.parquet`: the winner, runner-up, thresholds and
  rejection reason for every chunk;
- `episode_topics.parquet`: recurring/strong evidence aggregated per episode;
- `topic_calibration_samples.parquet`: score-stratified examples for reviewing
  all topic boundaries.

Reviewed per-topic floors can be placed in
`data/nlp/curated/topic_thresholds.json`; the assignment command loads that file
automatically. `calibrate_curated_topics.py` can prepare those thresholds with
Luna, but it sends the sampled transcript excerpts to the configured OpenAI API
and should therefore only be run as an explicit opt-in:

```sh
uv run python calibrate_curated_topics.py
uv run python curated_topics.py
```
