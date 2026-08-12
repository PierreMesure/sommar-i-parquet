# Sommar i Parquet

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
`public/episodes.json` is a symbolic link to the canonical `data/episodes.json`
written by `run.py`; run the data pipeline before starting the frontend.

Filters are divided into episode metadata (when), speaker metadata (who), and
episode content (what). The first two use the normalized episode and speaker
records. Content topics and keywords will be added after transcript analysis.

```shell
cd frontend
npm install
npm run dev
```

Create deployable static files with `npm run build`; the result is written to
`frontend/dist/`.

Development uses the local `/episodes.json` symbolic link. Production builds
fetch the canonical JSON directly from the repository on GitHub.

## Local transcripts

`transcribe.py` downloads an episode MP3, then writes a timestamped JSON
transcript under `data/transcripts/`. It uses the official CTranslate2 files
from KBLab's Swedish KB-Whisper-large model through WhisperX on an NVIDIA CUDA
GPU, with FP16 inference and batched transcription.

Transcribe a complete episode by its `sr_episode_id`:

```sh
uv run python transcribe.py 2550211
```

The first run downloads the official CTranslate2 model into `data/models/`; it
also downloads WhisperX's Swedish wav2vec2 alignment model. Local models, audio,
and transcripts are ignored by Git.

WhisperX performs VAD preprocessing, batched ASR, and forced alignment. Word
timestamps are enabled by default; to skip the alignment step:

```sh
uv run transcribe.py 2550211 --no-align-words
```

The default ASR batch size is 32 for the server's 48 GB RTX A6000. Reduce it if
the GPU is shared, for example `--batch-size 16`. Speaker diarization is
intentionally not enabled yet: it requires a Hugging Face token and accepting
Pyannote's model agreement.

To transcribe the entire archive sequentially (and safely resume later), run:

```sh
uv run transcribe_all.py
```

Pass `--batch-size` to tune GPU memory use. The all-episodes command reuses its
models within each worker and writes word-level timestamps by default.
