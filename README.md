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

The default output is `data/episodes.parquet`. For a quick development run:

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

The MVP columns are:

- `sr_episode_id`
- `sr_audio_id`
- `source_title`
- `speaker`
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
