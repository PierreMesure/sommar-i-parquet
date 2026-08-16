# Topic modelling plan

## Goal

Create useful, reviewable topic metadata from Sommar i P1 and Vinter i P1
transcripts. The end result should support:

- multi-label episode tags;
- a browsable hierarchy of broad themes and narrower topics;
- a two-dimensional episode map;
- precomputed related episodes for a static website.

This is not a one-shot classifier. It is an auditable discovery and editorial
workflow: unsupervised clusters suggest themes, then reviewed clusters become
stable tags.

## Principles

- Keep clustering, representation, and editorial tagging separate.
- Do not remove stopwords or otherwise heavily normalise text before embedding:
  embedding models need natural language context.
- Remove only non-semantic noise before embedding, such as repeatable Sveriges
  Radio boilerplate and obvious duplicated ASR output.
- Apply stronger cleaning only to topic representations, after clustering.
- Cache all expensive artefacts, especially embeddings, and make every model
  run reproducible through a versioned configuration.
- Retain raw evidence alongside every generated label.

## Data preparation

The semantic chunker remains the source of timestamped units. It should expose
two derived text fields and quality metadata.

### `embedding_text`

Used for Harrier embeddings and clustering.

- Preserve normal Swedish prose, names, grammar, and stopwords.
- Remove repeated SR podcast/copyright disclaimers and obvious duplicated
  sentence loops.
- Do not remove song references automatically; lyrics and music can instead be
  identified through quality flags and reviewed later.

### `representation_text`

Used only by `CountVectorizer` and c-TF-IDF after clustering.

- Start from `embedding_text`.
- Remove Swedish stopwords and an explicit, versioned archive-specific list of
  generic spoken-language terms, for example `år`, `sen`, `tycker`, `vet`,
  `hela`, `gång`, and programme framing.
- Keep meaningful proper nouns: people, places, events, and historical periods
  can be useful topic evidence.
- Use Swedish unigrams and bigrams.

### Quality flags

Keep chunks in the raw archive but flag them for topic modelling:

- `is_boilerplate`
- `is_probable_lyrics`
- `is_repetitive_asr`
- `is_usable_content`

Flagged content is excluded from discovery/topic-share aggregation by default,
not deleted.

## Embeddings

Use direct Harrier embeddings for final semantic chunks, not word-weighted means
of short-unit embeddings. The latter are useful for chunk boundary detection but
lose longer-range context needed for topic clustering.

- Benchmark direct embeddings with a realistic 4k–8k token cap and batch size
  one on Apple Silicon.
- Store model name, quantisation, prompt setting, maximum length, chunk text
  hash, and preprocessing version in the embedding cache recipe.
- Use the same direct chunk vectors for topic discovery, episode coordinates,
  and related-episode similarity.

## Topic discovery

HDBSCAN is the primary discovery model. It can leave ambiguous chunks as
outliers, unlike K-means, which forces all chunks into a topic.

Do not fix a desired topic count for the initial model. Run a small reproducible
experiment grid with precomputed embeddings:

| Parameter | Initial values |
| --- | --- |
| UMAP `n_neighbors` | 5, 10, 15, 30 |
| UMAP `n_components` | 5, 10 |
| HDBSCAN `min_cluster_size` | 8, 12, 20, 30 |
| HDBSCAN `min_samples` | 1, 3, 5 |
| random state | fixed and recorded |

Every run should create an inspectable report containing:

- number of non-outlier topics;
- outlier rate;
- topic size distribution;
- separation in the original embedding space;
- c-TF-IDF terms after representation cleanup;
- representative chunks;
- automatically detected low-quality clusters.

Choose a run through coherence and human inspectability, not simply topic count
or outlier rate.

After selecting a coherent HDBSCAN model, evaluate BERTopic automatic reduction
with `nr_topics="auto"` as a conservative optional merge. Compare it against the
unreduced model; do not assume it improves results.

## Topic representation

Keep multiple representations for each cluster rather than replacing raw
evidence with an LLM label.

| Field | Purpose |
| --- | --- |
| `ctfidf_keywords` | Auditable statistical evidence |
| `diverse_keywords` | Less redundant keyword representation |
| `llm_label` | Short Swedish display label |
| `llm_summary` | One-sentence explanation |
| `representative_chunks` | Evidence for editorial review |
| `quality_status` | `candidate`, `approved`, `lyrics`, `boilerplate`, `asr_noise`, `merged` |

### c-TF-IDF

Use a `CountVectorizer` over `representation_text` with Swedish and
archive-specific stopwords, `ngram_range=(1, 2)`, and a tested `min_df`.
Test:

```python
ClassTfidfTransformer(
    bm25_weighting=True,
    reduce_frequent_words=True,
)
```

This improves topic labels without changing cluster membership.

### Multi-aspect representation

Use BERTopic multi-aspect representations to retain raw c-TF-IDF output while
adding a diversified-keyword representation and generated labels/summaries.

### LLM representation

Use BERTopic's own OpenAI representation mechanism, not an ad-hoc excerpt
selector:

- c-TF-IDF selects representative documents;
- use four documents with `diversity=0.1`;
- truncate each with a model tokenizer to 120 tokens;
- use a strict Swedish `topic: <label>` prompt;
- use `gpt-5.6-luna` with `reasoning_effort="none"` for labels and a separate
  summary prompt;
- store model/version and prompt metadata;
- preserve c-TF-IDF terms and excerpts for validation.

LLM labels make clusters legible. They do not make an incoherent cluster valid.

## Hierarchical topics

Hierarchical topic modelling is post-hoc agglomerative clustering over existing
topics. It does not determine the initial number of clusters and cannot repair
poor base topics.

After selecting a coherent HDBSCAN model and improving c-TF-IDF:

1. Build BERTopic's c-TF-IDF hierarchy.
2. Persist both a Parquet tree and interactive hierarchy HTML.
3. Label only meaningful parent nodes.
4. Review the tree to identify sensible manual merges and parent themes.
5. Preserve parent-child relationships rather than flattening them immediately.

For example:

```text
Kultur och skapande
├── Teater och scenkonst
├── Film och filmskapande
└── Författarskap och skrivande
```

Compare the documented c-TF-IDF hierarchy with an embedding-based hierarchy as
an experiment, but use human review before accepting either merge structure.

## Episode tags

Topic IDs are not final user-facing tags. Calculate multi-topic distributions
for usable chunks, then aggregate their weighted scores to episode level.

```json
{
  "episode_id": 123,
  "topics": [
    {"topic_id": 17, "score": 0.48},
    {"topic_id": 42, "score": 0.21}
  ]
}
```

Reviewed topics can later map to a narrower controlled taxonomy for the
frontend. Preserve both the discovered topic and any controlled tag mapping.

## Related episodes and map

Keep relatedness separate from BERTopic:

- aggregate direct usable-content chunk embeddings to an episode vector;
- precompute top-k cosine neighbours for every episode;
- store scores and reasons without requiring runtime vector search;
- derive stable two-dimensional coordinates from the same episode vectors.

This is lightweight enough for the static frontend.

## Implementation order

1. Freeze exploratory outputs; do not expose them as tags in the frontend.
2. Implement preprocessing and quality flags.
3. Produce cached direct chunk embeddings.
4. Implement the experiment runner and comparable review reports.
5. Select an HDBSCAN configuration from evidence.
6. Tune c-TF-IDF/vectorisation without retraining clusters.
7. Add multi-aspect representations and LLM labels/summaries.
8. Generate and review the hierarchy.
9. Produce episode-level topic distributions, controlled tags, map coordinates,
   and related episodes.
