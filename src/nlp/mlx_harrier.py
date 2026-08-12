"""MLX support for quantized Harrier embedding checkpoints.

The published 270M MLX conversion has no Sentence Transformers dense head. The
generic ``mlx-embeddings`` Gemma adapter currently assumes that head exists and
mean-pools tokens, whereas Harrier uses causal attention and last-token pooling.
This module supplies that small model-specific adapter without modifying the
installed package.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


PROMPTS = {
    "sts_query": "Instruct: Retrieve semantically similar text\nQuery: ",
    "web_search_query": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query: "
    ),
    "bitext_query": "Instruct: Retrieve parallel sentences\nQuery: ",
}


def _patch_gemma3_text_adapter() -> None:
    """Make mlx-embeddings' Gemma adapter faithfully run Harrier 270M."""
    import mlx.core as mx
    from mlx_embeddings.models import gemma3_text
    from mlx_embeddings.models.base import BaseModelOutput, normalize_embeddings

    model_class = gemma3_text.Model
    if getattr(model_class, "_sommar_harrier_patch", False):
        return

    original_init = model_class.__init__

    def patched_init(self, config):  # type: ignore[no-untyped-def]
        original_init(self, config)
        # harrier-oss-v1-270m has Transformer -> Pooling -> Normalize only.
        self.dense = []

    def patched_call(self, input_ids, attention_mask=None):  # type: ignore[no-untyped-def]
        inputs = input_ids
        if attention_mask is None:
            attention_mask = mx.ones(inputs.shape, dtype=mx.int32)
        # Omitting the mask preserves Gemma's causal attention. With causal
        # attention, right-padding cannot affect the last non-padding token.
        hidden_states = self.model(inputs)
        positions = mx.sum(attention_mask, axis=-1).astype(mx.int32) - 1
        batch_indices = mx.arange(inputs.shape[0]).astype(mx.int32)
        text_embeds = hidden_states[batch_indices, positions]
        text_embeds = normalize_embeddings(text_embeds)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            text_embeds=text_embeds,
            pooler_output=None,
        )

    model_class.__init__ = patched_init
    model_class.__call__ = patched_call
    model_class._sommar_harrier_patch = True


class MlxHarrierEncoder:
    """Embed text with a quantized Harrier checkpoint on Apple Silicon."""

    def __init__(
        self,
        model_name: str,
        prompt_name: str | None,
        *,
        cache_limit_mb: int = 512,
    ) -> None:
        _patch_gemma3_text_adapter()
        import mlx.core as mx
        from mlx_embeddings import load

        self.model_name = model_name
        self.prompt_name = prompt_name
        mx.set_cache_limit(cache_limit_mb * 1024 * 1024)
        self.model, self.tokenizer = load(model_name)
        self.dimension = int(self.model.config.hidden_size)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
        progress_desc: str | None = None,
    ) -> np.ndarray:
        import mlx.core as mx
        from mlx_embeddings import generate

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        prompt = PROMPTS.get(self.prompt_name or "", "")
        indexed_texts = sorted(enumerate(texts), key=lambda item: len(item[1]))
        vectors: list[np.ndarray | None] = [None] * len(texts)
        from tqdm.auto import tqdm

        starts = range(0, len(indexed_texts), batch_size)
        for start in tqdm(
            starts,
            total=(len(indexed_texts) + batch_size - 1) // batch_size,
            desc=progress_desc or "Embedding",
            unit="batch",
            dynamic_ncols=True,
        ):
            indexed_batch = indexed_texts[start:start + batch_size]
            batch = [prompt + text for _, text in indexed_batch]
            output = generate(
                self.model,
                self.tokenizer,
                texts=batch,
                max_length=max_length,
                padding=True,
                truncation=True,
            )
            result = output.text_embeds
            mx.eval(result)
            for (index, _), vector in zip(indexed_batch, np.asarray(result, dtype=np.float32), strict=True):
                vectors[index] = vector
            # MLX otherwise retains peak temporary allocations for later,
            # potentially much smaller batches.
            mx.clear_cache()
        return np.stack([vector for vector in vectors if vector is not None])
