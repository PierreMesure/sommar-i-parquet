"""Incrementally cached embeddings from Sentence Transformers or MLX."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "majentik/harrier-oss-v1-270m-MLX-4bit"
# Harrier instructions are query-side only. Transcript units are documents in
# both semantic chunking and topic modelling, so they must stay unprompted.
DEFAULT_EMBEDDING_PROMPT: str | None = None
DEFAULT_EMBEDDING_BACKEND = "mlx"


def _model_slug(model_name: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name).strip("-")
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{readable}-{digest}"


class EmbeddingEncoder:
    """Load one embedding backend and reuse cached vectors by content ID."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        model_cache_dir: Path = Path("data/models/embeddings"),
        device: str | None = None,
        batch_size: int = 16,
        prompt_name: str | None = DEFAULT_EMBEDDING_PROMPT,
        backend: str = DEFAULT_EMBEDDING_BACKEND,
        max_length: int = 4096,
        mlx_cache_limit_mb: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.prompt_name = prompt_name
        self.backend = backend
        self.max_length = max_length
        if backend == "mlx":
            from src.nlp.mlx_harrier import MlxHarrierEncoder

            self.model = MlxHarrierEncoder(
                model_name,
                prompt_name,
                cache_limit_mb=mlx_cache_limit_mb,
            )
        elif backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                cache_folder=str(model_cache_dir),
                device=device,
                model_kwargs={"dtype": "auto"},
            )
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        progress_desc: str | None = None,
    ) -> np.ndarray:
        """Encode documents as normalized float32 vectors."""
        if not texts:
            dimension = (
                self.model.dimension
                if self.backend == "mlx"
                else self.model.get_embedding_dimension() or 0
            )
            return np.empty((0, dimension), dtype=np.float32)
        if self.backend == "mlx":
            return self.model.encode(
                texts,
                batch_size=batch_size or self.batch_size,
                max_length=self.max_length,
                progress_desc=progress_desc,
            )
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=batch_size or self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                prompt_name=self.prompt_name,
            ),
            dtype=np.float32,
        )

    def encode_cached(
        self,
        records: Sequence[dict[str, Any]],
        *,
        id_key: str,
        text_key: str,
        cache_root: Path,
        cache_name: str,
        force: bool = False,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Reuse matching rows and append embeddings for newly seen content IDs."""
        embedding_recipe = (
            f"{self.backend}::{self.model_name}::prompt={self.prompt_name or 'none'}"
            f"::max_length={self.max_length}"
        )
        directory = cache_root / _model_slug(embedding_recipe) / cache_name
        ids_path = directory / "ids.json"
        embeddings_path = directory / "embeddings.npy"
        directory.mkdir(parents=True, exist_ok=True)

        old_ids: list[str] = []
        old_embeddings: np.ndarray | None = None
        if not force and ids_path.exists() and embeddings_path.exists():
            old_ids = list(json.loads(ids_path.read_text(encoding="utf-8")))
            old_embeddings = np.load(embeddings_path)
            if len(old_ids) != len(old_embeddings):
                old_ids = []
                old_embeddings = None

        old_index = {record_id: index for index, record_id in enumerate(old_ids)}
        current_ids = [str(record[id_key]) for record in records]
        missing_positions = [
            position for position, record_id in enumerate(current_ids)
            if record_id not in old_index
        ]
        missing_embeddings = self.encode(
            [str(records[position][text_key]) for position in missing_positions],
            batch_size=batch_size,
            progress_desc=f"Embedding {cache_name}",
        )

        if old_embeddings is not None:
            dimension = int(old_embeddings.shape[1])
        elif len(missing_embeddings):
            dimension = int(missing_embeddings.shape[1])
        else:
            dimension = int(
                self.model.dimension
                if self.backend == "mlx"
                else self.model.get_embedding_dimension() or 0
            )
        result = np.empty((len(records), dimension), dtype=np.float32)
        missing_lookup = {
            position: missing_embeddings[index]
            for index, position in enumerate(missing_positions)
        }
        for position, record_id in enumerate(current_ids):
            if position in missing_lookup:
                result[position] = missing_lookup[position]
            elif old_embeddings is not None:
                result[position] = old_embeddings[old_index[record_id]]

        temporary_embeddings = embeddings_path.with_suffix(".tmp.npy")
        np.save(temporary_embeddings, result.astype(np.float16))
        temporary_embeddings.replace(embeddings_path)
        temporary_ids = ids_path.with_suffix(".tmp.json")
        temporary_ids.write_text(
            json.dumps(current_ids, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_ids.replace(ids_path)
        return result
