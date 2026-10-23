"""Embedding model wrapper."""

from __future__ import annotations

import os
from typing import Protocol

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Lazily loads a sentence-transformers model on first use (first run downloads ~90 MB)."""

    def __init__(self, model_name: str | None = None, dim: int = 384):
        self.model_name = model_name or os.environ.get("LIBRARIAN_EMBED_MODEL", DEFAULT_MODEL)
        self.dim = dim
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            # get_embedding_dimension replaced get_sentence_embedding_dimension in newer releases.
            get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            actual = get_dim()
            if actual != self.dim:
                raise ValueError(f"{self.model_name} produces {actual}-dim embeddings, expected {self.dim}")
        return self._model

    @property
    def max_seq_length(self) -> int:
        return self._load().max_seq_length

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()
