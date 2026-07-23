"""Embedding helper — talks to OpenAI text-embedding-3-small.

Kept tiny + isolated so swapping to a different embedder later (local
sentence-transformers, Cohere, etc.) is one drop-in replacement. The
schema dim is 1536 — change here AND in migrations/002_*.sql together
if you swap.

Embedder is OPTIONAL — if no API key is present, the ingester stores
chunks WITHOUT embeddings; topic-tag filtering still works as a fallback
retrieval mode (no cosine similarity until embeddings populate).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Must match migrations/002_knowledge_archive.sql — vector(1536).
_EMBEDDING_DIM: int = 1536
_DEFAULT_MODEL: str = "text-embedding-3-small"


class Embedder:
    """OpenAI text-embedding-3-small wrapper. Construct via from_env()
    so the missing-key fallback (returns None) is uniform across callers.
    """

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        if not api_key:
            msg = "Embedder requires non-empty OpenAI API key"
            raise ValueError(msg)
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=api_key)
        self.model = model

    @classmethod
    def from_env(cls, env_var: str = "OPENAI_API_KEY") -> Embedder | None:
        """Return an Embedder if the env var is set, else None.

        Lets the ingester run end-to-end without an OpenAI key — chunks
        get persisted with embedding=NULL and the retriever falls back
        to topic-tag filtering. The operator can re-run the ingester
        later to backfill embeddings.
        """
        key = os.environ.get(env_var, "")
        if not key:
            logger.info("%s not set — chunks will be stored without embeddings", env_var)
            return None
        return cls(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. OpenAI accepts up to ~2048 inputs per
        call; we don't batch beyond what's passed in."""
        if not texts:
            return []
        resp: Any = self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in resp.data]


def embedding_dim() -> int:
    """The vector dimension the schema expects. Imported by the ingester
    and retriever to avoid magic numbers scattered around."""
    return _EMBEDDING_DIM
