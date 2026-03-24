"""Optional embedding-based retrieval using sentence-transformers.

This module upgrades keyword-based retrieval to true semantic search.
It's optional — the core works without it. Install the extra:

    pip install sentence-transformers

Usage:
    from agentmemory import AgentMemory
    from agentmemory.embeddings import EmbeddingIndex

    memory = AgentMemory("agent.db")
    index = EmbeddingIndex()

    # Index existing episodes
    for ep in memory.recent(100):
        index.add(ep)

    # Semantic recall (uses embeddings instead of keyword overlap)
    results = index.query("outdoor activities", k=5)
"""

from __future__ import annotations

import math
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from agentmemory.config import MemoryConfig
from agentmemory.models import Episode, RetrievalResult


def _import_sentence_transformers():
    """Lazy import to avoid hard dependency."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embedding retrieval. "
            "Install it with: pip install sentence-transformers"
        )


class EmbeddingIndex:
    """In-memory embedding index for semantic episode retrieval.

    Uses sentence-transformers for encoding and brute-force cosine
    similarity for search. Fast enough for 100K+ episodes.

    For larger scale (1M+), swap in FAISS or Annoy.

    Args:
        model_name: Sentence-transformers model to use.
            Default 'all-MiniLM-L6-v2' is a good balance of speed and quality.
        cache_dir: Optional directory to cache the model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 cache_dir: str | None = None) -> None:
        SentenceTransformer = _import_sentence_transformers()
        self._model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self._embeddings: list[list[float]] = []
        self._episodes: list[Episode] = []
        self._id_to_idx: dict[str, int] = {}

    def add(self, episode: Episode) -> None:
        """Add a single episode to the index."""
        if episode.id in self._id_to_idx:
            return  # Already indexed
        embedding = self._model.encode(episode.content).tolist()
        self._id_to_idx[episode.id] = len(self._episodes)
        self._episodes.append(episode)
        self._embeddings.append(embedding)

    def add_batch(self, episodes: list[Episode]) -> int:
        """Batch-add episodes. Returns number of new episodes added."""
        new_eps = [ep for ep in episodes if ep.id not in self._id_to_idx]
        if not new_eps:
            return 0
        contents = [ep.content for ep in new_eps]
        embeddings = self._model.encode(contents).tolist()
        for ep, emb in zip(new_eps, embeddings):
            self._id_to_idx[ep.id] = len(self._episodes)
            self._episodes.append(ep)
            self._embeddings.append(emb)
        return len(new_eps)

    def query(self, text: str, k: int = 5,
              config: MemoryConfig | None = None,
              now: datetime | None = None) -> list[RetrievalResult]:
        """Semantic search: find episodes most similar to the query.

        Combines embedding similarity with recency and importance:
          score = w_relevance * cosine_sim + w_recency * recency + w_importance * importance
        """
        if not self._episodes:
            return []

        config = config or MemoryConfig()
        now = now or datetime.now()

        query_emb = self._model.encode(text).tolist()

        # Score all episodes
        results = []
        for i, (ep, emb) in enumerate(zip(self._episodes, self._embeddings)):
            # Cosine similarity
            cos_sim = self._cosine_similarity(query_emb, emb)
            cos_sim = max(0.0, cos_sim)  # Clamp negative

            # Recency
            hours = max(0, (now - ep.timestamp).total_seconds() / 3600.0)
            recency = math.exp(-config.decay_rate * hours)

            # Combined score
            score = (config.w_relevance * cos_sim +
                     config.w_recency * recency +
                     config.w_importance * ep.importance)

            results.append(RetrievalResult(
                episode=ep,
                score=score,
                recency=recency,
                relevance=cos_sim,
                importance=ep.importance,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def __len__(self) -> int:
        return len(self._episodes)

    def __contains__(self, episode_id: str) -> bool:
        return episode_id in self._id_to_idx
