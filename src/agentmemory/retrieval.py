"""Retrieval engine: score, rank, and return memories.

Combines recency decay, keyword relevance, and stored importance
into a single score per episode. No embedding dependency — uses
keyword overlap for the core. Embeddings can be layered on top.
"""

from __future__ import annotations

import math
from datetime import datetime

from agentmemory.config import MemoryConfig
from agentmemory.episodic import EpisodicStore
from agentmemory.models import Episode, RetrievalResult


def recency_score(episode: Episode, now: datetime,
                  decay_rate: float = 0.02) -> float:
    """Exponential decay based on time elapsed.

    recency = e^(-decay_rate × hours_ago)

    At default rate (0.02):
      1 hour ago  → 0.98
      24 hours    → 0.62
      7 days      → 0.04
    """
    hours = max(0, (now - episode.timestamp).total_seconds() / 3600.0)
    return math.exp(-decay_rate * hours)


def relevance_score(query: str, content: str) -> float:
    """Keyword overlap relevance. Fast, no dependencies.

    Computes Jaccard-like overlap between query tokens and content tokens.
    For production semantic search, layer embeddings on top.
    """
    if not query or not content:
        return 0.0
    q_tokens = set(query.lower().split())
    c_tokens = set(content.lower().split())
    if not q_tokens:
        return 0.0
    overlap = q_tokens & c_tokens
    # Weighted by query coverage (what fraction of query words matched)
    return len(overlap) / len(q_tokens)


def score_episode(episode: Episode, query: str, now: datetime,
                  config: MemoryConfig) -> RetrievalResult:
    """Score a single episode against a query. Returns a RetrievalResult
    with component breakdown (recency, relevance, importance)."""
    r = recency_score(episode, now, config.decay_rate)
    v = relevance_score(query, episode.content)
    i = episode.importance

    combined = config.w_recency * r + config.w_relevance * v + config.w_importance * i

    return RetrievalResult(
        episode=episode,
        score=combined,
        recency=r,
        relevance=v,
        importance=i,
    )


def recall(store: EpisodicStore, query: str,
           k: int = 5, config: MemoryConfig | None = None,
           now: datetime | None = None,
           candidate_pool: int = 100) -> list[RetrievalResult]:
    """Retrieve the top-k most relevant episodes for a query.

    Algorithm:
    1. Pull recent episodes as candidates (fast, index-backed)
    2. Score each candidate: w_recency*R + w_relevance*V + w_importance*I
    3. Sort by combined score, return top-k

    Args:
        store: EpisodicStore to search.
        query: User query string.
        k: Number of results to return.
        config: Scoring weights and decay rate.
        now: Reference time for recency. Defaults to datetime.now().
        candidate_pool: How many recent episodes to consider.
    """
    config = config or MemoryConfig()
    now = now or datetime.now()

    # Get candidate pool (recent episodes, index-backed = fast)
    candidates = store.recent(candidate_pool)

    # Also add keyword-matched episodes that might not be recent
    keyword_matches = store.search(query.split()[0] if query.split() else "", limit=candidate_pool)
    seen_ids = {ep.id for ep in candidates}
    for ep in keyword_matches:
        if ep.id not in seen_ids:
            candidates.append(ep)
            seen_ids.add(ep.id)

    # Score all candidates
    scored = [score_episode(ep, query, now, config) for ep in candidates]

    # Sort by combined score (descending)
    scored.sort(key=lambda r: r.score, reverse=True)

    # Bump access count for returned results
    for result in scored[:k]:
        store.increment_access(result.episode.id)

    return scored[:k]
