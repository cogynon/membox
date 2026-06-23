"""Retrieval engine: score, rank, and return memories.

Combines recency decay, keyword relevance, and stored importance
into a single score per episode. No embedding dependency — uses
keyword overlap for the core. Embeddings can be layered on top.
"""

from __future__ import annotations

import math
from datetime import datetime

from membox.config import MemoryConfig
from membox.episodic import EpisodicStore
from membox.models import Episode, RetrievalResult
from membox.tokens import tokenize as _tokenize

if False:
    from membox.embedding_store import EmbeddingStore


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

    Returns query coverage: the fraction of the query's distinct word tokens
    that also appear in the content (asymmetric, not symmetric Jaccard). A
    perfect single-word query against a long document scores 1.0.
    For production semantic search, layer embeddings on top.
    """
    if not query or not content:
        return 0.0
    q_tokens = _tokenize(query)
    c_tokens = _tokenize(content)
    if not q_tokens:
        return 0.0
    overlap = q_tokens & c_tokens
    # Weighted by query coverage (what fraction of query words matched)
    return len(overlap) / len(q_tokens)


def score_episode(episode: Episode, query: str, now: datetime,
                  config: MemoryConfig) -> RetrievalResult:
    """Score a single episode against a query. Returns a RetrievalResult
    with component breakdown (recency, relevance, importance).

    The final score is normalized by the sum of weights so that custom
    weight configurations always produce results in the [0, 1] range.
    """
    r = recency_score(episode, now, config.decay_rate)
    v = relevance_score(query, episode.content)
    i = episode.importance

    total_weight = config.w_recency + config.w_relevance + config.w_importance
    combined = (config.w_recency * r + config.w_relevance * v + config.w_importance * i) / total_weight

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
           candidate_pool: int = 100,
           embedding_store: "EmbeddingStore | None" = None,
           min_score: float | None = None,
           min_relevance: float | None = None) -> list[RetrievalResult]:
    """Retrieve the top-k most relevant episodes for a query.

    Algorithm:
    1. Pull recent episodes as candidates (fast, index-backed)
    2. Score each candidate: w_recency*R + w_relevance*V + w_importance*I
    3. If embedding_store is provided, blend in semantic similarity
    4. Sort by combined score, return top-k (optionally filtered by min_score)

    Args:
        store: EpisodicStore to search.
        query: User query string.
        k: Number of results to return.
        config: Scoring weights and decay rate.
        now: Reference time for recency. Defaults to datetime.now().
        candidate_pool: How many recent episodes to consider.
        embedding_store: Optional EmbeddingStore for semantic retrieval.
        min_score: Optional minimum combined score (0-1) for a result to be
            returned. Use this to avoid feeding weakly-related noise to the LLM.
        min_relevance: Optional minimum *relevance* component (0-1). Unlike
            ``min_score`` (which blends recency/importance and can let a recent
            but irrelevant episode through), this is a principled "actually
            matched the query" floor.
    """
    config = config or MemoryConfig()
    now = now or datetime.now()

    # Get candidate pool (recent episodes, index-backed = fast)
    candidates = store.recent(candidate_pool)

    # Also add keyword-matched episodes that might not be recent.
    # Search EVERY query token (not just the first word) so that a relevant
    # old episode is reachable regardless of where its matching word sits in
    # the query. Single-character tokens are skipped to avoid matching noise.
    seen_ids = {ep.id for ep in candidates}
    query_tokens = [t for t in _tokenize(query) if len(t) >= 2]
    for token in query_tokens:
        for ep in store.search(token, limit=candidate_pool):
            if ep.id not in seen_ids:
                candidates.append(ep)
                seen_ids.add(ep.id)

    # Optional: fetch semantic matches from embedding store and merge in
    semantic_scores: dict[str, float] = {}
    if embedding_store is not None:
        try:
            semantic_matches = embedding_store.similarity(query, k=candidate_pool)
            for ep_id, sim in semantic_matches:
                semantic_scores[ep_id] = sim
                if ep_id not in seen_ids:
                    ep = store.get(ep_id)
                    if ep is not None:
                        candidates.append(ep)
                        seen_ids.add(ep_id)
        except Exception:
            pass  # Embedding retrieval failed; fall back to keyword-only

    # Score all candidates
    scored = []
    for ep in candidates:
        base = score_episode(ep, query, now, config)
        # Blend keyword relevance with semantic similarity if available
        if embedding_store is not None and ep.id in semantic_scores:
            sem_sim = semantic_scores[ep.id]
            # Replace keyword relevance with weighted blend
            blended_rel = (
                config.w_keyword * base.relevance +
                config.w_embedding * sem_sim
            ) / (config.w_keyword + config.w_embedding)
            # Recompute combined score with blended relevance
            combined = (
                config.w_recency * base.recency +
                config.w_relevance * blended_rel +
                config.w_importance * base.importance
            )
            scored.append(RetrievalResult(
                episode=ep,
                score=combined,
                recency=base.recency,
                relevance=blended_rel,
                importance=base.importance,
            ))
        else:
            scored.append(base)

    # Sort by combined score (descending)
    scored.sort(key=lambda r: r.score, reverse=True)

    # Filter by minimum relevance if requested; a principled "matched the query"
    # floor independent of recency/importance.
    if min_relevance is not None and min_relevance > 0.0:
        scored = [r for r in scored if r.relevance >= min_relevance]

    # Filter by minimum score if requested; catches noisy near-zero matches.
    if min_score is not None and min_score > 0.0:
        scored = [r for r in scored if r.score >= min_score]

    # Bump access count for returned results
    top_k = scored[:k]
    for result in top_k:
        store.increment_access(result.episode.id)

    return top_k
