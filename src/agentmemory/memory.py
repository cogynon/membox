"""AgentMemory — the main facade. One class, one import, full memory system.

This is what users interact with. It wires together all internal modules
(episodic, semantic, retrieval, forgetting, consolidation) behind a clean API.
"""

from __future__ import annotations

import math
from datetime import datetime

from agentmemory.config import MemoryConfig
from agentmemory.consolidation import Consolidator, RuleBasedConsolidator, consolidate
from agentmemory.episodic import EpisodicStore
from agentmemory.forgetting import forget
from agentmemory.models import Episode, Fact, RetrievalResult
from agentmemory.retrieval import recall
from agentmemory.semantic import SemanticStore


def _estimate_tokens(text: str) -> int:
    """~4 chars per token for English. Good enough for budgeting."""
    return max(1, len(text) // 4)


class AgentMemory:
    """Production-grade memory for any AI agent.

    Plug into any LLM, agent framework, or rule-based system:

        memory = AgentMemory("my_agent.db")
        memory.record("User said they love hiking")
        results = memory.recall("hobbies", k=3)
        memory.learn("user", "prefers", "black coffee", confidence=0.9)
        context_str = memory.context("what does the user like?")

    Backed by SQLite (single file, zero config, WAL mode).
    All state persists across restarts.
    """

    def __init__(self, db_path: str = "agent_memory.db",
                 config: MemoryConfig | None = None,
                 consolidator: Consolidator | None = None) -> None:
        self._config = config or MemoryConfig()
        self._episodic = EpisodicStore(db_path)
        self._semantic = SemanticStore(db_path, config=self._config)
        self._consolidator = consolidator or RuleBasedConsolidator()
        self._db_path = db_path

    # ── Episodic: record and recall ─────────────────────────────────

    def record(self, content: str, importance: float = 0.5,
               emotion: str | None = None, source: str = "conversation",
               context: dict | None = None,
               timestamp: datetime | None = None) -> Episode:
        """Store a new episodic memory.

        Args:
            content: What happened (text).
            importance: 0.0 (trivial) to 1.0 (life-changing).
            emotion: Optional emotion tag.
            source: Origin label (e.g. "conversation", "email", "observation").
            context: Arbitrary metadata dict.
            timestamp: When it happened. Defaults to now.

        Returns:
            The stored Episode.
        """
        episode = Episode(
            content=content,
            importance=importance,
            emotion=emotion,
            source=source,
            context=context or {},
            timestamp=timestamp or datetime.now(),
        )
        self._episodic.record(episode)
        return episode

    def recall(self, query: str, k: int = 5,
               now: datetime | None = None) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant memories for a query.

        Returns RetrievalResult objects with component score breakdown.
        """
        return recall(self._episodic, query, k=k,
                      config=self._config, now=now)

    def recent(self, n: int = 10) -> list[Episode]:
        """Get the N most recent episodes."""
        return self._episodic.recent(n)

    def search(self, keyword: str, limit: int = 10) -> list[Episode]:
        """Keyword search across all episodes."""
        return self._episodic.search(keyword, limit=limit)

    # ── Semantic: learn and query facts ─────────────────────────────

    def learn(self, subject: str, predicate: str, obj: str,
              confidence: float = 0.5,
              source_episode_id: str | None = None) -> tuple[Fact, str]:
        """Learn a semantic fact (with automatic reinforce/contradict).

        Returns (Fact, action) where action is 'new' | 'reinforced' | 'contradicted'.
        """
        return self._semantic.learn(subject, predicate, obj,
                                    confidence=confidence,
                                    source_episode_id=source_episode_id)

    def about(self, subject: str) -> list[Fact]:
        """Get all active facts about a subject."""
        return self._semantic.about(subject)

    def find_fact(self, subject: str,
                  predicate: str | None = None) -> list[Fact]:
        """Find facts by subject (and optionally predicate)."""
        return self._semantic.find(subject, predicate)

    # ── Context builder ─────────────────────────────────────────────

    def context(self, query: str = "", max_tokens: int | None = None,
                now: datetime | None = None) -> str:
        """Build a formatted context string ready to inject into any prompt.

        Combines user facts + relevant memories into a single string
        that fits within the token budget. This is the main integration
        point — paste this into your system prompt.

        Returns:
            Formatted string like:
            ## User Profile
            - user prefers black coffee (90%)
            ...
            ## Relevant Memories
            - (2h ago) User ordered coffee
            ...
        """
        max_tokens = max_tokens or self._config.max_context_tokens
        now = now or datetime.now()

        sections = []
        tokens_used = 0

        # Section 1: User profile
        facts = self._semantic.about("user")
        if facts:
            fact_lines = []
            for f in facts[:self._config.max_facts_in_context]:
                line = f"- {f.subject} {f.predicate} {f.object} ({f.confidence:.0%})"
                fact_lines.append(line)
            profile_text = "## User Profile\n" + "\n".join(fact_lines)
            profile_tokens = _estimate_tokens(profile_text)
            if tokens_used + profile_tokens < max_tokens:
                sections.append(profile_text)
                tokens_used += profile_tokens

        # Section 2: Relevant memories
        results = recall(self._episodic, query,
                         k=self._config.max_episodes_in_context,
                         config=self._config, now=now)
        if results:
            mem_lines = []
            for r in results:
                age = now - r.episode.timestamp
                if age.days > 0:
                    age_str = f"{age.days}d ago"
                elif age.total_seconds() > 3600:
                    age_str = f"{int(age.total_seconds()/3600)}h ago"
                else:
                    age_str = f"{int(age.total_seconds()/60)}m ago"
                emo = f" [{r.episode.emotion}]" if r.episode.emotion else ""
                line = f"- ({age_str}{emo}) {r.episode.content}"
                line_tokens = _estimate_tokens(line)
                if tokens_used + line_tokens >= max_tokens:
                    break
                mem_lines.append(line)
                tokens_used += line_tokens
            if mem_lines:
                sections.append("## Relevant Memories\n" + "\n".join(mem_lines))

        return "\n\n".join(sections) if sections else ""

    # ── Maintenance ─────────────────────────────────────────────────

    def consolidate(self, now: datetime | None = None) -> dict:
        """Compress unconsolidated episodes into semantic facts."""
        return consolidate(
            self._episodic, self._semantic,
            consolidator=self._consolidator,
            config=self._config, now=now,
        )

    def forget(self, now: datetime | None = None) -> dict:
        """Run forgetting pass: delete/archive stale memories."""
        return forget(self._episodic, config=self._config, now=now)

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Health check: counts, averages, time range."""
        ep_stats = self._episodic.stats()
        sem_stats = self._semantic.stats()
        return {
            "episodes": ep_stats,
            "facts": sem_stats,
            "db_path": self._db_path,
            "config": {
                "decay_rate": self._config.decay_rate,
                "weights": f"R={self._config.w_recency} V={self._config.w_relevance} I={self._config.w_importance}",
            },
        }

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        self._episodic.close()
        self._semantic.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        ep_count = self._episodic.count()
        fact_count = self._semantic.count()
        return f"AgentMemory(episodes={ep_count}, facts={fact_count}, db='{self._db_path}')"
