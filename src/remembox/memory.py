"""Remembox — the main facade. One class, one import, full memory system.

This is what users interact with. It wires together all internal modules
(episodic, semantic, retrieval, forgetting, consolidation) behind a clean API.
"""

from __future__ import annotations

import math
from datetime import datetime

import sqlite3

from remembox.config import MemoryConfig
from remembox.connection import create_connection
from remembox.consolidation import Consolidator, RuleBasedConsolidator, consolidate
from remembox.episodic import EpisodicStore
from remembox.forgetting import forget
from remembox.importance import ImportanceScorer, RuleBasedImportanceScorer
from remembox.models import Episode, Fact, Procedure, RetrievalResult
from remembox.procedural import ProceduralStore
from remembox.reflection import ReflectionStore, RuleBasedReflectionExtractor, reflect
from remembox.retrieval import recall
from remembox.semantic import SemanticStore
from remembox.summarization import (
    Summarizer,
    ThreadSummaryResult,
    estimate_tokens,
    summarize_thread,
)

if False:
    from remembox.embedding_store import EmbeddingStore


# Single shared estimator (see summarization.estimate_tokens) so the context
# builder and summarization agree on token budgeting.
_estimate_tokens = estimate_tokens


class Remembox:
    """Production-grade memory for any AI agent.

    Plug into any LLM, agent framework, or rule-based system:

        memory = Remembox("my_agent.db")
        memory.record("User said they love hiking")
        results = memory.recall("hobbies", k=3)
        memory.learn("user", "prefers", "black coffee", confidence=0.9)
        context_str = memory.context("what does the user like?")

    Backed by SQLite (single file, zero config, WAL mode).
    All state persists across restarts.
    """

    def __init__(self, db_path: str = "remembox.db",
                 config: MemoryConfig | None = None,
                 consolidator: Consolidator | None = None,
                 owner_id: str = "default",
                 importance_scorer: ImportanceScorer | None = None,
                 summarizer: Summarizer | None = None) -> None:
        self._config = config or MemoryConfig()
        self._owner_id = owner_id
        self._summarizer = summarizer
        self._importance_scorer = (
            importance_scorer
            if importance_scorer is not None
            else (RuleBasedImportanceScorer() if self._config.auto_score_importance else None)
        )
        self._db_path = db_path

        # In-memory databases are isolated per connection, so all stores must
        # share a single connection to see each other's tables.
        shared_conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            shared_conn = create_connection(":memory:")

        self._episodic = EpisodicStore(db_path, owner_id=owner_id, connection=shared_conn)
        self._semantic = SemanticStore(db_path, config=self._config, owner_id=owner_id, connection=shared_conn)
        self._procedural = ProceduralStore(db_path, owner_id=owner_id, connection=shared_conn)
        self._reflection = ReflectionStore(db_path, owner_id=owner_id, connection=shared_conn)
        self._consolidator = consolidator or RuleBasedConsolidator()

        # Keep the shared connection alive as long as Remembox owns it.
        self._shared_memory_conn: sqlite3.Connection | None = shared_conn

        # Optional embedding store for semantic retrieval
        self._embedding_store: "EmbeddingStore | None" = None
        self._embedding_model: object | None = None
        if self._config.embedding_model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    self._config.embedding_model_name,
                    cache_folder=self._config.embedding_cache_dir,
                )
                from remembox.embedding_store import EmbeddingStore
                self._embedding_store = EmbeddingStore(
                    db_path=db_path,
                    owner_id=owner_id,
                    model=self._embedding_model,
                    model_name=self._config.embedding_model_name,
                    connection=shared_conn,
                )
            except ImportError:
                # Graceful degradation: log warning and fall back to keyword-only
                import warnings
                warnings.warn(
                    f"sentence-transformers not installed. "
                    f"Embedding model '{self._config.embedding_model_name}' disabled. "
                    f"Install with: pip install sentence-transformers",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._config.embedding_model_name = None

    # ── Episodic: record and recall ─────────────────────────────────

    def record(self, content: str, importance: float | None = None,
               emotion: str | None = None, source: str = "conversation",
               context: dict | None = None,
               timestamp: datetime | None = None,
               thread_id: str | None = None,
               parent_id: str | None = None,
               depth: int = 0) -> Episode:
        """Store a new episodic memory.

        Args:
            content: What happened (text).
            importance: 0.0 (trivial) to 1.0 (life-changing). If None and
                an importance scorer is configured, it will be inferred.
            emotion: Optional emotion tag. If None and a scorer is configured,
                it will be inferred.
            source: Origin label (e.g. "conversation", "email", "observation").
            context: Arbitrary metadata dict.
            timestamp: When it happened. Defaults to now.
            thread_id: Optional conversation thread ID.
            parent_id: Optional parent episode ID for hierarchical threads.
            depth: Hierarchical depth (0 = top-level thread node).

        Returns:
            The stored Episode.
        """
        # Auto-score importance/emotion if scorer is configured and not provided.
        if importance is None and self._importance_scorer is not None:
            score_result = self._importance_scorer.score(content)
            importance = score_result.importance
            if emotion is None:
                emotion = score_result.emotion

        importance = importance if importance is not None else 0.5

        episode = Episode(
            content=content,
            importance=importance,
            emotion=emotion,
            source=source,
            context=context or {},
            timestamp=timestamp or datetime.now(),
            thread_id=thread_id,
            parent_id=parent_id,
            depth=depth,
            owner_id=self._owner_id,
        )
        self._episodic.record(episode)
        if self._embedding_store is not None:
            self._embedding_store.add(episode)
        return episode

    def recall(self, query: str, k: int = 5,
               now: datetime | None = None,
               min_score: float | None = None) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant memories for a query.

        Args:
            query: User query string.
            k: Number of results to return.
            now: Optional reference time for recency scoring.
            min_score: Optional minimum combined score (0-1) for a result
                to be returned. Filters out weakly-related noise.

        Returns RetrievalResult objects with component score breakdown.
        """
        return recall(self._episodic, query, k=k,
                      config=self._config, now=now,
                      embedding_store=self._embedding_store,
                      min_score=min_score)

    def recent(self, n: int = 10) -> list[Episode]:
        """Get the N most recent episodes."""
        return self._episodic.recent(n)

    def thread(self, thread_id: str, limit: int = 1000) -> list[Episode]:
        """Get all episodes belonging to a thread."""
        return self._episodic.by_thread(thread_id, limit=limit)

    def thread_children(self, episode_id: str, limit: int = 1000) -> list[Episode]:
        """Get direct child episodes of an episode."""
        return self._episodic.by_parent(episode_id, limit=limit)

    def threads(self, limit: int = 100) -> list[str]:
        """List all distinct thread IDs for this owner."""
        return self._episodic.threads(limit=limit)

    def search(self, keyword: str, limit: int = 10) -> list[Episode]:
        """Keyword search across all episodes."""
        return self._episodic.search(keyword, limit=limit)

    # ── Semantic: learn and query facts ─────────────────────────────

    def learn(self, subject: str, predicate: str, obj: str,
              confidence: float = 0.5,
              source_episode_id: str | None = None,
              valid_from: datetime | None = None,
              valid_until: datetime | None = None,
              recurrence: str | None = None) -> tuple[Fact, str]:
        """Learn a semantic fact (with automatic reinforce/contradict).

        Returns (Fact, action) where action is 'new' | 'reinforced' | 'contradicted'.

        Temporal fields:
            valid_from / valid_until: window in which the fact is known true.
            recurrence: e.g. "weekdays", "weekly", "quarterly".
        """
        return self._semantic.learn(subject, predicate, obj,
                                    confidence=confidence,
                                    source_episode_id=source_episode_id,
                                    valid_from=valid_from,
                                    valid_until=valid_until,
                                    recurrence=recurrence)

    def about(self, subject: str, at_time: datetime | None = None) -> list[Fact]:
        """Get all active facts about a subject, optionally at a specific time."""
        return self._semantic.about(subject, at_time=at_time)

    def find_fact(self, subject: str,
                  predicate: str | None = None,
                  at_time: datetime | None = None) -> list[Fact]:
        """Find facts by subject (and optionally predicate/time)."""
        return self._semantic.find(subject, predicate, at_time=at_time)

    # ── Procedural: routines and skills ─────────────────────────────

    def learn_procedure(self, trigger: str, action: str,
                        confidence: float = 0.5,
                        metadata: dict | None = None) -> Procedure:
        """Store a procedural rule: when trigger, do action."""
        return self._procedural.record(
            trigger=trigger, action=action,
            confidence=confidence, metadata=metadata or {},
        )

    def match_procedures(self, text: str) -> list[Procedure]:
        """Return stored procedures whose trigger matches the text."""
        return self._procedural.match(text)

    def procedures(self) -> list[Procedure]:
        """Return all stored procedures for this owner."""
        return self._procedural.all()

    def delete_procedure(self, procedure_id: str) -> bool:
        """Delete a procedure by ID."""
        return self._procedural.delete(procedure_id)

    # ── Editing / correction ────────────────────────────────────────

    def update_episode(self, episode_id: str,
                       content: str | None = None,
                       importance: float | None = None,
                       emotion: str | None = None,
                       source: str | None = None,
                       context: dict | None = None,
                       timestamp: datetime | None = None,
                       merge_context: bool = True) -> Episode | None:
        """Edit an existing episodic memory in place.

        Args:
            episode_id: ID of the episode to update.
            content/importance/emotion/source/timestamp: Replace only if provided.
            context: Additional context dict. Merged with existing context unless
                ``merge_context=False``.
            merge_context: If False, replace the episode context entirely.

        Returns:
            Updated Episode, or None if the ID was not found.
        """
        existing = self._episodic.get(episode_id)
        if existing is None:
            return None

        new_context = existing.context
        if context is not None:
            new_context = {**existing.context, **context} if merge_context else context

        updated = Episode(
            id=existing.id,
            content=content if content is not None else existing.content,
            timestamp=timestamp if timestamp is not None else existing.timestamp,
            importance=importance if importance is not None else existing.importance,
            emotion=emotion if emotion is not None else existing.emotion,
            source=source if source is not None else existing.source,
            context=new_context,
            consolidated=existing.consolidated,
            access_count=existing.access_count,
            thread_id=existing.thread_id,
            parent_id=existing.parent_id,
            depth=existing.depth,
            owner_id=existing.owner_id,
        )
        self._episodic.update(updated)
        if self._embedding_store is not None and content is not None:
            self._embedding_store.add(updated)
        return updated

    def annotate_episode(self, episode_id: str,
                         correction: str | None = None,
                         accuracy: str | None = None,
                         notes: str | None = None,
                         extra: dict | None = None,
                         now: datetime | None = None) -> Episode | None:
        """Attach a correction, accuracy flag, or note to an episode.

        Annotations are timestamped and appended to the episode's context
        under ``__annotations__`` so the original record stays intact while
        providing an audit trail. Pass ``now`` for deterministic timestamps.
        """
        annotation = {"timestamp": (now or datetime.now()).isoformat()}
        if correction is not None:
            annotation["correction"] = correction
        if accuracy is not None:
            annotation["accuracy"] = accuracy
        if notes is not None:
            annotation["notes"] = notes
        if extra is not None:
            annotation.update(extra)
        return self._episodic.annotate(episode_id, annotation)

    def edit_fact(self, fact_id: str,
                  obj: str | None = None,
                  predicate: str | None = None,
                  confidence: float | None = None,
                  source_episode_ids: list[str] | None = None,
                  is_active: bool | None = None) -> Fact | None:
        """Edit a fact in place."""
        return self._semantic.edit_fact(fact_id, obj=obj, predicate=predicate,
                                        confidence=confidence,
                                        source_episode_ids=source_episode_ids,
                                        is_active=is_active)

    def correct_fact(self, fact_id: str,
                     new_object: str | None = None,
                     new_predicate: str | None = None,
                     new_confidence: float | None = None) -> tuple[Fact, str]:
        """Correct a fact, deactivating the old version and inserting a new one."""
        return self._semantic.correct_fact(fact_id, new_object=new_object,
                                           new_predicate=new_predicate,
                                           new_confidence=new_confidence)

    # ── Context builder ─────────────────────────────────────────────

    def context(self, query: str = "", max_tokens: int | None = None,
                now: datetime | None = None,
                min_score: float | None = None,
                profile_subject: str = "user") -> str:
        """Build a formatted context string ready to inject into any prompt.

        Combines user facts + relevant memories into a single string
        that fits within the token budget. This is the main integration
        point — paste this into your system prompt.

        Args:
            query: Query used to retrieve relevant memories/procedures.
            max_tokens: Token budget for the whole context block.
            now: Reference time; also filters facts to those currently valid.
            min_score: Optional relevance floor forwarded to recall() so weak
                noise does not populate the Memories section.
            profile_subject: Subject for the User Profile section.

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

        # Section 1: User profile (only currently-valid facts, not expired ones)
        facts = self._semantic.about(profile_subject, at_time=now)
        if not facts:
            # Fallback: try exact subject, otherwise any facts for this owner
            facts = self._semantic.all_active()[:self._config.max_facts_in_context]
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

        # Section 2: Active procedures
        procedures = self._procedural.match(query)[:3]
        if procedures:
            proc_lines = []
            for p in procedures:
                line = f"- When '{p.trigger}' → {p.action} ({p.confidence:.0%})"
                proc_lines.append(line)
            proc_text = "## Active Procedures\n" + "\n".join(proc_lines)
            proc_tokens = _estimate_tokens(proc_text)
            if tokens_used + proc_tokens < max_tokens:
                sections.append(proc_text)
                tokens_used += proc_tokens

        # Section 3: Relevant memories
        results = recall(self._episodic, query,
                         k=self._config.max_episodes_in_context,
                         config=self._config, now=now,
                         embedding_store=self._embedding_store,
                         min_score=min_score)
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

        # Section 4: Patterns (reflections) — surface higher-order patterns so
        # they are not write-only from the LLM's perspective.
        reflections = self._reflection.find(profile_subject)
        max_reflections = self._config.max_reflections_in_context
        if reflections and max_reflections > 0:
            ref_lines = []
            for ref in reflections[:max_reflections]:
                line = f"- {ref.subject} {ref.predicate} {ref.object} ({ref.confidence:.0%})"
                ref_lines.append(line)
            ref_text = "## Patterns\n" + "\n".join(ref_lines)
            ref_tokens = _estimate_tokens(ref_text)
            if tokens_used + ref_tokens < max_tokens:
                sections.append(ref_text)
                tokens_used += ref_tokens

        return "\n\n".join(sections) if sections else ""

    # ── Maintenance ─────────────────────────────────────────────────

    def consolidate(self, now: datetime | None = None) -> dict:
        """Compress one batch of unconsolidated episodes into semantic facts."""
        return consolidate(
            self._episodic, self._semantic,
            consolidator=self._consolidator,
            config=self._config, now=now,
        )

    def consolidate_all(self, now: datetime | None = None,
                        max_batches: int = 1000) -> dict:
        """Repeatedly consolidate until the backlog is drained (or capped).

        ``consolidate()`` only processes ``consolidation_batch_size`` episodes
        per call; on a busy agent the backlog can outpace a single pass. This
        loops until no eligible episodes remain (bounded by ``max_batches``).
        """
        now = now or datetime.now()
        total_processed = 0
        total_facts = 0
        facts: list = []
        batches = 0
        while batches < max_batches:
            result = self.consolidate(now=now)
            batches += 1
            total_processed += result["episodes_processed"]
            total_facts += result["facts_extracted"]
            facts.extend(result["facts"])
            # Stop when a pass makes no progress (nothing eligible or nothing
            # produced a fact).
            if result["episodes_processed"] == 0:
                break
        return {
            "episodes_processed": total_processed,
            "facts_extracted": total_facts,
            "facts": facts,
            "batches": batches,
        }

    def summarize_thread(self, thread_id: str,
                         keep_recent_tokens: int | None = None,
                         custom_instructions: str | None = None,
                         now: datetime | None = None) -> ThreadSummaryResult:
        """Compress the older part of a thread into one summary episode.

        pi-style compaction for conversation threads: episodes past the recent
        token window are folded into a single structured ``thread_summary``
        episode and marked consolidated, so the thread stays small without
        losing its narrative. Recent episodes are kept verbatim.

        Returns a ``ThreadSummaryResult``; ``.did_summarize`` is False when the
        thread is too short to need compression.
        """
        return summarize_thread(
            self._episodic, thread_id,
            summarizer=self._summarizer,
            config=self._config,
            owner_id=self._owner_id,
            keep_recent_tokens=keep_recent_tokens,
            custom_instructions=custom_instructions,
            now=now,
        )

    def maintain(self, now: datetime | None = None) -> dict:
        """Run all background maintenance in one pass.

        Mirrors pi's single auto-compaction trigger: one call advances every
        housekeeping job so callers don't have to orchestrate them. Honors the
        config flags ``auto_reflect`` and ``summary_trigger_tokens`` (the latter
        was previously unreachable). Steps run in dependency order:

        1. consolidate  - episodes → durable facts
        2. reflect      - patterns across episodes (only if ``auto_reflect``)
        3. summarize    - compress threads over ``summary_trigger_tokens``
        4. forget       - prune/archive stale episodes

        Returns a dict summarizing what each step did.
        """
        now = now or datetime.now()
        report: dict = {}

        report["consolidate"] = self.consolidate_all(now=now)

        if self._config.auto_reflect:
            report["reflect"] = self.reflect(now=now)

        summaries: list[ThreadSummaryResult] = []
        trigger = self._config.summary_trigger_tokens
        if trigger is not None:
            for thread_id in self._episodic.threads():
                episodes = self._episodic.by_thread(thread_id)
                thread_tokens = sum(estimate_tokens(e.content) for e in episodes)
                if thread_tokens > trigger:
                    result = self.summarize_thread(thread_id, now=now)
                    if result.did_summarize:
                        summaries.append(result)
        report["summarized_threads"] = summaries

        report["forget"] = self.forget(now=now)
        return report

    def reflect(self, episodes: list[Episode] | None = None,
                extractor: object | None = None,
                now: datetime | None = None) -> dict:
        """Synthesize higher-order patterns across episodes.

        If ``episodes`` is None, pulls the most recent ``config.reflection_batch_size``
        episodes older than ``config.reflection_min_age_hours``. Pass ``now`` for
        deterministic lookback windows.

        Returns a dict so callers can distinguish "no patterns" from "nothing
        was eligible":
            {"reflections": [...], "evaluated": N, "skipped_too_recent": M}
        """
        now = now or datetime.now()
        skipped = 0
        if episodes is None:
            from datetime import timedelta
            cutoff = now - timedelta(hours=self._config.reflection_min_age_hours)
            candidates = self._episodic.recent(self._config.reflection_batch_size)
            episodes = [ep for ep in candidates if ep.timestamp <= cutoff]
            skipped = len(candidates) - len(episodes)
        reflections = reflect(episodes, store=self._reflection,
                              extractor=extractor, now=now)
        return {
            "reflections": reflections,
            "evaluated": len(episodes),
            "skipped_too_recent": skipped,
        }

    def reflections(self, subject: str = "user",
                    predicate: str | None = None) -> list:
        """Return active reflections about a subject."""
        return self._reflection.find(subject, predicate=predicate)

    def forget(self, now: datetime | None = None) -> dict:
        """Run forgetting pass: delete/archive stale memories."""
        return forget(self._episodic, config=self._config, now=now,
                      embedding_store=self._embedding_store)

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Health check: counts, averages, time range."""
        ep_stats = self._episodic.stats()
        sem_stats = self._semantic.stats()
        emb_stats = self._embedding_store.stats() if self._embedding_store else None
        proc_stats = {"total": self._procedural.count()}
        return {
            "episodes": ep_stats,
            "facts": sem_stats,
            "procedures": proc_stats,
            "embeddings": emb_stats,
            "db_path": self._db_path,
            "owner_id": self._owner_id,
            "config": {
                "decay_rate": self._config.decay_rate,
                "weights": f"R={self._config.w_recency} V={self._config.w_relevance} I={self._config.w_importance}",
                "embedding_model": self._config.embedding_model_name,
            },
        }

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self) -> None:
        """Close database connections.

        When using a shared in-memory connection, Remembox owns the
        connection and closes it directly; the stores skip closing.
        """
        self._episodic.close()
        self._semantic.close()
        self._procedural.close()
        self._reflection.close()
        if self._embedding_store is not None:
            self._embedding_store.close()
        if self._shared_memory_conn is not None:
            self._shared_memory_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        ep_count = self._episodic.count()
        fact_count = self._semantic.count()
        return (f"Remembox(owner='{self._owner_id}', episodes={ep_count}, "
                f"facts={fact_count}, db='{self._db_path}')")
