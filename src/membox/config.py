"""Configuration. Every tunable knob in one place, with sensible defaults."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MemoryConfig:
    """Central configuration for the entire memory system.

    Every parameter has a production-tested default. Override only
    what you need:

        config = MemoryConfig(decay_rate=0.05, max_episodes=500_000)
        memory = Membox("agent.db", config=config)

    Or use presets:

        config = MemoryConfig.fast()    # Low latency, aggressive forgetting
        config = MemoryConfig.deep()    # Long retention, slow decay
    """

    # ── Retrieval weights ───────────────────────────────────────────
    # Combined score = w_recency*R + w_relevance*V + w_importance*I
    w_recency: float = 0.3
    w_relevance: float = 0.4
    w_importance: float = 0.3

    # ── Decay rate ──────────────────────────────────────────────────
    # Controls exponential decay: recency = e^(-decay_rate * hours_ago)
    # 0.01 = slow (memories last weeks), 0.1 = fast (fade in days)
    decay_rate: float = 0.02

    # ── Forgetting thresholds ───────────────────────────────────────
    # Tiered by importance:
    #   (max_importance, max_age_days, action)
    #   action: "delete" | "archive" | "keep"
    forgetting_tiers: list[tuple[float, int, str]] = field(default_factory=lambda: [
        (0.3,   7,  "delete"),   # Low importance: delete after 7 days
        (0.5,  14,  "archive"),  # Low-med: archive after 14 days
        (0.7,  60,  "archive"),  # Medium: archive after 60 days
        (0.9, 180,  "archive"),  # High: archive after 180 days
        (1.0, 999,  "keep"),     # Critical: effectively never
    ])

    # ── Consolidation ───────────────────────────────────────────────
    consolidation_batch_size: int = 20
    consolidation_min_age_hours: float = 1.0

    # ── Reinforcement ───────────────────────────────────────────────
    # When a fact is seen again, boost confidence by this fraction
    # of the remaining gap: new_conf = old + (1 - old) * boost_rate
    reinforce_boost_rate: float = 0.15

    # ── Context builder ─────────────────────────────────────────────
    max_context_tokens: int = 2000
    max_facts_in_context: int = 20
    max_episodes_in_context: int = 15
    max_reflections_in_context: int = 5

    # ── Importance scoring ────────────────────────────────────────
    # When True, Membox.auto_score_importance = True and a configured
    # scorer will automatically rate importance on record().
    auto_score_importance: bool = False

    # ── Reflection ──────────────────────────────────────────────────
    # How many recent episodes to feed into the reflection extractor and
    # how often (in hours) reflection should run automatically.
    reflection_batch_size: int = 50
    reflection_min_age_hours: float = 24.0
    auto_reflect: bool = False

    # ── Thread summarization ────────────────────────────────────────
    # Compresses long conversation threads in place (pi-style compaction):
    # episodes older than the recent window are folded into one structured
    # summary episode so a thread can keep going without unbounded growth.
    # summary_keep_recent_tokens: recent tokens kept verbatim (not summarized).
    # summary_trigger_tokens: a thread above this size is eligible during
    #   maintain(); None disables auto-summarization.
    # max_serialized_chars: per-episode truncation when building summary input.
    summary_keep_recent_tokens: int = 2000
    summary_trigger_tokens: int | None = 6000
    max_serialized_chars: int = 2000

    # ── Embeddings ────────────────────────────────────────────────
    # Set embedding_model_name to enable semantic retrieval.
    # "all-MiniLM-L6-v2" is a good default (fast, small, high quality).
    # Set to None (default) to use keyword-only retrieval.
    embedding_model_name: str | None = None
    embedding_cache_dir: str | None = None
    # Hybrid retrieval: how much to weight embedding similarity
    # vs keyword overlap. Only used when embeddings are enabled.
    w_embedding: float = 0.6
    w_keyword: float = 0.4

    # ── Presets ──────────────────────────────────────────────────────

    @classmethod
    def fast(cls) -> MemoryConfig:
        """Aggressive forgetting, fast queries. Good for chatbots."""
        return cls(
            decay_rate=0.1,
            forgetting_tiers=[
                (0.3,   3, "delete"),
                (0.5,   7, "archive"),
                (0.7,  30, "archive"),
                (0.9,  90, "archive"),
                (1.0, 365, "keep"),
            ],
            max_context_tokens=1000,
            embedding_model_name=None,
            # Chatbots accumulate chatter fast: compress threads early.
            summary_keep_recent_tokens=1000,
            summary_trigger_tokens=3000,
        )

    @classmethod
    def deep(cls) -> MemoryConfig:
        """Long retention, slow forgetting. Good for personal assistants."""
        return cls(
            decay_rate=0.005,
            forgetting_tiers=[
                (0.3,  30, "archive"),
                (0.5,  90, "archive"),
                (0.7, 365, "archive"),
                (0.9, 999, "keep"),
                (1.0, 999, "keep"),
            ],
            max_context_tokens=4000,
            max_episodes_in_context=25,
            embedding_model_name="all-MiniLM-L6-v2",
            # Assistants want long verbatim recall: summarize later, keep more.
            summary_keep_recent_tokens=4000,
            summary_trigger_tokens=12000,
        )
