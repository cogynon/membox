"""Forgetting engine: retention scoring and tiered cleanup policies.

Prevents unbounded memory growth. Importance-weighted: trivial memories
die fast, critical ones survive indefinitely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from agentmemory.config import MemoryConfig
from agentmemory.episodic import EpisodicStore
from agentmemory.models import Episode


@dataclass(slots=True)
class ForgetAction:
    """What to do with a specific episode."""
    episode_id: str
    action: str          # "delete" | "archive" | "keep"
    retention_score: float
    reason: str


def retention_score(episode: Episode, now: datetime,
                    config: MemoryConfig) -> float:
    """How strongly this memory should be retained (0=forget, 1=keep).

    Inspired by ACT-R cognitive architecture:
      retention = 0.3 × recency + 0.4 × importance + 0.2 × access_bonus - 0.1 × consolidated_penalty
    """
    hours = max(0, (now - episode.timestamp).total_seconds() / 3600.0)
    recency = math.exp(-config.decay_rate * hours)

    # Access frequency bonus: each access adds a small boost (diminishing)
    access_bonus = min(1.0, episode.access_count * 0.1)

    # Consolidated episodes are less needed (their knowledge is in facts now)
    consolidated_penalty = 0.3 if episode.consolidated else 0.0

    score = (
        0.3 * recency +
        0.4 * episode.importance +
        0.2 * access_bonus -
        0.1 * consolidated_penalty
    )
    return max(0.0, min(1.0, score))


def evaluate_episode(episode: Episode, now: datetime,
                     config: MemoryConfig) -> ForgetAction:
    """Decide what to do with an episode based on tiered forgetting policy.

    Uses config.forgetting_tiers: list of (max_importance, max_age_days, action)
    sorted by importance threshold ascending.
    """
    age_days = (now - episode.timestamp).total_seconds() / 86400.0
    ret_score = retention_score(episode, now, config)

    for max_imp, max_age, action in config.forgetting_tiers:
        if episode.importance <= max_imp and age_days > max_age:
            return ForgetAction(
                episode_id=episode.id,
                action=action,
                retention_score=ret_score,
                reason=f"imp={episode.importance:.1f} ≤ {max_imp}, age={age_days:.0f}d > {max_age}d",
            )

    return ForgetAction(
        episode_id=episode.id,
        action="keep",
        retention_score=ret_score,
        reason=f"above all thresholds (imp={episode.importance:.1f}, age={age_days:.0f}d)",
    )


def forget(store: EpisodicStore, config: MemoryConfig | None = None,
           now: datetime | None = None) -> dict:
    """Run forgetting pass over all episodes. Returns summary.

    Actions:
    - "delete": permanently removed
    - "archive": marked as consolidated (soft-archive)
    - "keep": no action

    Returns dict with counts: {"deleted": N, "archived": N, "kept": N, "actions": [...]}
    """
    config = config or MemoryConfig()
    now = now or datetime.now()

    to_delete = []
    to_archive = []
    actions = []
    kept = 0

    for episode in store.iter_all():
        action = evaluate_episode(episode, now, config)
        actions.append(action)

        if action.action == "delete":
            to_delete.append(episode.id)
        elif action.action == "archive":
            to_archive.append(episode.id)
        else:
            kept += 1

    # Execute
    deleted = store.delete(to_delete) if to_delete else 0
    archived = store.mark_consolidated(to_archive) if to_archive else 0

    return {
        "deleted": deleted,
        "archived": archived,
        "kept": kept,
        "total_evaluated": len(actions),
        "actions": actions,
    }
