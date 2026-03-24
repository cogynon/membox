"""Consolidation: extract structured facts from episode batches.

Rule-based extraction for the core (no LLM dependency).
Provides abstract interface to plug in LLM-based extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from agentmemory.config import MemoryConfig
from agentmemory.episodic import EpisodicStore
from agentmemory.models import Fact
from agentmemory.semantic import SemanticStore


class Consolidator(ABC):
    """Abstract interface for episode → fact extraction.

    Implement extract() to support different extraction strategies:
    - RuleBasedConsolidator: keyword patterns (no deps, ships with core)
    - LLM-based: subclass and call your preferred LLM API
    """

    @abstractmethod
    def extract(self, contents: list[str]) -> list[dict]:
        """Extract facts from episode content strings.

        Returns list of dicts with keys: subject, predicate, object, confidence.
        """
        ...


class RuleBasedConsolidator(Consolidator):
    """Pattern-matching fact extractor. No external dependencies.

    Catches common patterns like "I prefer X", "I live in Y", etc.
    Good enough for demos; swap with LLM-based for production accuracy.
    """

    _PATTERNS = [
        # (trigger_phrases, predicate)
        (["i prefer ", "i like ", "i love ", "i enjoy "], "prefers"),
        (["i live in ", "i moved to ", "i'm based in ", "i am based in "], "lives_in"),
        (["i work at ", "i work for ", "i joined "], "works_at"),
        (["my name is ", "i'm ", "i am "], "name"),
        (["my favorite ", "my favourite "], "favorite"),
    ]

    def extract(self, contents: list[str]) -> list[dict]:
        results = []
        for content in contents:
            cl = content.lower()
            for triggers, predicate in self._PATTERNS:
                for trigger in triggers:
                    if trigger in cl:
                        # Extract the object (value after the trigger)
                        idx = cl.index(trigger) + len(trigger)
                        obj = content[idx:]
                        # Clean: stop at sentence boundaries
                        for sep in [".", ",", "!", "?", " and ", " but "]:
                            obj = obj.split(sep)[0]
                        obj = obj.strip().strip("'\"")
                        if obj and 2 < len(obj) < 50:
                            results.append({
                                "subject": "user",
                                "predicate": predicate,
                                "object": obj,
                                "confidence": 0.7,
                            })
                        break  # Only first trigger match per pattern group
        return results


def consolidate(episodic: EpisodicStore, semantic: SemanticStore,
                consolidator: Consolidator | None = None,
                config: MemoryConfig | None = None,
                now: datetime | None = None) -> dict:
    """Run consolidation: extract facts from unconsolidated episodes.

    1. Get batch of unconsolidated episodes older than min_age
    2. Extract facts using the consolidator
    3. Learn each fact in the semantic store
    4. Mark episodes as consolidated

    Returns summary: {"episodes_processed": N, "facts_extracted": N, "facts": [...]}
    """
    config = config or MemoryConfig()
    now = now or datetime.now()
    consolidator = consolidator or RuleBasedConsolidator()

    # Get unconsolidated episodes older than min_age
    min_age = timedelta(hours=config.consolidation_min_age_hours)
    candidates = episodic.unconsolidated(limit=config.consolidation_batch_size)
    eligible = [ep for ep in candidates if (now - ep.timestamp) >= min_age]

    if not eligible:
        return {"episodes_processed": 0, "facts_extracted": 0, "facts": []}

    # Extract facts
    contents = [ep.content for ep in eligible]
    raw_facts = consolidator.extract(contents)

    # Learn each fact
    learned = []
    for fact_data in raw_facts:
        fact, action = semantic.learn(
            subject=fact_data["subject"],
            predicate=fact_data["predicate"],
            obj=fact_data["object"],
            confidence=fact_data["confidence"],
            source_episode_id=eligible[0].id if eligible else None,
        )
        learned.append({"fact": fact, "action": action})

    # Mark as consolidated
    episodic.mark_consolidated([ep.id for ep in eligible])

    return {
        "episodes_processed": len(eligible),
        "facts_extracted": len(learned),
        "facts": learned,
    }
