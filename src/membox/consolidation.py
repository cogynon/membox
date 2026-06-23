"""Consolidation: extract structured facts from episode batches.

Rule-based extraction for the core (no LLM dependency).
Provides abstract interface to plug in LLM-based extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from membox._store import EpisodicStoreProtocol, SemanticStoreProtocol
from membox.config import MemoryConfig
from membox.models import Episode, Fact


class Consolidator(ABC):
    """Abstract interface for episode → fact extraction.

    Implement extract_facts() to support different extraction strategies:
    - RuleBasedConsolidator: keyword patterns (no deps, ships with core)
    - LLM-based: subclass and call your preferred LLM API
    """

    @abstractmethod
    def extract_facts(self, episodes: list[Episode]) -> list[dict]:
        """Extract facts from a list of Episode objects.

        Returns list of dicts with keys: subject, predicate, object, confidence.

        Receives full Episode objects (not just strings) so consolidators
        can use timestamp, importance, emotion, and context when deciding
        what facts to extract.
        """
        ...

    def extract(self, contents: list[str]) -> list[dict]:
        """Deprecated compatibility shim.

        Kept for backward compatibility; forwards to extract_facts().
        New code should override extract_facts().
        """
        episodes = [
            Episode(content=c, timestamp=datetime.now(), importance=0.5)
            for c in contents
        ]
        return self.extract_facts(episodes)


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

    def extract_facts(self, episodes: list[Episode]) -> list[dict]:
        """Extract facts from episodes, preserving source episode IDs.

        Each episode yields at most one fact: patterns are tried in order and
        the *earliest-matching* (lowest index in the content) trigger wins, so
        overlapping triggers like "i'm" vs "i'm based in" don't both fire.
        """
        results = []
        for ep in episodes:
            cl = ep.content.lower()
            best: tuple[int, int, str] | None = None  # (position, trigger_len, predicate)
            for triggers, predicate in self._PATTERNS:
                for trigger in triggers:
                    pos = cl.find(trigger)
                    if pos == -1:
                        continue
                    # Prefer the trigger that appears earliest; on a tie, prefer
                    # the longer (more specific) trigger.
                    if best is None or pos < best[0] or (
                        pos == best[0] and len(trigger) > best[1]
                    ):
                        best = (pos, len(trigger), predicate)
            if best is None:
                continue
            pos, trigger_len, predicate = best
            obj = ep.content[pos + trigger_len:]
            # Clean: stop at sentence boundaries
            for sep in [".", ",", "!", "?", " and ", " but "]:
                obj = obj.split(sep)[0]
            obj = obj.strip().strip("'\"")
            # Strip trailing time adverbs that get greedily captured
            # ("Berlin now" -> "Berlin"). Known limitation: this is a coarse
            # heuristic; LLM-based consolidation is the production path.
            for adverb in (" now", " today", " currently", " these days", " lately"):
                if obj.lower().endswith(adverb):
                    obj = obj[: -len(adverb)].rstrip()
            if obj and 2 < len(obj) < 50:
                results.append({
                    "subject": "user",
                    "predicate": predicate,
                    "object": obj,
                    "confidence": 0.7,
                    "source_episode_id": ep.id,
                })
        return results


def consolidate(episodic: EpisodicStoreProtocol, semantic: SemanticStoreProtocol,
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

    # Extract facts, passing full Episode objects to the consolidator.
    raw_facts = consolidator.extract_facts(eligible)

    # Learn each fact, linking it to the specific episode it came from.
    learned = []
    productive_ids: set[str] = set()
    for fact_data in raw_facts:
        source_id = fact_data.get("source_episode_id")
        fact, action = semantic.learn(
            subject=fact_data["subject"],
            predicate=fact_data["predicate"],
            obj=fact_data["object"],
            confidence=fact_data["confidence"],
            source_episode_id=source_id,
        )
        learned.append({"fact": fact, "action": action})
        if source_id:
            productive_ids.add(source_id)

    # Only mark episodes that actually produced ≥1 fact as consolidated.
    # Episodes that extracted nothing stay unconsolidated so a better
    # extractor can reprocess them later.
    to_mark = [ep.id for ep in eligible if ep.id in productive_ids]
    episodic.mark_consolidated(to_mark)

    return {
        "episodes_processed": len(to_mark),
        "facts_extracted": len(learned),
        "facts": learned,
    }
