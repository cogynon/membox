"""Tests for Consolidator API alignment with lessons (Episode-based)."""

import pytest

from membox.models import Episode
from membox.consolidation import Consolidator, RuleBasedConsolidator


class MetadataAwareConsolidator(Consolidator):
    """Custom consolidator that needs access to episode metadata."""

    def extract_facts(self, episodes: list[Episode]) -> list[dict]:
        results = []
        for ep in episodes:
            if ep.importance >= 0.8:
                results.append({
                    "subject": "user",
                    "predicate": "high_importance_event",
                    "object": ep.content[:40],
                    "confidence": ep.importance,
                    "source_episode_id": ep.id,
                })
        return results


class TestConsolidatorAPI:

    def test_extract_facts_receives_episodes(self):
        consolidator = RuleBasedConsolidator()
        episodes = [Episode(content="I prefer black coffee", importance=0.6)]
        facts = consolidator.extract_facts(episodes)
        assert len(facts) == 1
        assert facts[0]["object"] == "black coffee"
        assert facts[0]["source_episode_id"] == episodes[0].id

    def test_legacy_extract_still_works(self):
        consolidator = RuleBasedConsolidator()
        facts = consolidator.extract(["I live in Mumbai"])
        assert len(facts) == 1
        assert facts[0]["predicate"] == "lives_in"

    def test_custom_consolidator_can_use_metadata(self):
        consolidator = MetadataAwareConsolidator()
        episodes = [
            Episode(content="I got promoted", importance=0.95),
            Episode(content="I ate lunch", importance=0.2),
        ]
        facts = consolidator.extract_facts(episodes)
        assert len(facts) == 1
        assert "promoted" in facts[0]["object"]
