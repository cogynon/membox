"""Regression tests for issues tracked in BUGS.md."""

from datetime import datetime, timedelta

import pytest

from remembox import Remembox
from remembox.config import MemoryConfig
from remembox.consolidation import RuleBasedConsolidator
from remembox.models import Episode, Procedure
from remembox.reflection import RuleBasedReflectionExtractor

NOW = datetime(2026, 1, 1, 12, 0, 0)


@pytest.fixture
def memory():
    m = Remembox(":memory:")
    yield m
    m.close()


class TestBug1UpdateEpisodeThread:
    def test_update_preserves_thread_fields(self, memory):
        ep = memory.record("hello", thread_id="t1", parent_id="p1",
                            depth=2, timestamp=NOW)
        memory.update_episode(ep.id, content="hello edited")
        got = memory._episodic.get(ep.id)
        assert got.thread_id == "t1"
        assert got.parent_id == "p1"
        assert got.depth == 2
        assert len(memory.thread("t1")) == 1


class TestBug2And3Reflection:
    def test_repeats_in_one_episode_do_not_pass_threshold(self):
        ex = RuleBasedReflectionExtractor(min_mentions=3)
        out = ex.extract([
            Episode(content="coffee coffee coffee", timestamp=NOW),
            Episode(content="more coffee", timestamp=NOW),
        ], now=NOW)
        assert not any(r.object == "coffee" for r in out)

    def test_punctuation_tokenized_consistently(self):
        ex = RuleBasedReflectionExtractor(min_mentions=2)
        out = ex.extract([
            Episode(content="I love coffee.", timestamp=NOW),
            Episode(content="coffee is great", timestamp=NOW),
        ], now=NOW)
        assert any(r.object == "coffee" for r in out)


class TestBug4ExpiredFacts:
    def test_expired_fact_excluded_from_context(self, memory):
        memory.learn("user", "lives_in", "Mumbai", valid_until=datetime(2023, 1, 1))
        memory.learn("user", "lives_in", "Berlin")
        ctx = memory.context("where", now=NOW)
        assert "Mumbai" not in ctx
        assert "Berlin" in ctx


class TestBug5And6Consolidation:
    def test_zero_fact_episode_stays_unconsolidated(self, memory):
        memory.record("The weather is sunny today.",
                      timestamp=NOW - timedelta(hours=2))
        memory.consolidate(now=NOW)
        assert len(memory._episodic.unconsolidated()) == 1

    def test_no_duplicate_facts_from_overlapping_triggers(self):
        facts = RuleBasedConsolidator().extract_facts(
            [Episode(content="I'm based in San Francisco.")]
        )
        assert len(facts) == 1
        assert facts[0]["predicate"] == "lives_in"
        assert facts[0]["object"] == "San Francisco"

    def test_trailing_adverb_stripped(self):
        facts = RuleBasedConsolidator().extract_facts(
            [Episode(content="I live in Berlin now.")]
        )
        assert facts[0]["object"] == "Berlin"


class TestBug7And8Forgetting:
    def test_archive_does_not_set_consolidated(self, memory):
        ep = memory.record("low importance chatter", importance=0.4,
                           timestamp=NOW - timedelta(days=20))
        memory.forget(now=NOW)
        got = memory._episodic.get(ep.id)
        assert got.archived is True
        assert got.consolidated is False

    def test_thread_summary_not_deleted(self):
        cfg = MemoryConfig(summary_trigger_tokens=50, summary_keep_recent_tokens=20)
        m = Remembox(":memory:", config=cfg)
        old = NOW - timedelta(days=30)
        for i in range(10):
            m.record("chatter " + "x" * 40, importance=0.2,
                     timestamp=old, thread_id="t1")
        m.maintain(now=NOW)
        summaries = [e for e in m.thread("t1") if e.source == "thread_summary"]
        assert len(summaries) >= 1
        m.close()


class TestBug9Reflect:
    def test_reflect_returns_dict_with_skip_info(self, memory):
        for _ in range(3):
            memory.record("recent", emotion="happy", timestamp=NOW)
        result = memory.reflect(now=NOW)
        assert isinstance(result, dict)
        assert "reflections" in result
        assert "evaluated" in result
        assert "skipped_too_recent" in result


class TestBug10ProcedureCreatedAt:
    def test_created_at_stable_across_serialization(self):
        p = Procedure(trigger="a", action="b")
        assert p.to_dict()["created_at"] == p.to_dict()["created_at"]

    def test_created_at_round_trips(self, memory):
        proc = memory.learn_procedure("trigger", "action")
        reloaded = memory.procedures()[0]
        assert reloaded.created_at == proc.created_at


class TestBug11ReflectionsInContext:
    def test_patterns_section_present(self, memory):
        memory.reflect(episodes=[Episode(content="x", emotion="happy", timestamp=NOW)
                                 for _ in range(3)], now=NOW)
        ctx = memory.context("anything", now=NOW)
        assert "## Patterns" in ctx


class TestBug13WildcardInjection:
    def test_percent_does_not_match_all(self, memory):
        memory.record("hello world")
        assert memory.search("%") == []

    def test_underscore_literal(self, memory):
        memory.record("hello world")
        memory.record("a_b literal")
        results = memory.search("_")
        assert all("_" in r.content for r in results)


class TestBug16IterAll:
    def test_iter_all_returns_everything(self, memory):
        for i in range(1200):
            memory.record(f"msg {i}", timestamp=NOW + timedelta(seconds=i))
        seen = list(memory._episodic.iter_all(batch_size=100))
        assert len(seen) == 1200
        assert len({e.id for e in seen}) == 1200


class TestBug26ConsolidateAll:
    def test_drains_backlog(self):
        cfg = MemoryConfig(consolidation_batch_size=5)
        m = Remembox(":memory:", config=cfg)
        for i in range(23):
            m.record("I prefer item " + str(i),
                     timestamp=NOW - timedelta(hours=2))
        result = m.consolidate_all(now=NOW)
        assert result["episodes_processed"] == 23
        assert len(m._episodic.unconsolidated()) == 0
        m.close()
