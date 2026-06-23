"""Tests for pi-style thread summarization."""

from datetime import datetime, timedelta

import pytest

from remembox import Remembox, MemoryConfig, RuleBasedSummarizer, Summarizer
from remembox.models import Episode
from remembox.summarization import (
    _find_cut_index,
    estimate_tokens,
    serialize_episodes,
    summarize_thread,
)

NOW = datetime(2026, 6, 21, 12, 0, 0)


def _thread(m: Remembox, thread_id: str, n: int, *, words: int = 200,
            start: datetime = NOW - timedelta(hours=50)) -> None:
    """Record n chunky episodes into one thread, oldest first."""
    for i in range(n):
        m.record(
            f"Episode {i}: " + ("lorem ipsum dolor " * words),
            importance=0.3 + (i % 5) * 0.1,
            timestamp=start + timedelta(hours=i),
            thread_id=thread_id,
        )


class TestSerialization:
    def test_truncates_long_episodes(self):
        ep = Episode(content="x" * 5000, timestamp=NOW)
        out = serialize_episodes([ep], max_chars=2000)
        assert "truncated" in out
        assert len(out) < 5000

    def test_labels_each_episode(self):
        eps = [
            Episode(content="hello", timestamp=NOW, source="conversation"),
            Episode(content="world", timestamp=NOW, emotion="happy"),
        ]
        out = serialize_episodes(eps)
        assert "hello" in out and "world" in out
        assert "[happy]" in out


class TestCutPoint:
    def test_keeps_at_least_one_recent(self):
        eps = [Episode(content="a " * 500, timestamp=NOW + timedelta(hours=i))
               for i in range(5)]
        cut = _find_cut_index(eps, keep_recent_tokens=10, max_chars=2000)
        assert cut == len(eps) - 1  # tiny budget keeps just the newest

    def test_summarizes_at_least_one_when_budget_huge(self):
        eps = [Episode(content="a", timestamp=NOW + timedelta(hours=i))
               for i in range(3)]
        cut = _find_cut_index(eps, keep_recent_tokens=10_000, max_chars=2000)
        assert cut == 1  # always fold at least the oldest


class TestRuleBasedSummarizer:
    def test_empty_returns_previous(self):
        s = RuleBasedSummarizer()
        assert s.summarize([], previous_summary="prev") == "prev"

    def test_structured_sections(self):
        s = RuleBasedSummarizer()
        eps = [Episode(content=f"did thing {i}", timestamp=NOW, importance=0.5)
               for i in range(3)]
        out = s.summarize(eps)
        assert "## Goal" in out
        assert "## Progress" in out
        assert "## Critical Context" in out

    def test_compounds_previous_summary(self):
        s = RuleBasedSummarizer()
        eps = [Episode(content="new work", timestamp=NOW)]
        out = s.summarize(eps, previous_summary="## Goal\nold goal")
        assert "Earlier Summary" in out
        assert "old goal" in out

    def test_custom_instructions(self):
        s = RuleBasedSummarizer()
        eps = [Episode(content="work", timestamp=NOW)]
        out = s.summarize(eps, custom_instructions="focus on blockers")
        assert "focus on blockers" in out


class TestSummarizeThread:
    def test_short_thread_is_noop(self):
        m = Remembox(":memory:")
        m.record("only one", thread_id="t1", timestamp=NOW)
        result = m.summarize_thread("t1", now=NOW)
        assert not result.did_summarize
        assert result.summarized_ids == []

    def test_compresses_and_keeps_recent(self):
        m = Remembox(":memory:")
        _thread(m, "t1", 12)
        result = m.summarize_thread("t1", now=NOW)
        assert result.did_summarize
        assert result.summarized_ids
        assert result.kept_ids
        # Summarized episodes are marked consolidated (out of active set).
        for ep_id in result.summarized_ids:
            assert m._episodic.get(ep_id).consolidated
        # Compression actually reduced the token footprint.
        assert result.tokens_after < result.tokens_before

    def test_summary_episode_persisted_in_thread(self):
        m = Remembox(":memory:")
        _thread(m, "t1", 12)
        result = m.summarize_thread("t1", now=NOW)
        summaries = [e for e in m.thread("t1") if e.source == "thread_summary"]
        assert len(summaries) == 1
        assert summaries[0].id == result.summary_episode.id
        assert summaries[0].context["summarized_count"] == len(result.summarized_ids)

    def test_repeated_summarization_compounds(self):
        m = Remembox(":memory:")
        _thread(m, "t1", 12)
        m.summarize_thread("t1", now=NOW)
        # Add more and summarize again; prior summary should be folded in.
        for i in range(12, 20):
            m.record("more " * 200, thread_id="t1",
                     timestamp=NOW + timedelta(hours=i))
        result2 = m.summarize_thread("t1", now=NOW + timedelta(hours=21))
        assert result2.did_summarize
        assert "Earlier Summary" in result2.summary_episode.content

    def test_custom_summarizer_injection(self):
        class TaggingSummarizer(Summarizer):
            def summarize(self, episodes, previous_summary=None,
                          custom_instructions=None):
                return f"CUSTOM::{len(episodes)}"

        m = Remembox(":memory:", summarizer=TaggingSummarizer())
        _thread(m, "t1", 12)
        result = m.summarize_thread("t1", now=NOW)
        assert result.summary_episode.content.startswith("CUSTOM::")


class TestMaintain:
    def test_maintain_runs_all_steps(self):
        m = Remembox(":memory:")
        m.record("User said: I prefer green tea",
                 timestamp=NOW - timedelta(hours=2))
        report = m.maintain(now=NOW)
        assert "consolidate" in report
        assert "forget" in report
        assert "summarized_threads" in report
        # auto_reflect defaults to False, so reflect should be skipped.
        assert "reflect" not in report

    def test_maintain_honors_auto_reflect(self):
        cfg = MemoryConfig(auto_reflect=True)
        m = Remembox(":memory:", config=cfg)
        m.record("happy day", emotion="happy", timestamp=NOW - timedelta(days=2))
        report = m.maintain(now=NOW)
        assert "reflect" in report

    def test_maintain_auto_summarizes_big_threads(self):
        cfg = MemoryConfig(summary_trigger_tokens=500,
                           summary_keep_recent_tokens=200)
        m = Remembox(":memory:", config=cfg)
        _thread(m, "big", 12)
        report = m.maintain(now=NOW)
        assert len(report["summarized_threads"]) >= 1

    def test_maintain_skips_summary_when_disabled(self):
        cfg = MemoryConfig(summary_trigger_tokens=None)
        m = Remembox(":memory:", config=cfg)
        _thread(m, "big", 12)
        report = m.maintain(now=NOW)
        assert report["summarized_threads"] == []


class TestTokenEstimate:
    def test_estimate_nonzero(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("a" * 40) == 10
