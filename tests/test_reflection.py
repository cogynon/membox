"""Tests for reflection / pattern memory."""

import pytest

from membox import Membox
from membox.models import Episode
from membox.reflection import (
    Reflection,
    ReflectionStore,
    RuleBasedReflectionExtractor,
    reflect,
)


@pytest.fixture
def memory(tmp_path):
    m = Membox(str(tmp_path / "reflect.db"))
    yield m
    m.close()


class TestRuleBasedExtractor:
    def test_extracts_recurring_emotion(self):
        eps = [
            Episode(content="x", emotion="stressed"),
            Episode(content="y", emotion="stressed"),
            Episode(content="z", emotion="stressed"),
        ]
        extractor = RuleBasedReflectionExtractor(min_mentions=3)
        reflections = extractor.extract(eps)
        assert any(r.predicate == "often_feels" and r.object == "stressed" for r in reflections)

    def test_extracts_frequent_keyword(self):
        eps = [
            Episode(content="I love hiking"),
            Episode(content="hiking again today"),
            Episode(content="went hiking"),
        ]
        extractor = RuleBasedReflectionExtractor(min_mentions=3)
        reflections = extractor.extract(eps)
        assert any(r.predicate == "frequently_mentions" and r.object == "hiking" for r in reflections)

    def test_below_threshold_returns_empty(self):
        eps = [Episode(content="I like hiking")]
        extractor = RuleBasedReflectionExtractor(min_mentions=3)
        assert extractor.extract(eps) == []


class TestReflectionStore:
    def test_record_and_find(self, memory):
        r = Reflection(subject="user", predicate="often_feels", object="stressed")
        memory._reflection.record(r)
        found = memory._reflection.find("user")
        assert len(found) == 1
        assert found[0].object == "stressed"

    def test_deactivate(self, memory):
        r = Reflection(subject="user", predicate="often_feels", object="tired")
        memory._reflection.record(r)
        assert memory._reflection.deactivate(r.id) is True
        assert memory._reflection.count() == 0

    def test_owner_isolation(self, tmp_path):
        db = tmp_path / "shared.db"
        a = Membox(str(db), owner_id="alice")
        b = Membox(str(db), owner_id="bob")
        a._reflection.record(Reflection(subject="user", predicate="p", object="v"))
        assert a._reflection.count() == 1
        assert b._reflection.count() == 0
        a.close()
        b.close()


class TestReflectFunction:
    def test_persist_and_merge_evidence(self, memory):
        eps = [
            Episode(content="x", emotion="stressed"),
            Episode(content="y", emotion="stressed"),
            Episode(content="z", emotion="stressed"),
        ]
        results = reflect(eps, store=memory._reflection)
        assert len(results) >= 1
        r = results[0]
        assert r.object == "stressed"
        assert len(r.evidence) == 3

    def test_idempotent_merge(self, memory):
        eps = [Episode(content="x", emotion="stressed") for _ in range(3)]
        reflect(eps, store=memory._reflection)
        reflect(eps, store=memory._reflection)
        rows = memory._reflection.find("user", predicate="often_feels")
        assert len(rows) == 1
        assert len(rows[0].evidence) == 3


class TestMemboxIntegration:
    def test_reflect_defaults_pull_recent(self, memory):
        for _ in range(5):
            memory.record("I love hiking", emotion="happy")
        result = memory.reflect(episodes=[
            Episode(content="I love hiking", emotion="happy"),
            Episode(content="hiking again", emotion="happy"),
            Episode(content="more hiking", emotion="happy"),
        ])
        assert any(r.object == "happy" for r in result["reflections"])

    def test_reflections_api(self, memory):
        memory.reflect(episodes=[
            Episode(content="a", emotion="stressed"),
            Episode(content="b", emotion="stressed"),
            Episode(content="c", emotion="stressed"),
        ])
        found = memory.reflections("user")
        assert len(found) == 1
        assert found[0].predicate == "often_feels"
