"""Tests for temporal / recurring fact support."""

from datetime import datetime, timedelta

import pytest

from membox import Membox
from membox.models import Fact


@pytest.fixture
def memory(tmp_path):
    m = Membox(str(tmp_path / "temporal.db"))
    yield m
    m.close()


class TestTemporalLearn:
    def test_learn_with_valid_window(self, memory):
        start = datetime(2020, 1, 1)
        end = datetime(2022, 1, 1)
        fact, action = memory.learn(
            "user", "works_at", "Google",
            valid_from=start, valid_until=end,
        )
        assert action == "new"
        assert fact.valid_from == start
        assert fact.valid_until == end

    def test_query_as_of_returns_fact_in_window(self, memory):
        memory.learn(
            "user", "works_at", "Google",
            valid_from=datetime(2020, 1, 1), valid_until=datetime(2022, 1, 1),
        )
        results = memory.about("user", at_time=datetime(2021, 6, 1))
        assert len(results) == 1
        assert results[0].object == "Google"

    def test_query_as_of_excludes_fact_outside_window(self, memory):
        memory.learn(
            "user", "works_at", "Google",
            valid_from=datetime(2020, 1, 1), valid_until=datetime(2022, 1, 1),
        )
        results = memory.about("user", at_time=datetime(2023, 1, 1))
        assert len(results) == 0

    def test_non_overlapping_values_coexist(self, memory):
        memory.learn(
            "user", "works_at", "Google",
            valid_from=datetime(2018, 1, 1), valid_until=datetime(2020, 1, 1),
        )
        memory.learn(
            "user", "works_at", "OpenAI",
            valid_from=datetime(2020, 1, 2), valid_until=datetime(2022, 1, 1),
        )
        assert memory.about("user", at_time=datetime(2019, 1, 1))[0].object == "Google"
        assert memory.about("user", at_time=datetime(2021, 1, 1))[0].object == "OpenAI"

    def test_recurrence_stored(self, memory):
        fact, _ = memory.learn(
            "user", "hobby", "running",
            recurrence="weekday_mornings",
        )
        assert fact.recurrence == "weekday_mornings"


class TestTemporalContradiction:
    def test_overlapping_contradict_deactivates_old(self, memory):
        old, _ = memory.learn(
            "user", "works_at", "Google",
            valid_from=datetime(2020, 1, 1), valid_until=datetime(2025, 1, 1),
        )
        new, action = memory.learn(
            "user", "works_at", "OpenAI",
            valid_from=datetime(2022, 1, 1), valid_until=datetime(2026, 1, 1),
        )
        assert action == "contradicted"
        assert memory._semantic.get(old.id).is_active is False
        assert new.is_active is True


class TestFactRoundTrip:
    def test_put_and_get_preserves_temporal_fields(self, memory):
        start = datetime(2021, 3, 15)
        end = datetime(2022, 3, 15)
        fact = Fact(
            subject="user", predicate="lives_in", object="Berlin",
            valid_from=start, valid_until=end, recurrence="summer",
        )
        memory._semantic.put(fact)
        fetched = memory._semantic.get(fact.id)
        assert fetched.valid_from == start
        assert fetched.valid_until == end
        assert fetched.recurrence == "summer"
