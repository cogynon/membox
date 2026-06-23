"""Tests for procedural memory (routines/skills)."""

import pytest

from membox import Membox
from membox.models import Procedure
from membox.procedural import ProceduralStore


class TestProceduralStore:

    def test_record_and_match(self):
        store = ProceduralStore(":memory:")
        proc = store.record("goodnight", "Dim lights, set alarm 6:30am")
        assert proc.trigger == "goodnight"

        matches = store.match("Goodnight Jarvis")
        assert len(matches) == 1
        assert matches[0].action == "Dim lights, set alarm 6:30am"

    def test_match_is_case_insensitive(self):
        store = ProceduralStore(":memory:")
        store.record("coffee", "Order black coffee")
        matches = store.match("I need COFFEE")
        assert len(matches) == 1

    def test_match_best_returns_highest_confidence(self):
        store = ProceduralStore(":memory:")
        store.record("run", "Track 5km", confidence=0.5)
        store.record("run", "Play upbeat music", confidence=0.9)
        best = store.match_best("I went for a run")
        assert best is not None
        assert best.action == "Play upbeat music"

    def test_owner_isolation(self, tmp_db):
        alice = ProceduralStore(tmp_db, owner_id="alice")
        bob = ProceduralStore(tmp_db, owner_id="bob")

        alice.record("morning", "Alice's morning briefing")
        bob.record("morning", "Bob's morning briefing")

        assert len(alice.match("morning")) == 1
        assert alice.match("morning")[0].action == "Alice's morning briefing"

    def test_delete(self):
        store = ProceduralStore(":memory:")
        proc = store.record("test", "do something")
        assert store.count() == 1

        assert store.delete(proc.id) is True
        assert store.count() == 0
        assert store.delete(proc.id) is False


class TestMemboxProcedural:

    def test_learn_and_match_procedure(self):
        memory = Membox(":memory:")
        proc = memory.learn_procedure("server outage", "Run diagnostics immediately", confidence=0.95)
        assert isinstance(proc, Procedure)

        matches = memory.match_procedures("We have a server outage!")
        assert len(matches) == 1
        assert matches[0].action == "Run diagnostics immediately"

    def test_procedures_list(self):
        memory = Membox(":memory:")
        memory.learn_procedure("a", "do A")
        memory.learn_procedure("b", "do B")
        assert len(memory.procedures()) == 2

    def test_context_includes_active_procedures(self):
        memory = Membox(":memory:")
        memory.learn_procedure("goodnight", "Dim lights, set alarm", confidence=0.9)
        ctx = memory.context("Goodnight Jarvis")
        assert "Active Procedures" in ctx
        assert "Dim lights" in ctx

    def test_stats_includes_procedures(self):
        memory = Membox(":memory:")
        memory.learn_procedure("x", "do X")
        stats = memory.stats()
        assert stats["procedures"]["total"] == 1
