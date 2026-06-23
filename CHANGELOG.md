# Changelog — membox v0.2.0

> All changes merged into the main package (no longer isolated in `v2/`).
Newest entries at the top. See [`BUGS.md`](BUGS.md) for the known-issues tracker.

---

## Unreleased

### Docs: runnable notebooks + documentation index

- **Added** `notebooks/quickstart.ipynb` — a 5-minute executable path from install to a memory-augmented prompt (8 cells, verified end-to-end).
- **Added** `notebooks/walkthrough.ipynb` — a 17-section simple→advanced tour covering every public API: episodic/semantic/procedural memory, temporal facts, retrieval scoring, context builder, threads + `summarize_thread`, consolidation, reflection, tiered forgetting, `maintain()`, config presets, editing/annotations, multi-user isolation, optional embeddings, and the full LLM loop (18 cells, verified).
- **Added** `notebooks/_build_notebooks.py` — generator that builds both `.ipynb` files from readable cell lists.
- **Added** `notebooks/README.md` — index + run/rebuild instructions.
- **Changed** `README.md` — added a `Documentation` index linking to `CHANGELOG.md`, `BUGS.md`, `notebooks/`, `lessons/`, `demos/`; pointed the Quick Start at the new notebooks; fixed the stale test count (`129` → `262 passed, 19 skipped`); removed stray non-doc text that had been appended at the end of the file.
- **Changed** `BUGS.md` — added a living-tracker status header noting the audit is current through this release.
- **Removed** `OBSERVATIONS.md` — the older v2-era architecture audit, superseded by `BUGS.md` (per-issue tracker) and this changelog (P0/P1/P2 history). Its gap analysis had gone stale (it listed multi-user isolation, procedural memory, reflection, and temporal facts as "missing" — all shipped in v0.2.0) and it was not referenced anywhere in the repo. The original is preserved in git history.
- **Added** `integrations/pi/` — long-term memory for [pi](https://github.com/earendil-works/pi-mono): a stdlib-only Python HTTP sidecar (`server.py`) exposing `record`/`learn`/`recall`/`context`/`maintain`; a `CodingImportanceScorer` (`coding.py`) that exercises the library's pluggable scorer with coding-tuned heuristics; a TypeScript pi extension (`index.ts`) wiring the sidecar into pi's lifecycle (`session_start` → spawn, `before_agent_start` → inject context, `agent_end` → record, `session_shutdown` → maintain) plus `/memory`, `/remember`, `/memory-stats` commands. 18 integration tests (`test_server.py`).
- **Added** two-tier memory: a **global** sidecar (`~/.pi/agent/memory.db`, `owner=user`, cross-project) + a **project** sidecar (`<cwd>/.pi/memory.db`, `owner=project`, per-codebase). `before_agent_start` recalls from both and merges with `[global]`/`[project]` headers; `agent_end` records to the project store (or global fallback). The global sidecar persists across sessions for amortized startup.
- **Added** `LLMConsolidator` (`llm_consolidator.py`) — LLM-based fact extraction via any OpenAI-compatible client (OpenRouter, OpenAI, Ollama). Handles `content=None` from reasoning models, strips markdown fences, clamps confidence, fails soft. Wired into the server via `--llm-consolidator`/`--llm-base-url`/`--llm-api-key` flags. 8 mock-client tests.
- **Changed** the extension from `long-term-memory.ts` to `index.ts` (pi requires `index.ts` for subdirectory auto-discovery — the old name silently failed to load). Added file logging (`extension.log`, `sidecar-global.log`, `sidecar-project.log`) and a `recordedThisTurn` guard against double-recording.
- **Verified live** with OpenRouter (`meta-llama/llama-3.1-8b-instruct`): two sidecars spawn, project memory activates when `<cwd>/.pi/` exists, cross-session recall injects memory into session 2, LLM consolidation extracts 3 facts from 4 episodes (skipping transient questions).
- **Added** recording of the assistant's final reply (not just the user prompt). `agent_end` now records both `User: <prompt>` and `Assistant: <reply>`, truncated to `PI_MEMORY_REPLY_MAX_CHARS` (default 1500, `…[truncated]` marker beyond). Closes the one-sided-memory gap — the answer/resolution is often more worth remembering than the question. The `LLMConsolidator` strips both `User:`/`Assistant:` prefixes and truncates per-episode (1000 chars) so long replies don't blow up the consolidation prompt. 2 new tests (assistant-prefix strip, long-episode truncation).
- **Added** OpenAPI 3.0 spec + Swagger UI to the sidecar (no framework, no deps): `GET /openapi.json` serves a hand-written spec; `GET /docs` (and `/`) serve a self-contained Swagger UI page (renderer from CDN) with try-it-out enabled. Covers all 8 endpoints with input/response schemas, including the `/learn` gotcha (JSON key is `object`, not `obj`). 2 new tests (spec shape + docs HTML).
- **Added** `GET /facts` endpoint to list stored facts (consolidated knowledge), highest-confidence first. Supports query filters: `?subject=user`, `?subject=user&predicate=prefers`, `?q=postgres` (keyword search). Documented in the OpenAPI spec with a `Fact`/`FactList` schema. 4 new tests (list all, filter by subject, filter by subject+predicate, keyword search).
- **Notebooks introduce no code changes and no new bugs** — they are docs only; all 262 core tests still pass.

---

## P0 — Critical Fixes

### P0.1: Multi-User Isolation ✅

**Problem:** All episodes and facts hardcoded `subject="user"` with no `owner_id`. Multiple users sharing one DB would contaminate each other's memory.

**Solution:** Added `owner_id` scoping at the SQLite schema level. All stores accept `owner_id` and filter every query by it.

```python
alice = Membox("shared.db", owner_id="alice")
bob   = Membox("shared.db", owner_id="bob")

alice.record("Alice's secret")
bob.record("Bob's secret")

alice.recent(10)  # only Alice's episode
bob.recent(10)    # only Bob's episode
```

**Tests:** `tests/test_multi_user.py` — 13 tests ✅

---

### P0.2: Embedding ↔ SQLite Sync ✅

**Problem:** `EmbeddingIndex` stored vectors in-memory only. No persistence, no sync with deletes.

**Solution:** Created `EmbeddingStore` — SQLite-backed embedding persistence in the same `.db` file as episodes. Embeddings are created on `record()`, deleted on `forget()`, and scoped by `owner_id`.

**Tests:** `tests/test_embedding_store.py` — 12 tests ✅ (skipped when `sentence-transformers` unavailable)

---

### P0.3: In-Memory SQLite Connection Isolation ✅

**Problem:** With `db_path=":memory:"`, each store opened its own isolated in-memory database. Cross-store queries like `DELETE FROM embeddings WHERE episode_id IN (SELECT id FROM episodes)` crashed with `no such table: episodes`.

**Solution:** `Membox` now creates one shared SQLite connection for all stores when `db_path=":memory:"`. File databases still use separate connections for WAL concurrency.

**Tests:** `tests/test_in_memory_connection.py` — 6 tests ✅

---

### P0.4: Connection Leak / Double-Close on `close()` ✅

**Problem:** `Membox.close()` did not close the embedding store. After adding shared connections, stores could also double-close the same connection.

**Solution:** Stores track whether they own their connection (`_owns_connection`) and only close connections they created. `Membox.close()` now closes episodic, semantic, procedural, embedding, and shared memory connections correctly.

---

## P1 — High-Impact Improvements

### P1.1: Storage Protocol Completeness ✅

**Problem:** `EpisodicStoreProtocol` and `SemanticStoreProtocol` in `_store.py` omitted methods that `forgetting.py` and `consolidation.py` actually called (`iter_all`, `delete`, `mark_consolidated`, `unconsolidated`, `learn`). Implementing a custom backend via the protocol caused `AttributeError` during maintenance.

**Solution:** Protocols now declare all required methods. Maintenance modules are typed against the protocols, not concrete SQLite classes.

**Tests:** `tests/test_store_protocol.py` — 3 tests ✅

---

### P1.2: Relevance Threshold + Normalized Scoring ✅

**Problem:** `recall(query, k=5)` always returned 5 results, even if all matches were near-zero relevance. Also, custom weights that didn't sum to 1.0 produced unnormalized scores.

**Solution:**
- Added `min_score` parameter to `recall()` and `Membox.recall()`
- Normalized combined retrieval score by sum of weights

```python
results = memory.recall("hobbies", k=5, min_score=0.4)
# Only memories with score ≥ 0.4 are returned
```

**Tests:** `tests/test_retrieval.py` ✅

---

### P1.3: Pluggable Importance + Emotion Scorer ✅

**Problem:** Importance scoring was manual or relied on hardcoded keyword heuristics. No LLM-based auto-scoring.

**Solution:** Added `ImportanceScorer` abstraction:
- `RuleBasedImportanceScorer` — dependency-free default
- `LLMImportanceScorer` — inject any OpenAI-compatible LLM client

```python
scorer = LLMImportanceScorer(client=openai_client)
memory = Membox("agent.db", importance_scorer=scorer)
memory.record("I got promoted!")  # auto-scored + emotion detected
```

**Tests:** `tests/test_importance.py` — 13 tests ✅

---

### P1.4: Procedural Memory ✅

**Problem:** The third memory pillar taught in the lessons (routines/skills) was missing from the production package.

**Solution:**
- Added `Procedure` model
- Added `ProceduralStore` with `owner_id` scoping
- `Membox` gains `learn_procedure()`, `match_procedures()`, `procedures()`, `delete_procedure()`
- Active procedures are injected into `context()` output

```python
memory.learn_procedure("goodnight", "Dim lights, set alarm 6:30am")
memory.context("Goodnight")  # includes the active procedure
```

**Tests:** `tests/test_procedural.py` — 9 tests ✅

---

### P1.5: In-Memory Embedding Index Hardening ✅

**Problem:** The legacy `EmbeddingIndex` was not thread-safe and had no removal path, causing memory leaks and race conditions.

**Solution:** Added `threading.RLock` around all operations and a `remove(episode_ids)` method. New code should prefer `EmbeddingStore`, but `EmbeddingIndex` is now safe for lightweight use.

---

### P1.6: Consolidator API Alignment ✅

**Problem:** `Consolidator.extract(contents: list[str])` threw away all episode metadata, preventing custom consolidators from using importance, emotion, or timestamp.

**Solution:**
- New primary method: `extract_facts(episodes: list[Episode])`
- Old `extract(contents)` kept as a backward-compatible shim
- `consolidate()` now links each extracted fact to its source episode ID

**Tests:** `tests/test_consolidator_api.py` — 3 tests ✅

---

### P1.7: SQLite Schema Migrations ✅

**Problem:** `CREATE TABLE IF NOT EXISTS` cannot add new columns. Upgrading an existing database after schema changes crashed with `no such column: owner_id`.

**Solution:** Added lightweight migration framework in `migrations.py`:
- Tracks `_schema_version` in the database
- Runs numbered migrations idempotently
- Migration 1 adds `owner_id` columns and indexes to legacy tables

**Tests:** `tests/test_migrations.py` — 3 tests ✅

---

### P1.8: Memory Editing & Correction ✅

**Problem:** Episodes were immutable once recorded; there was no way to fix a misrecorded event or flag it as inaccurate. Semantic facts could be contradicted but not explicitly corrected.

**Solution:** Added editing and annotation APIs:
- `Membox.update_episode()` — edit content, importance, emotion, source, context, or timestamp in place
- `Membox.annotate_episode()` — append timestamped corrections/accuracy flags/notes to an episode's audit trail
- `Membox.edit_fact()` — in-place fact edits
- `Membox.correct_fact()` — correct a fact while deactivating the old version to preserve history

```python
memory.record("User lives in Mumbai")
ep = memory.recent(1)[0]
memory.update_episode(ep.id, content="User lives in Delhi")
memory.annotate_episode(ep.id, correction="User corrected city", accuracy="verified")
```

**Tests:** `tests/test_memory_editing.py` — 15 tests ✅

---

### P2.1: Reflection / Pattern Memory ✅

**Problem:** The system stored isolated episodes and facts but never synthesized *patterns* across multiple episodes (e.g. "User gets stressed every quarter" or "User prefers quiet places on weekends").

**Solution:** Added a reflection layer:
- `Reflection` model + `ReflectionStore` with owner-aware persistence
- `RuleBasedReflectionExtractor` detects recurring emotions and frequent keywords as safe default
- `Membox.reflect()` runs reflection over recent episodes
- `Membox.reflections(subject)` queries discovered patterns
- Pluggable `ReflectionExtractor` interface for LLM-based pattern synthesis

```python
memory.reflect()  # synthesize patterns from recent episodes
memory.reflections("user")
```

**Tests:** `tests/test_reflection.py` — 10 tests ✅

---

### P2.2: Temporal / Recurring Facts ✅

**Problem:** Semantic facts were static triples with no validity window or recurrence. `user → works_at → Google` would be stored forever, even if the user changed jobs.

**Solution:** Added temporal fields to `Fact` and the semantic store:
- `valid_from` / `valid_until` windows
- `recurrence` field for patterns like "weekday mornings"
- `Membox.about(subject, at_time=...)` queries facts known to be true at a specific time
- Temporal-aware contradiction: non-overlapping values for the same predicate can coexist

```python
memory.learn("user", "works_at", "Google", valid_from=date(2020,1,1), valid_until=date(2023,1,1))
memory.learn("user", "works_at", "OpenAI", valid_from=date(2023,1,2), valid_until=date(2025,1,1))
memory.about("user", at_time=date(2024,1,1))  # OpenAI
```

**Tests:** `tests/test_temporal_facts.py` — 7 tests ✅

---

## P2 — Thread Summarization (pi-inspired compaction)

### P2.3: Persist thread hierarchy on write ✅

**Problem:** Migration 3 added `thread_id` / `parent_id` / `depth` columns and
read APIs (`by_thread`, `by_parent`, `threads`), but `record()`,
`record_batch()`, and `update()` never listed those columns in their INSERTs.
Thread data was silently dropped on write, so 4 of 6 `test_threads.py` tests
failed and threads were unusable end-to-end.

**Solution:** Added the three columns to all episode INSERT statements (values
already present in `Episode.to_dict()`).

### P2.4: Thread Summarization ✅

**Idea (borrowed from [pi](https://github.com/earendil-works/pi-mono)'s
compaction):** long conversation threads grow unbounded. Summarize the older
portion in place while keeping recent episodes verbatim.

```python
result = memory.summarize_thread("incident-42")
# older episodes → one structured `thread_summary` episode (marked consolidated)
# recent episodes kept intact; result reports tokens_before/after
```

- Token-budgeted cut point (`summary_keep_recent_tokens`) — keep recent context.
- Structured summary format (Goal / Progress / Critical Context), like pi.
- Per-episode input truncation (`max_serialized_chars`) so a few huge episodes
  don't dominate the summary input.
- Iterative compounding: a prior `thread_summary` is folded into the next one.
- Pluggable `Summarizer` ABC (matches existing `Consolidator` /
  `ReflectionExtractor` pattern); ships dependency-free `RuleBasedSummarizer`.

### P2.5: Unified `maintain()` + wire up `auto_reflect` ✅

**Problem:** `config.auto_reflect` was declared but referenced nowhere (dead
flag). Callers also had to orchestrate consolidate/reflect/forget by hand.

**Solution:** Added `Membox.maintain()` — one call runs consolidate →
reflect (if `auto_reflect`) → summarize oversized threads (if
`summary_trigger_tokens`) → forget. Mirrors pi's single auto-compaction trigger.

**Tests:** `tests/test_summarization.py` — 18 tests ✅

---

## Test Summary

```
Current: 262 passed, 19 skipped, 1 warning
```

| Test File | Coverage |
|-----------|----------|
| `tests/test_multi_user.py` | Multi-user isolation |
| `tests/test_embedding_store.py` | SQLite-backed embedding sync |
| `tests/test_in_memory_connection.py` | Shared `:memory:` connection |
| `tests/test_store_protocol.py` | Storage protocol completeness |
| `tests/test_retrieval.py` | Retrieval scoring + `min_score` |
| `tests/test_importance.py` | Importance/emotion scorers |
| `tests/test_procedural.py` | Procedural memory |
| `tests/test_consolidator_api.py` | Episode-based consolidators |
| `tests/test_migrations.py` | SQLite schema migrations |
| `tests/test_memory_editing.py` | In-place episode/fact editing |
| `tests/test_reflection.py` | Higher-order pattern synthesis |
| `tests/test_temporal_facts.py` | Temporal / recurring facts |
| `tests/test_threads.py` | Thread hierarchy persistence |
| `tests/test_summarization.py` | pi-style thread summarization + `maintain()` |

---

## Repo Organization

- `src/membox/` — production package
- `tests/` — full test suite
- `lessons/` — 8 lesson scripts (gitignored, not part of package)
- `demos/` — `demo_openrouter.py` (gitignored, not part of package)

---

## Branch Protection

Direct commits to `dev` are blocked by `.git/hooks/pre-commit`. All changes land via feature-branch merges.
