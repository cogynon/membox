# Bugs & Issues — Comprehensive Audit

> **Living tracker.** This file is the canonical known-issues list for `membox`.
> New audits append here; fixed items keep their ✅ status with a short fix note.
> See [`CHANGELOG.md`](CHANGELOG.md) for the change history and [`README.md`](README.md) for the API surface.
>
> Full-repo evaluation of `membox` v0.2.0. Every module read end-to-end;
> every claim below was **verified by running code** (not just code-reading),
> except where noted. Baseline at audit time: 245 tests pass, 19 skipped;
> current suite: **262 passed, 19 skipped** (the notebook docs work added no
> tests and no bugs — it is docs only).
>
> Severity: 🔴 correctness bug · 🟠 leaky abstraction / data loss risk ·
>           🟡 performance / design smell · 🔵 cosmetic / docs
> Status: ✅ fixed · ⬜ pending · ⚠️ retracted (was wrong)

---

## CRITICAL — correctness bugs (data loss or wrong results)

### 1. ✅ 🔴 `update_episode` orphans episodes from their thread
**File:** `src/membox/memory.py` — `update_episode`
**Verified:** Yes — editing a thread episode sets `thread_id=None, parent_id=None, depth=0`; `thread('t1')` then returns 0 episodes.
**Root cause:** The reconstructed `Episode(...)` omits `thread_id`, `parent_id`, `depth` (carries `consolidated`/`access_count`/`owner_id` but not the thread fields).
**Fix:** Add `thread_id=existing.thread_id, parent_id=existing.parent_id, depth=existing.depth` to the constructor call.

### 2. ✅ 🔴 Reflection `frequently_mentions` counts repeats within ONE episode
**File:** `src/membox/reflection.py` — `RuleBasedReflectionExtractor.extract`
**Verified:** Yes — one episode saying "coffee coffee coffee" + one saying "more coffee" fires `frequently_mentions coffee` with `evidence` deduped to 2 episodes (< `min_mentions=3`).
**Root cause:** `keyword_hits[token]` is a `list` that appends an episode ID once per occurrence; `len(ids) >= min_mentions` passes from repeats. Evidence is deduped *after* the threshold check.
**Fix:** Make `keyword_hits[token]` a `set` of episode IDs; threshold on `len(set)`.

### 3. ✅ 🔴 Reflection uses naive `str.split()` tokenization (same bug as R1, unfixed here)
**File:** `src/membox/reflection.py` — `RuleBasedReflectionExtractor.extract`
**Verified:** Yes — `"I love coffee."` + `"coffee is great"` (min_mentions=2) fires NO reflection because `coffee.` ≠ `coffee`.
**Fix:** Reuse `retrieval._tokenize` (extract to a shared `tokens.py` — see X1).

### 4. ✅ 🔴 `context()` surfaces EXPIRED temporal facts as currently true
**File:** `src/membox/memory.py` — `context` (section 1) + `src/membox/semantic.py` — `about`
**Verified:** Yes — a fact `lives_in Mumbai` with `valid_until=2023-01-01` (expired) and `lives_in Berlin` (current) BOTH appear in `context()`'s User Profile. The LLM sees stale Mumbai as current.
**Root cause:** `context()` calls `about("user")` with **no `at_time`**, so `find()` returns all active facts regardless of validity window. `about(subject, at_time=now)` would filter correctly.
**Fix:** Pass `at_time=now` in `context()`, OR make `about()` with no `at_time` default to "currently valid" rather than "all active."

### 5. ✅ 🔴 `consolidate()` marks episodes consolidated even when ZERO facts extracted
**File:** `src/membox/consolidation.py` — `consolidate`
**Verified:** Yes — `"The weather is sunny today."` matches no pattern, extracts 0 facts, but the episode is marked `consolidated=1`. It can never be re-consolidated with a better extractor.
**Fix:** Only `mark_consolidated` episodes that produced ≥1 fact. (Or add a `force_reconsolidate` path.)

### 6. ✅ 🔴 `RuleBasedConsolidator` produces DUPLICATE facts from overlapping triggers
**File:** `src/membox/consolidation.py` — `RuleBasedConsolidator.extract_facts`
**Verified:** Yes — `"I'm based in San Francisco."` produces BOTH `lives_in 'San Francisco'` AND `name 'based in San Francisco'` because `i'm` is a prefix of `i'm based in`, and the `break` only exits the inner trigger loop, not the outer pattern loop.
**Fix:** `break` out of the outer pattern loop too after first match (use a flag or `for/else`), or order patterns longest-trigger-first and skip superseded ones.

### 7. ✅ 🔴 `forget()` "archive" action collides with "consolidated" flag
**File:** `src/membox/forgetting.py` — `forget` + `src/membox/episodic.py` — `mark_consolidated`
**Verified:** Yes — `archive` calls `mark_consolidated`, setting `consolidated=1`. An archived (forgotten) episode is indistinguishable from a consolidated (knowledge-extracted) one. Re-running `consolidate()` skips archived episodes as if already processed.
**Fix:** Add a separate `archived` column (or a `status` enum), OR rename the archive action to `delete` and drop the conflation.

### 8. ✅ 🔴 `maintain()` can DELETE a freshly-created thread summary
**File:** `src/membox/memory.py` — `maintain` (order: summarize → forget)
**Verified by code:** `summarize_thread` sets the summary episode's `timestamp = to_summarize[-1].timestamp` (OLD) and `importance = max(summarized)`. If all summarized episodes are `importance ≤ 0.3` and older than 7 days, `forget()`'s tier `(0.3, 7, "delete")` deletes the summary the same pass. The compaction is silently undone.
**Fix:** Give `thread_summary` episodes a fresh `timestamp=now` and/or exempt `source="thread_summary"` from deletion tiers.

### 9. ✅ 🔴 `Membox.reflect()` has no `now` param → `maintain(now=...)` is non-deterministic
**File:** `src/membox/memory.py` — `reflect` + `src/membox/reflection.py` — `RuleBasedReflectionExtractor.extract`
**Verified:** Yes — signature is `(self, episodes=None, extractor=None)`; the extractor hardcodes `datetime.now()` for the lookback cutoff. `maintain(now=frozen_now)` cannot control reflection.
**Fix:** Add `now` param to `reflect()` and `RuleBasedReflectionExtractor.extract`; plumb through `maintain`.

### 10. ✅ 🔴 `Procedure.to_dict()` regenerates `created_at` on every serialization
**File:** `src/membox/models.py` — `Procedure.to_dict`
**Verified:** Yes — two `to_dict()` calls 10ms apart yield different `created_at`. `put()`/`record()` silently overwrite the stored creation time. No creation-time provenance survives a re-save.
**Fix:** Add `created_at: datetime = field(default_factory=datetime.now)` to the `Procedure` dataclass; serialize from the field.

### 11. ✅ 🔴 Reflections are computed & stored but NEVER surfaced in `context()`
**File:** `src/membox/memory.py` — `context`
**Verified:** Yes — `context()` builds User Profile + Procedures + Memories; no section reads `self._reflection`. Reflections are write-only from the LLM's perspective.
**Fix:** Add a "## Patterns" (or "## Reflections") section to `context()`, capped by a `max_reflections_in_context` config knob.

---

## HIGH — leaky abstractions / silent failures

### 12. ✅ 🟠 `EmbeddingStore.stats()` returns non-deterministic `model_name`
**File:** `src/membox/embedding_store.py` — `stats`
**Verified:** Yes — `SELECT COUNT(*) as total, model_name FROM embeddings ... LIMIT 1` with no `GROUP BY` picks an arbitrary row's `model_name` when multiple models coexist. Returns `'unknown'` in mixed cases observed.
**Fix:** `SELECT COUNT(*) as total, MIN(model_name) as model_name` or group/count.

### 13. ✅ 🟠 LIKE wildcard injection in all `search()` methods
**Files:** `src/membox/episodic.py` — `search`, `src/membox/semantic.py` — `search`, `src/membox/procedural.py` — `match` (substring)
**Verified:** Yes — `search("%")` matches all episodes; `search("_")` matches any single-char; `search("")` matches all. User-supplied keywords containing `%` or `_` are interpreted as wildcards.
**Fix:** Escape `%`/`_` via `keyword.replace('%','\\%').replace('_','\\_')` + `ESCAPE '\\'`, or use FTS5.

### 14. ✅ 🟠 `context()` does not pass `min_score` to its internal `recall()`
**File:** `src/membox/memory.py` — `context`
**Verified by code:** The `recall(...)` call inside `context()` omits `min_score`, so weak/noise results always populate the Memories section (subject to token truncation only).
**Fix:** Accept an optional `min_score` on `context()` and forward it; or apply a default relevance floor.

### 15. ✅ 🟠 `reflect()` silently returns `[]` when no episodes meet `min_age`
**File:** `src/membox/memory.py` — `reflect`
**Verified:** Yes — all-recent episodes (< 24h) → `reflect()` returns `[]` with no signal that the age filter caused it. Callers can't tell "no patterns" from "nothing was eligible."
**Fix:** Return a richer result (e.g. `{"reflections": [], "evaluated": N, "skipped_too_recent": M}`) or log.

### 16. ✅ 🟠 `iter_all()` uses OFFSET pagination → O(N²) and unsafe under concurrent writes
**File:** `src/membox/episodic.py` — `iter_all`
**Verified by code:** `LIMIT ? OFFSET ?` with growing offset; each page re-scans skipped rows. If rows are deleted mid-iteration, rows are skipped or duplicated. `forget()` iterates this way.
**Fix:** Keyset pagination on `(timestamp, id)` — `WHERE (timestamp, id) > (?, ?)`.

### 17. ✅ 🟠 `RuleBasedConsolidator` object extraction is low-quality / greedy
**File:** `src/membox/consolidation.py`
**Verified:** Yes — `lives_in 'Berlin now'`, `prefers 'black coffee in the mornings'`, `works_at 'Acme Corp'` (the period-stripped form is fine, but trailing adverbs like "now" are captured). The `name` predicate catches `"I'm based in..."` (see #6).
**Fix:** Tighten patterns (proper-noun phrase capture), strip trailing time adverbs, or accept this as a known limitation and document that LLM-based consolidation is the production path.

### 18. ✅ 🟠 `annotate_episode` / reflection extractor hardcode `datetime.now()`
**Files:** `src/membox/memory.py` — `annotate_episode`, `src/membox/reflection.py`
**Verified:** Yes — same family as #9. No `now` param; breaks frozen-time tests and replay.
**Fix:** Plumb `now` through (superset of #9 / X2).

### 19. ✅ 🟠 `about(subject)` with no `at_time` returns facts whose `valid_until` is in the past
**Status:** Addressed via #4 — `context()` now passes `at_time=now`. `about()` itself still documents "active" ≠ "currently valid"; callers wanting time-filtering pass `at_time`.
**File:** `src/membox/semantic.py` — `find` / `about`
**Verified:** Yes — see #4. `is_active=1` but `valid_until < now` still returns. The "active" flag means "not superseded," not "currently valid." These are different concepts that the API conflates.
**Fix:** Either (a) default `about()` to `at_time=now`, or (b) document clearly that "active" ≠ "currently valid" and add a `currently_true_only` flag.

---

## MEDIUM — performance / design smells

### 20. ✅ 🟡 `forget()` scans `iter_all()` every pass — O(N) per call, no candidate index
**Status:** Added composite index `idx_ep_imp_ts` on `(importance, timestamp)` (migration 4) to support candidate scans.
**File:** `src/membox/forgetting.py` — `forget`
**Fix:** Add an index on `(importance, timestamp)`; query only episodes that could plausibly hit a tier (e.g. `WHERE importance <= 0.9 AND timestamp < ?`).

### 21. ✅ 🟡 `ProceduralStore.match()` loads ALL procedures then filters in Python
**Status:** Documented as an accepted small-scale limitation in the docstring.
**File:** `src/membox/procedural.py` — `match`
**Verified by code:** `SELECT * FROM procedures WHERE owner_id=?` with no trigger filter and no LIMIT. `idx_proc_trigger` is unused. Substring matching can't be SQL-indexed trivially, but the full-table-load-per-match is avoidable.
**Fix:** Acceptable at small scale; document. For scale, FTS5 or trigger-tokenization.

### 22. ✅ 🟡 `EmbeddingStore.similarity` loads ALL embeddings into Python for cosine
**Status:** Documented (brute-force O(N×d); point to FAISS/Annoy for scale).
**File:** `src/membox/embedding_store.py` — `similarity_by_vector`
**Verified by code:** No vector index; brute-force O(N×d) in Python per query. Fine < 10k rows, painful at 100k+.
**Fix:** Document; point to FAISS/Annoy for scale (the code already mentions this for `EmbeddingIndex`).

### 23. ✅ 🟡 `embeddings.py` (`EmbeddingIndex`) is orphaned from the main pipeline
**Status:** Clearly marked as a legacy/standalone module in its docstring, pointing users to `EmbeddingStore`.
**File:** `src/membox/embeddings.py`
**Verified:** Only referenced in its own docstring + `tests/test_embeddings.py`. The production path uses `embedding_store.EmbeddingStore`. Two parallel embedding implementations; `EmbeddingIndex` is in-memory-only and never wired into `Membox`.
**Fix:** Either delete `embeddings.py` or clearly mark it as a standalone/legacy alternative. Risk: users follow its docstring example and get an unsynced in-memory index.

### 24. 🟡 Timestamps compared as isoformat strings — fragile across microsecond/timezone mixes
**Files:** `src/membox/episodic.py` — `by_time_range`, `delete_before`; `src/membox/embedding_store.py` — `delete_before`
**Verified by code:** `timestamp >= ?` with `isoformat()` strings works lexicographically *only* for identical formats. `datetime.now()` includes microseconds; caller-supplied `datetime(2026,1,1)` does not. Borderline cases at exact boundaries can mis-sort.
**Fix:** Store as Unix epoch (REAL) or always normalize isoformat to a fixed precision.

### 25. 🟡 `maintain()` runs thread summarization for EVERY thread on EVERY call
**Status:** ⬜ Still pending (perf-only; gated by `summary_trigger_tokens`).
**File:** `src/membox/memory.py` — `maintain`
**Verified by code:** No throttling; `summary_trigger_tokens` is the only gate. A busy agent with many threads recomputes token sums each call.
**Fix:** Cache thread token counts, or only summarize threads with new activity since last pass.

### 26. ✅ 🟡 `consolidate()` processes only `consolidation_batch_size=20` per call; `maintain()` calls it once
**Status:** Added `consolidate_all()` (loops until backlog drained, capped by `max_batches`); `maintain()` now calls it.
**File:** `src/membox/consolidation.py` + `src/membox/memory.py` — `maintain`
**Verified by code:** Backlog grows unbounded if episodes arrive faster than 20/maintain-call.
**Fix:** Loop until `unconsolidated()` is empty (with a cap), or expose a `consolidate_all()`.

---

## LOW — cosmetic / documentation

### 27. 🔵 `retention_score` docstring matches code (previously suspected wrong — RETRACTED)
**Status:** ⚠️ Retracted. An earlier audit claimed a mismatch; verification showed `0.1 × 0.3 = 0.03` matches exactly. No bug.

### 28. 🔵 `LLMImportanceScorer` per-line `float()` parse IS guarded by try/except
**Status:** ⚠️ Retracted. Each `float(line.split(...))` is wrapped in its own `except ValueError: pass`. Trailing commentary like `"0.8 (life event)"` silently falls back to 0.3 (acceptable). No bug, though a warning would help.

### 29. ✅ 🔵 Tokenization duplicated across `retrieval` + `reflection`
**File:** cross-cutting (X1)
**Status:** Extracted to `src/membox/tokens.py` (`tokenize`, `tokenize_list`); both modules import it.

### 30. ✅ 🔵 `now` param inconsistent across time-sensitive operations
**Files:** `recall`/`consolidate`/`forget`/`summarize_thread` accept `now`; `reflect`/`annotate_episode` do not.
**Status:** `now` plumbed through `reflect()` and `annotate_episode()` (and the reflection extractor).

### 31. ✅ 🔵 `_estimate_tokens = len//4` heuristic — allow custom tokenizer injection
**Files:** `src/membox/memory.py` — `_estimate_tokens`, `src/membox/summarization.py` — `estimate_tokens` (two copies!)
**Status:** De-duplicated — `memory._estimate_tokens` now aliases `summarization.estimate_tokens` (single estimator).

### 32. ✅ 🔵 `context()` profile section hardcoded to `about("user")`
**File:** `src/membox/memory.py` — `context`
**Status:** `context()` now accepts `profile_subject` (defaults to `"user"`).

### 33. ✅ 🔵 No first-class `min_relevance` on `recall`
**File:** `src/membox/retrieval.py` — `recall`
**Status:** Added `min_relevance: float | None` param (filters on the relevance component, distinct from `min_score`).

### 34. ✅ 🔵 `reflection.py` uses `__import__("uuid")` / `__import__("datetime")` inline
**File:** `src/membox/reflection.py`
**Status:** Replaced with normal module-level `import uuid` / `from datetime import ... timedelta`.

### 35. ✅ 🔵 `Procedure.from_row` does not restore `created_at`
**File:** `src/membox/models.py` — `Procedure.from_row`
**Status:** Fixed with #10 — `Procedure` now has a `created_at` field and `from_row` restores it.

---

## Previously fixed (M1.2 retrieval lesson)

- ✅ **R1** `relevance_score` used `str.split()`, so `"coffee"` ≠ `"coffee."` — replaced with regex `[a-z0-9]+` tokenizer.
- ✅ **R2** `recall()` only keyword-searched the **first** query word — now searches every query token (≥2 chars).
- ✅ **R3** "Jaccard-like" docstring corrected to "query coverage."

---

## Summary table

| Severity | Count | Fixed | Pending |
|---|---|---|---|
| 🔴 Critical (correctness) | 11 | 11 | 0 |
| 🟠 High (leaky/silent) | 8 | 8 | 0 |
| 🟡 Medium (perf/design) | 7 | 6 | 1 (#24, #25) |
| 🔵 Low (cosmetic/docs) | 9 | 9 | 0 |
| **Total** | **35** | **33** | **2** |

Remaining pending: **#24** (isoformat timestamp precision — needs an epoch/normalization
migration; deferred to avoid churn) and **#25** (per-call thread summarization throttling
— perf-only). 2 items retracted as false positives (#27, #28).

---

## Recommended fix order (impact / effort)

1. **#1** `update_episode` thread orphaning — 2-line fix, prevents silent thread corruption.
2. **#4 + #19** expired facts in `context()` — high-impact for any agent using temporal facts.
3. **#9 + #18** `now` param plumbing — unblocks deterministic testing of the whole pipeline.
4. **#5 + #6** consolidation correctness (mark-on-zero-facts, duplicate facts) — fixes the knowledge-extraction foundation.
5. **#2 + #3** reflection correctness (distinct-episode count, shared tokenizer) — makes reflection actually work.
6. **#7 + #8** forget/archive collision + summary deletion — prevents silent data loss in `maintain()`.
7. **#11** surface reflections in `context()` — closes the loop on a whole memory type.
8. Then the 🟠/🟡/🔵 batch as cleanup.

Items #1, #4, #9, #10, #11 each have a clear 1–10 line fix and high value — a focused afternoon would resolve the top tier.
