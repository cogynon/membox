# remembox

**Production-grade plug-and-play memory for AI agents.**

Give any LLM, agent, or rule-based system persistent episodic + semantic memory. Zero config, single-file storage, framework-agnostic.

```python
from remembox import Remembox

memory = Remembox("my_agent.db")
memory.record("User said they love hiking in the Himalayas", importance=0.8)
memory.learn("user", "prefers", "black coffee", confidence=0.9)

# Retrieve relevant memories
results = memory.recall("what are the user's hobbies?", k=3)

# Get prompt-ready context string
context = memory.context("what does the user like?")
# → "## User Profile\n- user prefers black coffee (90%)\n\n## Relevant Memories\n- ..."
```

## Features

| Feature | Description |
|---|---|
| **Episodic memory** | Timestamped events with importance + emotion scoring |
| **Semantic memory** | Fact triples with reinforcement & contradiction |
| **Procedural memory** | Trigger → action routines, surfaced in context |
| **Temporal facts** | `valid_from` / `valid_until` windows + recurrence patterns |
| **Reflection** | Synthesizes higher-order patterns across episodes |
| **Smart retrieval** | Recency × relevance × importance (keyword or embeddings) |
| **Context builder** | Token-budgeted prompt string: profile + procedures + memories + patterns |
| **Thread summarization** | pi-style compaction of long conversation threads |
| **Consolidation** | Compress episodes → stable facts |
| **Forgetting** | Importance-weighted decay (trivial memories die, critical ones survive) |
| **Editing & correction** | In-place episode edits + timestamped annotation audit trail |
| **Multi-user isolation** | `owner_id` scoping — one DB, many users, no cross-contamination |
| **Embeddings** *(optional)* | SQLite-backed semantic retrieval via `sentence-transformers` |
| **Zero dependencies** | Core runs on Python stdlib only (SQLite) |
| **Single-file storage** | One `.db` file = entire memory. Copy it, back it up, version it. |

## Install

```bash
# From source (development)
git clone <repo-url> && cd remembox
uv sync --extra dev

# Coming soon: pip install remembox
```

## Quick Start
> Want to *run* these examples in a browser? Open [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) — a 5-minute executable version of everything below. For the full simple→advanced tour, see [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb).

### 1. Record Events

```python
from remembox import Remembox

memory = Remembox("jarvis.db")

# Record what happens
memory.record("User got promoted to Director of Engineering!", importance=1.0, emotion="ecstatic")
memory.record("User ordered black coffee", importance=0.3)
memory.record("User's dog Rocky had a vet visit. All clear.", importance=0.5, emotion="relieved")
```

### 2. Learn Facts

```python
# Semantic facts with automatic conflict resolution
memory.learn("user", "name", "Pranav", confidence=0.95)
memory.learn("user", "prefers", "black coffee", confidence=0.9)
memory.learn("user", "lives_in", "Delhi", confidence=0.8)

# Repeat a fact → confidence increases (reinforcement)
memory.learn("user", "prefers", "black coffee")  # → confidence: 0.9 → 0.915

# New info → old fact deactivated (contradiction)
memory.learn("user", "lives_in", "Mumbai")  # Delhi deactivated, Mumbai active
```

### 3. Retrieve Memories

```python
# Keyword + recency + importance scoring
results = memory.recall("coffee", k=3)
for r in results:
    print(f"Score: {r.score:.3f} | {r.episode.content}")
    # Score breakdown: R=recency, V=relevance, I=importance
    print(f"  R={r.recency:.2f} V={r.relevance:.2f} I={r.importance:.2f}")
```

### 4. Build LLM Context

```python
# Generate a prompt-ready context string
context = memory.context("What should I get for dinner?", max_tokens=2000)
print(context)
# Output:
# ## User Profile
# - user name Pranav (95%)
# - user prefers black coffee (92%)
# - user lives_in Mumbai (80%)
#
# ## Relevant Memories
# - (2h ago) User ordered black coffee
# - (1d ago [ecstatic]) User got promoted to Director of Engineering!
# (when procedures or reflections exist, `## Active Procedures` and `## Patterns`
#  sections are added too)

# Plug into any LLM:
messages = [
    {"role": "system", "content": f"You are a helpful assistant.\n\n{context}"},
    {"role": "user", "content": "What should I get for dinner?"},
]
# → Send to OpenAI, Anthropic, Ollama, etc.
```

### 5. Maintenance

```python
# Compress old episodes into semantic facts
memory.consolidate()        # one batch; use consolidate_all() to drain the backlog

# Synthesize higher-order patterns across episodes
memory.reflect()

# Prune stale memories (importance-weighted; critical ones survive)
result = memory.forget()
print(f"Deleted: {result['deleted']}, Archived: {result['archived']}")

# …or run all of the above in one call:
memory.maintain()

# Health check
print(memory.stats())
```

## Configuration

```python
from remembox import Remembox, MemoryConfig

# Custom config
config = MemoryConfig(
    decay_rate=0.05,          # How fast memories fade (higher = faster)
    w_recency=0.3,            # Retrieval weight: recency
    w_relevance=0.4,          # Retrieval weight: keyword match
    w_importance=0.3,         # Retrieval weight: stored importance
    max_context_tokens=2000,  # Token budget for context()
)
memory = Remembox("agent.db", config=config)

# Or use presets:
fast_memory = Remembox("chatbot.db", config=MemoryConfig.fast())    # Aggressive forgetting
deep_memory = Remembox("assistant.db", config=MemoryConfig.deep())  # Long retention
```

## Integration Examples

### With OpenAI

```python
from openai import OpenAI
from remembox import Remembox

client = OpenAI()
memory = Remembox("openai_agent.db")

def chat(user_message: str) -> str:
    # Store the user message
    memory.record(f"User: {user_message}", importance=0.5)

    # Build context-enhanced prompt
    context = memory.context(user_message)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant.\n\n{context}"},
            {"role": "user", "content": user_message},
        ],
    )
    reply = response.choices[0].message.content

    # Store the response
    memory.record(f"Assistant: {reply}", importance=0.2, source="response")
    return reply
```

### With Anthropic

```python
import anthropic
from remembox import Remembox

client = anthropic.Anthropic()
memory = Remembox("claude_agent.db")

def chat(user_message: str) -> str:
    memory.record(f"User: {user_message}")
    context = memory.context(user_message)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system=f"You are a helpful assistant.\n\n{context}",
        messages=[{"role": "user", "content": user_message}],
        max_tokens=1024,
    )
    reply = response.content[0].text
    memory.record(f"Assistant: {reply}", importance=0.2, source="response")
    return reply
```

### With Any Agent Framework

```python
from remembox import Remembox

memory = Remembox("agent.db")

# In your agent's observe/act loop:
def agent_step(observation: str) -> str:
    # 1. Store observation
    memory.record(observation, importance=score_importance(observation))

    # 2. Retrieve relevant context
    context = memory.context(observation)

    # 3. Generate action (your logic here)
    action = your_llm_or_rules(observation, context)

    # 4. Periodic maintenance
    if should_consolidate():
        memory.consolidate()
        memory.forget()

    return action
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Remembox                            │
│                                                              │
│  record()  recall()  learn()  context()  consolidate()      │
│     │         │        │         │            │              │
│     ▼         ▼        ▼         ▼            ▼              │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │Episodic│ │Retriev.│ │Semantic│ │Context │ │Consol. │   │
│  │ Store  │ │ Engine │ │ Store  │ │Builder │ │Pipeline│   │
│  └───┬────┘ └────────┘ └───┬────┘ └────────┘ └────────┘   │
│      │                      │                               │
│      └──────────┬───────────┘                               │
│                 ▼                                            │
│           ┌──────────┐                                      │
│           │  SQLite   │  ← Single .db file                  │
│           │  (WAL)    │                                      │
│           └──────────┘                                      │
└─────────────────────────────────────────────────────────────┘
```

## API Reference

Grouped by memory type. All methods are on `Remembox`.

| Category | Method | Returns |
|---|---|---|
| Episodic | `record(content, importance, emotion, source, context, thread_id, parent_id, depth)` | `Episode` |
| Episodic | `recall(query, k, min_score)` | `list[RetrievalResult]` |
| Episodic | `recent(n)` | `list[Episode]` |
| Episodic | `search(keyword, limit)` | `list[Episode]` |
| Threads | `thread(thread_id, limit)` | `list[Episode]` |
| Threads | `thread_children(episode_id, limit)` | `list[Episode]` |
| Threads | `threads(limit)` | `list[str]` |
| Threads | `summarize_thread(thread_id)` | `ThreadSummaryResult` |
| Semantic | `learn(subject, predicate, obj, confidence, valid_from, valid_until, recurrence)` | `(Fact, action)` |
| Semantic | `about(subject, at_time)` | `list[Fact]` |
| Semantic | `find_fact(subject, predicate, at_time)` | `list[Fact]` |
| Procedural | `learn_procedure(trigger, action, confidence)` | `Procedure` |
| Procedural | `match_procedures(text)` | `list[Procedure]` |
| Procedural | `procedures()` / `delete_procedure(id)` | `list[Procedure]` / `bool` |
| Context | `context(query, max_tokens, min_score, profile_subject)` | `str` |
| Maintenance | `consolidate()` / `consolidate_all()` | `dict` |
| Maintenance | `reflect(episodes)` / `reflections(subject)` | `dict` / `list` |
| Maintenance | `forget()` | `dict` |
| Maintenance | `maintain()` — runs consolidate → reflect → summarize → forget | `dict` |
| Editing | `update_episode(id, ...)` / `annotate_episode(id, ...)` | `Episode` |
| Editing | `edit_fact(id, ...)` / `correct_fact(id, ...)` | `Fact` / `(Fact, action)` |
| Lifecycle | `stats()` / `close()` | `dict` / `None` |

`learn()`'s `action` is `'new'` / `'reinforced'` / `'contradicted'`. Retrieval scores combine as `w_recency·R + w_relevance·V + w_importance·I` (see [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) §5).

## Design Decisions

| Decision | Why |
|---|---|
| **SQLite, not Postgres** | Zero config, ~0.05ms reads, single-file backup. Handles 18M rows (10yr heavy use). |
| **Stdlib only** | No supply-chain risk. Embedding search is an optional extra. |
| **Sync API** | Lowest latency path. SQLite calls are <1ms — async overhead isn't worth it. |
| **Store protocols** | `EpisodicStoreProtocol` / `SemanticStoreProtocol` — swap in Postgres/Redis by implementing the protocol. |
| **Tiered forgetting** | Critical memories (importance ≥ 0.9) never die. Trivial ones fade in days. |

## Development

```bash
# Install dev deps
uv sync --extra dev

# Run tests (262 passed, 19 skipped, ~1.5s)
uv run pytest

# Run with verbose output
uv run pytest -v

# Run scale tests only
uv run pytest tests/test_episodic.py::TestEpisodicScale -v
```

## Documentation

| Doc | What it's for |
|---|---|
| **README.md** *(this file)* | Front door: features, install, quick start, API, design |
| [`notebooks/`](notebooks) | Runnable examples — `quickstart.ipynb` (5-min) and `walkthrough.ipynb` (simple→advanced) |
| [`CHANGELOG.md`](CHANGELOG.md) | History of changes, release by release |
| [`BUGS.md`](BUGS.md) | Known-issues tracker — audit with severity, status, and fix notes |
| [`lessons/`](lessons) | 8 incremental teaching scripts that build the library from scratch |
| [`demos/`](demos) | `demo_openrouter.py` — live end-to-end run with a real LLM |
| [`integrations/pi/`](integrations/pi) | Long-term memory for [pi](https://github.com/earendil-works/pi-mono): HTTP sidecar + TypeScript extension |

## License

MIT
