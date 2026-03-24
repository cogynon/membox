# agentmemory

**Production-grade plug-and-play memory for AI agents.**

Give any LLM, agent, or rule-based system persistent episodic + semantic memory. Zero config, single-file storage, framework-agnostic.

```python
from agentmemory import AgentMemory

memory = AgentMemory("my_agent.db")
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
| **Episodic memory** | Timestamped events with importance scoring |
| **Semantic memory** | Fact triples with reinforcement & contradiction |
| **Smart retrieval** | Recency × relevance × importance scoring |
| **Context builder** | Token-budgeted prompt-ready strings |
| **Consolidation** | Compress episodes → stable facts |
| **Forgetting** | Importance-weighted decay (trivial memories die, critical ones survive) |
| **Zero dependencies** | Core runs on Python stdlib only (SQLite) |
| **Single-file storage** | One `.db` file = entire memory. Copy it, back it up, version it. |

## Install

```bash
# From source (development)
git clone <repo-url> && cd agentmemory
uv sync --extra dev

# Coming soon: pip install agentmemory
```

## Quick Start

### 1. Record Events

```python
from agentmemory import AgentMemory

memory = AgentMemory("jarvis.db")

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
memory.consolidate()

# Prune stale memories (importance-weighted)
result = memory.forget()
print(f"Deleted: {result['deleted']}, Archived: {result['archived']}")

# Health check
print(memory.stats())
```

## Configuration

```python
from agentmemory import AgentMemory, MemoryConfig

# Custom config
config = MemoryConfig(
    decay_rate=0.05,          # How fast memories fade (higher = faster)
    w_recency=0.3,            # Retrieval weight: recency
    w_relevance=0.4,          # Retrieval weight: keyword match
    w_importance=0.3,         # Retrieval weight: stored importance
    max_context_tokens=2000,  # Token budget for context()
)
memory = AgentMemory("agent.db", config=config)

# Or use presets:
fast_memory = AgentMemory("chatbot.db", config=MemoryConfig.fast())    # Aggressive forgetting
deep_memory = AgentMemory("assistant.db", config=MemoryConfig.deep())  # Long retention
```

## Integration Examples

### With OpenAI

```python
from openai import OpenAI
from agentmemory import AgentMemory

client = OpenAI()
memory = AgentMemory("openai_agent.db")

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
from agentmemory import AgentMemory

client = anthropic.Anthropic()
memory = AgentMemory("claude_agent.db")

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
from agentmemory import AgentMemory

memory = AgentMemory("agent.db")

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
│                      AgentMemory                            │
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

| Method | Description | Returns |
|---|---|---|
| `record(content, importance, emotion, source)` | Store an episodic event | `Episode` |
| `recall(query, k)` | Retrieve top-k relevant memories | `list[RetrievalResult]` |
| `learn(subject, predicate, obj, confidence)` | Learn a semantic fact | `(Fact, action)` |
| `about(subject)` | Get all facts about a subject | `list[Fact]` |
| `context(query, max_tokens)` | Build prompt-ready context string | `str` |
| `consolidate()` | Compress episodes → facts | `dict` |
| `forget()` | Prune stale memories | `dict` |
| `stats()` | Health check | `dict` |
| `recent(n)` | Get N most recent episodes | `list[Episode]` |
| `search(keyword)` | Keyword search in episodes | `list[Episode]` |

## Design Decisions

| Decision | Why |
|---|---|
| **SQLite, not Postgres** | Zero config, ~0.05ms reads, single-file backup. Handles 18M rows (10yr heavy use). |
| **Stdlib only** | No supply-chain risk. Embedding search is an optional extra. |
| **Sync API** | Lowest latency path. SQLite calls are <1ms — async overhead isn't worth it. |
| **`BaseStore` protocol** | Anyone can swap in Postgres/Redis by implementing 5 methods. |
| **Tiered forgetting** | Critical memories (importance ≥ 0.9) never die. Trivial ones fade in days. |

## Development

```bash
# Install dev deps
uv sync --extra dev

# Run tests (129 tests, ~1s)
uv run pytest

# Run with verbose output
uv run pytest -v

# Run scale tests only
uv run pytest tests/test_episodic.py::TestEpisodicScale -v
```

## License

MIT
