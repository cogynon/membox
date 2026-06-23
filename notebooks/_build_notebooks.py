"""Helper to build the two example notebooks from readable cell lists.

Run:  uv run python notebooks/_build_notebooks.py
"""
from __future__ import annotations
import json


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True) or [""]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True) or [""],
    }


def build(cells: list[dict], path: str) -> None:
    nbobj = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w") as f:
        json.dump(nbobj, f, indent=1)
    print("wrote", path, "(", len(cells), "cells )")


# ---------------------------------------------------------------------------
# QUICKSTART
# ---------------------------------------------------------------------------
quickstart = [
    md("""# membox — Quickstart

**Production-grade plug-and-play memory for AI agents.**
Zero config, single-file storage, framework-agnostic.

This notebook gets you from `pip install` to a memory-augmented prompt in **5 minutes**.

> Run cells top to bottom. Everything uses an in-memory database (`":memory:"`) so no files are created on disk.
"""),
    md("""## 0. Install & import

Install from source (the package lives in `src/`):

```bash
uv sync --extra dev
```
"""),
    code("""# Make the local package importable when running this notebook from the repo root.
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from membox import Membox, MemoryConfig
"""),
    md("""## 1. Create a memory

One class, one import. Backed by SQLite (WAL mode). State persists across restarts when you point it at a real file path.
"""),
    code("""memory = Membox(":memory:")  # use "my_agent.db" to persist to disk
memory"""),
    md("""## 2. Record events (episodic memory)

`record()` stores a timestamped event with an **importance** score (0.0 trivial → 1.0 life-changing) and an optional emotion tag.
"""),
    code("""memory.record("User got promoted to Director of Engineering!", importance=1.0, emotion="ecstatic")
memory.record("User ordered black coffee", importance=0.3)
memory.record("User's dog Rocky had a vet visit. All clear.", importance=0.5, emotion="relieved")

# Peek at the most recent events
for ep in memory.recent(3):
    print(f"[{ep.emotion or '-':9}] imp={ep.importance:.1f} | {ep.content}")
"""),
    md("""## 3. Learn facts (semantic memory)

`learn()` stores a `(subject, predicate, object)` triple with automatic **reinforcement** (repeat a fact → confidence rises) and **contradiction** handling (new value → old value deactivated).
"""),
    code("""memory.learn("user", "name",      "Pranav",      confidence=0.95)
memory.learn("user", "prefers",   "black coffee", confidence=0.9)

# Reinforcement: same fact again → confidence climbs toward 1.0
fact, action = memory.learn("user", "prefers", "black coffee")
print(f"action={action}, new confidence={fact.confidence:.3f}")

# Contradiction: new value → old fact deactivated, new one becomes active
memory.learn("user", "lives_in", "Delhi", confidence=0.8)
memory.learn("user", "lives_in", "Mumbai")
print("lives_in facts:", [(f.object, f.is_active) for f in memory.about("user") if f.predicate == "lives_in"])
"""),
    md("""## 4. Retrieve memories (smart recall)

`recall()` scores episodes by **Recency × Relevance × Importance** and returns the top-k with a score breakdown.
"""),
    code("""results = memory.recall("coffee", k=3)
for r in results:
    print(f"score={r.score:.3f}  R={r.recency:.2f} V={r.relevance:.2f} I={r.importance:.2f}")
    print(f"   → {r.episode.content}")
"""),
    md("""## 5. Build LLM context

`context()` returns a **prompt-ready string** — user profile + relevant memories — that fits inside a token budget. Paste it straight into your system prompt.
"""),
    code("""context = memory.context("What does the user like?", max_tokens=2000)
print(context)
"""),
    code("""# Plug into any LLM (OpenAI shown; works with Anthropic, Ollama, OpenRouter, ...)
messages = [
    {"role": "system", "content": f"You are a helpful assistant.\\n\\n{context}"},
    {"role": "user",   "content": "What should I get for dinner?"},
]
messages  # send to: client.chat.completions.create(model=..., messages=messages)
"""),
    md("""## 6. Maintenance

Two background jobs keep memory healthy over time:

- `consolidate()` — compress old episodes into durable facts
- `forget()` — prune/archive stale memories (importance-weighted; critical ones survive)
"""),
    code("""print("consolidate:", memory.consolidate())
print("forget:      ", memory.forget())
print("stats:       ", memory.stats()["episodes"])
"""),
    md("""## ✅ You're done

You now know the full loop:

```
record → learn → recall → context → [LLM] → record → ... → consolidate / forget
```

**Next:**
- `notebooks/walkthrough.ipynb` — a detailed simple→advanced tour (procedural memory, threads, reflection, temporal facts, embeddings, multi-user, LLM integration).
- `README.md` — full API reference & design notes.
- `demos/demo_openrouter.py` — a live end-to-end demo with a real LLM.
"""),
]


# ---------------------------------------------------------------------------
# WALKTHROUGH
# ---------------------------------------------------------------------------
walkthrough = [
    md("""# membox — Detailed Walkthrough (Simple → Advanced)

A progressive tour of the whole library. Each section builds on the previous one.

| # | Section | Concept |
|---|---------|---------|
| 1 | Episodic basics | `record`, `recent`, `search` |
| 2 | Importance & emotion | manual + auto scoring |
| 3 | Semantic facts | `learn`, reinforce, contradict |
| 4 | Temporal facts | `valid_from`/`valid_until`, recurrence |
| 5 | Retrieval scoring | R × V × I, `min_score`, decay |
| 6 | Context builder | sections, token budget |
| 7 | Procedural memory | triggers → actions |
| 8 | Conversation threads | `thread_id`, `summarize_thread` |
| 9 | Consolidation | episodes → facts |
| 10 | Reflection | higher-order patterns |
| 11 | Forgetting | tiered decay |
| 12 | `maintain()` | one-call housekeeping |
| 13 | Configuration & presets | `fast()` / `deep()` / custom |
| 14 | Editing & annotations | correct the record |
| 15 | Multi-user | `owner_id` isolation |
| 16 | Embeddings (optional) | semantic retrieval |
| 17 | LLM integration | importance scorer + chat loop |

> Uses in-memory DBs so nothing is written to disk. Swap `":memory:"` for a path to persist.
"""),
    code("""import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from datetime import datetime, timedelta
from membox import Membox, MemoryConfig
"""),

    # 1. Episodic basics ---------------------------------------------------------
    md("""## 1. Episodic memory — *what happened?*

Episodes are timestamped events with an importance score and optional emotion/source/metadata. This is the raw log of your agent's life.
"""),
    code("""m = Membox(":memory:")

m.record("User said they love hiking in the Himalayas", importance=0.8, source="conversation")
m.record("User ordered black coffee", importance=0.3, emotion="calm", source="order")
m.record("Rocky the dog had a vet checkup — all clear", importance=0.5, context={"pet": "Rocky"})

# Three ways to browse episodes
print("recent:", [e.content for e in m.recent(3)])
print("search:", [e.content for e in m.search("coffee")])
print("fields:", m.recent(1)[0].importance, m.recent(1)[0].context)
"""),

    # 2. Importance & emotion ----------------------------------------------------
    md("""## 2. Importance & emotion

Importance (0.0–1.0) drives two things: **retrieval ranking** and **forgetting** (critical memories never die). You can set it manually or let a scorer infer it.

| Mode | How |
|------|-----|
| Manual | pass `importance=` to `record()` |
| Rule-based auto | `MemoryConfig(auto_score_importance=True)` |
| LLM auto | inject an `LLMImportanceScorer` (see §17) |
"""),
    code("""from membox.importance import RuleBasedImportanceScorer

auto = Membox(":memory:", config=MemoryConfig(auto_score_importance=True))
ep = auto.record("I just got promoted to Director of Engineering!")
print(f"auto-scored  -> importance={ep.importance:.2f} emotion={ep.emotion}")

ep2 = auto.record("hey, what's up?")
print(f"small talk   -> importance={ep2.importance:.2f} emotion={ep2.emotion}")
"""),

    # 3. Semantic facts ----------------------------------------------------------
    md("""## 3. Semantic facts — *what I know*

Facts are `(subject, predicate, object)` triples with a confidence score. `learn()` handles three cases automatically:

| Action | Trigger | Effect |
|--------|---------|--------|
| `new` | First time this triple is seen | Inserts with given confidence |
| `reinforced` | Same triple seen again | Confidence rises toward 1.0 |
| `contradicted` | Same subject+predicate, new object | Old fact deactivated, new one active |
"""),
    code("""s = Membox(":memory:")

print(s.learn("user", "name",    "Pranav",      confidence=0.95)[1])      # new
print(s.learn("user", "prefers", "black coffee", confidence=0.9)[1])     # new
print(s.learn("user", "prefers", "black coffee")[1])                     # reinforced
print(s.learn("user", "lives_in","Delhi", confidence=0.8)[1])            # new
print(s.learn("user", "lives_in","Mumbai")[1])                           # contradicted

print("\\nActive facts about user:")
for f in s.about("user"):
    print(f"  {f.subject} {f.predicate} {f.object}  (conf={f.confidence:.2f}, active={f.is_active})")
"""),

    # 4. Temporal facts ----------------------------------------------------------
    md("""## 4. Temporal facts

Facts can have a **validity window** (`valid_from` / `valid_until`) and a **recurrence** pattern. `about(subject, at_time=...)` returns only facts that were true at that moment.
"""),
    code("""t = Membox(":memory:")

t.learn("user", "travels_to", "Tokyo",
        valid_from=datetime(2026, 7, 1), valid_until=datetime(2026, 7, 10),
        confidence=0.9)

before = datetime(2026, 6, 15)
during = datetime(2026, 7, 5)
after  = datetime(2026, 7, 20)

print("before trip:", [f.object for f in t.about("user", at_time=before)])
print("during trip:", [f.object for f in t.about("user", at_time=during)])
print("after  trip:", [f.object for f in t.about("user", at_time=after)])
"""),

    # 5. Retrieval scoring -------------------------------------------------------
    md("""## 5. Retrieval scoring

`recall()` combines three signals:

```
score = w_recency * R   +   w_relevance * V   +   w_importance * I
```

- **R (recency)** — `e^(-decay_rate * hours_ago)`, decays with time
- **V (relevance)** — keyword overlap with the query (or embedding sim, §16)
- **I (importance)** — the stored importance of the episode

Each `RetrievalResult` exposes the breakdown so you can debug rankings.
"""),
    code("""r = Membox(":memory:", config=MemoryConfig(w_recency=0.3, w_relevance=0.4, w_importance=0.3))

# Record at different importances
r.record("User loves hiking in the Himalayas", importance=0.9)
r.record("User mentioned a hiking trail once", importance=0.2)

for res in r.recall("hiking", k=3):
    print(f"score={res.score:.3f}  R={res.recency:.2f} V={res.relevance:.2f} I={res.importance:.2f}")
    print(f"   {res.episode.content}")

# min_score filters out weak noise
print("\\nWith min_score=0.6:",
      [res.episode.content for res in r.recall("hiking", k=5, min_score=0.6)])
"""),

    # 6. Context builder ---------------------------------------------------------
    md("""## 6. The context builder — your integration point

`context()` assembles a token-budgeted string with up to four sections:

1. **User Profile** — currently-valid facts
2. **Active Procedures** — matching if-then rules (§7)
3. **Relevant Memories** — top episodes for the query
4. **Patterns** — reflections (§10)

Paste the return value into your system prompt. That's the entire integration.
"""),
    code("""c = Membox(":memory:")
c.learn("user", "prefers", "black coffee", confidence=0.9)
c.learn("user", "name", "Pranav", confidence=0.95)
c.record("User got promoted to Director!", importance=1.0, emotion="ecstatic")
c.record("User ordered black coffee", importance=0.3)

print(c.context("what does the user like?", max_tokens=500))
"""),

    # 7. Procedural memory -------------------------------------------------------
    md("""## 7. Procedural memory — *how I do things*

Procedures are if-then rules: *when trigger matches, do action*. They show up in the context's **Active Procedures** section so the LLM can act on them.
"""),
    code("""p = Membox(":memory:")
p.learn_procedure("goodnight", "dim the lights and lock the door", confidence=0.9)
p.learn_procedure("meeting starts", "mute notifications and open the agenda", confidence=0.8)

print("matched:", [(proc.trigger, proc.action) for proc in p.match_procedures("time to say goodnight")])
print("\\nin context:")
print(p.context("goodnight"))
"""),

    # 8. Threads -----------------------------------------------------------------
    md("""## 8. Conversation threads

Episodes can be grouped into a `thread_id` and nested via `parent_id`/`depth`. Long threads get compressed with `summarize_thread()` (pi-style compaction): older messages fold into one summary episode, recent ones stay verbatim.
"""),
    code("""th = Membox(":memory:")

# A short thread
th.record("Hi, I need help with a Python bug", thread_id="t1")
th.record("Sure, what's the error?", thread_id="t1")
th.record("It's a KeyError on 'user'", thread_id="t1")
print("thread episodes:", len(th.thread("t1")))

# Force a long thread so summarization kicks in
for i in range(6):
    th.record(" ".join([f"message-{i}"] * 60), thread_id="t2")

result = th.summarize_thread("t2")
print("summarized?", result.did_summarize)
print("summary preview:", (result.summary_episode.content[:80] + "...") if result.summary_episode else None)
"""),

    # 9. Consolidation -----------------------------------------------------------
    md("""## 9. Consolidation — episodes → facts

`consolidate()` scans unconsolidated episodes and extracts stable facts (e.g. *"I love coffee"* → `user prefers coffee`). The rule-based extractor matches common first-person patterns; swap in an LLM consolidator for production accuracy.

Episodes must be older than `consolidation_min_age_hours` (default 1h) to be eligible.
"""),
    code("""con = Membox(":memory:")
con.record("I love black coffee", importance=0.7)
con.record("I live in Mumbai", importance=0.6)
con.record("the weather is nice today", importance=0.2)  # no pattern → no fact

# Pretend time has passed so episodes are eligible
future = datetime.now() + timedelta(hours=2)
report = con.consolidate(now=future)
print("report:", {k: v for k, v in report.items() if k != "facts"})
print("facts learned:")
for f in con.about("user"):
    print(f"  {f.subject} {f.predicate} {f.object}  (conf={f.confidence:.2f})")
"""),

    # 10. Reflection -------------------------------------------------------------
    md("""## 10. Reflection — higher-order patterns

`reflect()` looks across many episodes and surfaces recurring patterns (e.g. *"user frequently mentions coffee"*). These feed the **Patterns** section of the context. Run it explicitly or enable `auto_reflect` in config.
"""),
    code("""ref = Membox(":memory:")
for _ in range(4):
    ref.record("User is stressed about the deadline again")

result = ref.reflect(episodes=ref.recent(10))   # explicit pass
print("evaluated:", result["evaluated"])
for r in result["reflections"]:
    print(f"  {r.subject} {r.predicate} {r.object}  (conf={r.confidence:.2f})")
"""),

    # 11. Forgetting -------------------------------------------------------------
    md("""## 11. Forgetting — tiered decay

`forget()` walks every episode and applies importance-tiered rules. Trivial memories fade in days; critical ones (importance ≥ 0.9) effectively never die. Actions are `delete`, `archive`, or `keep`.
"""),
    code("""fg = Membox(":memory:")

# A trivial old memory vs. a critical old memory
fg.record("User said 'lol'", importance=0.1, timestamp=datetime.now() - timedelta(days=30))
fg.record("User's mother passed away", importance=1.0, timestamp=datetime.now() - timedelta(days=30))

result = fg.forget(now=datetime.now())
print("summary:", {k: v for k, v in result.items() if k != "actions"})
for a in result["actions"]:
    print(f"  {a.action:8} imp={a.retention_score:.2f}  {a.reason}")
"""),

    # 12. maintain() -------------------------------------------------------------
    md("""## 12. `maintain()` — one-call housekeeping

Runs consolidation → reflection → thread summarization → forgetting in dependency order. Call it periodically (e.g. after every N turns) so you don't have to orchestrate each step.
"""),
    code("""mt = Membox(":memory:")
for _ in range(3):
    mt.record("I love black coffee")
for i in range(6):
    mt.record(" ".join([f"chat-{i}"] * 60), thread_id="t1")

report = mt.maintain(now=datetime.now() + timedelta(hours=2))
print("keys:", list(report.keys()))
print("consolidate episodes_processed:", report["consolidate"]["episodes_processed"])
print("forget summary:", {k: v for k, v in report["forget"].items() if k != "actions"})
"""),

    # 13. Configuration ----------------------------------------------------------
    md("""## 13. Configuration & presets

Every knob lives in `MemoryConfig`. Override what you need, or use a preset:

| Preset | Use case | Behavior |
|--------|----------|----------|
| `MemoryConfig()` (default) | General | balanced |
| `MemoryConfig.fast()` | Chatbots | aggressive forgetting, small context |
| `MemoryConfig.deep()` | Personal assistants | long retention, embeddings on |
"""),
    code("""fast = Membox(":memory:", config=MemoryConfig.fast())
deep = Membox(":memory:", config=MemoryConfig.deep())

# Custom: emphasise relevance, forget slowly
custom = MemoryConfig(decay_rate=0.005, w_relevance=0.6, w_recency=0.2, w_importance=0.2,
                      max_context_tokens=4000)
cust = Membox(":memory:", config=custom)
print("fast decay:", fast._config.decay_rate)
print("deep decay:", deep._config.decay_rate)
print("custom weights: R V I =", cust._config.w_recency, cust._config.w_relevance, cust._config.w_importance)
"""),

    # 14. Editing & annotations --------------------------------------------------
    md("""## 14. Editing & annotations — correct the record

Mistakes happen. You can:
- `update_episode()` — edit an episode in place
- `annotate_episode()` — append a timestamped correction/note (audit trail, original intact)
- `edit_fact()` / `correct_fact()` — fix or supersede a fact
"""),
    code("""ed = Membox(":memory:")
ep = ed.record("User lives in Paris")

updated = ed.update_episode(ep.id, content="User lives in Lyon", importance=0.8)
print("after update:", ed.recent(1)[0].content)

annotated = ed.annotate_episode(ep.id, correction="actually it's Lyon, not Paris")
print("annotations:", annotated.context.get("__annotations__"))

# Facts
f, _ = ed.learn("user", "job", "engineer", confidence=0.7)
fixed, action = ed.correct_fact(f.id, new_object="staff engineer")
print(f"correct_fact action={action}: {fixed.subject} {fixed.predicate} {fixed.object}")
"""),

    # 15. Multi-user -------------------------------------------------------------
    md("""## 15. Multi-user isolation

Pass `owner_id` to partition memory per user/agent. Each owner only sees its own episodes, facts, procedures, and reflections in the same database file.
"""),
    code("""import tempfile, os
db = os.path.join(tempfile.mkdtemp(), "multi.db")

alice = Membox(db, owner_id="alice")
bob   = Membox(db, owner_id="bob")

alice.record("Alice loves rock climbing", importance=0.8)
bob.record("Bob prefers baking", importance=0.6)

# Alice's recall sees only her memories
print("alice sees:", [r.episode.content for r in alice.recall("hobbies", k=5)])
print("bob   sees:", [r.episode.content for r in bob.recall("hobbies", k=5)])
"""),

    # 16. Embeddings -------------------------------------------------------------
    md("""## 16. (Optional) Embeddings — true semantic retrieval

Keyword overlap misses synonyms ("hiking" vs "outdoor activities"). Enable an embedding model for semantic similarity:

```bash
uv sync --extra embeddings
```

Set `embedding_model_name` in config and `Membox` will persist embeddings in SQLite and use them in `recall()` / `context()`. Hybrid scoring blends embeddings with keyword overlap via `w_embedding` / `w_keyword`.
"""),
    code("""try:
    cfg = MemoryConfig(embedding_model_name="all-MiniLM-L6-v2")
    em = Membox(":memory:", config=cfg)
    em.record("I enjoy long walks in the mountains", importance=0.7)
    em.record("I love debugging Python code", importance=0.5)

    # Synonym query: keyword overlap is low, embeddings rescue it
    for r in em.recall("outdoor activities", k=2):
        print(f"score={r.score:.3f} | {r.episode.content}")
except ImportError as e:
    print("Embeddings extra not installed. Install with: uv sync --extra embeddings")
    print("(", str(e)[:80], "...)")
"""),

    # 17. LLM integration --------------------------------------------------------
    md("""## 17. LLM integration — the full loop

Two extension points connect `membox` to a real LLM:

1. **`LLMImportanceScorer`** — infer importance + emotion on every `record()` (works with any OpenAI-compatible client).
2. **The chat loop** — `record` → `context` → `LLM` → `record`. Framework-agnostic.

Below is a self-contained example using the rule-based scorer (no API key needed). Swap in `LLMImportanceScorer(client, model=...)` and a real client to go live.
"""),
    code("""from membox.importance import RuleBasedImportanceScorer

# 1) Auto importance via the rule-based scorer (no API key)
scorer = RuleBasedImportanceScorer()
agent = Membox(":memory:", config=MemoryConfig(auto_score_importance=True))

# Pre-seed some preferences
agent.learn("user", "name", "Pranav", confidence=0.95)
agent.learn("user", "prefers", "black coffee", confidence=0.9)

# 2) The chat loop (mocked LLM; replace `my_llm` with a real client call)
def my_llm(system_prompt: str, user_message: str) -> str:
    # In production: client.chat.completions.create(model=..., messages=[...])
    return f"[mocked reply using context of {len(system_prompt)} chars]"

def chat(user_message: str) -> str:
    ep = agent.record(user_message)                       # auto-scored
    context = agent.context(user_message)
    system_prompt = f"You are a helpful assistant.\\n\\n{context}"
    reply = my_llm(system_prompt, user_message)
    agent.record(f"Assistant: {reply}", importance=0.2, source="response")
    return reply

print("turn 1:", chat("I just got promoted to Director!")[:60], "...")
print("turn 2:", chat("What do you know about me?")[:60], "...")
print("\\nfinal memory:\\n", agent.context("who is the user?"))
"""),
    md("""## 📚 Cheatsheet

| Goal | Call |
|------|------|
| Store an event | `memory.record(content, importance=, emotion=, source=, thread_id=)` |
| Browse recent events | `memory.recent(n)`, `memory.search(keyword)` |
| Store a fact | `memory.learn(subject, predicate, obj, confidence=)` |
| Query facts | `memory.about(subject, at_time=)`, `memory.find_fact(subject, predicate=)` |
| Retrieve memories | `memory.recall(query, k=, min_score=)` |
| Build prompt context | `memory.context(query, max_tokens=)` |
| Store a routine | `memory.learn_procedure(trigger, action, confidence=)` |
| Compress episodes → facts | `memory.consolidate()` / `consolidate_all()` |
| Compress a thread | `memory.summarize_thread(thread_id)` |
| Surface patterns | `memory.reflect()` |
| Prune stale memories | `memory.forget()` |
| Run all maintenance | `memory.maintain()` |
| Edit an episode | `memory.update_episode(id, ...)`, `annotate_episode(id, ...)` |
| Fix a fact | `memory.edit_fact(id, ...)`, `correct_fact(id, ...)` |
| Health check | `memory.stats()` |
| Per-user isolation | `Membox(db, owner_id="alice")` |
| Semantic retrieval | `MemoryConfig(embedding_model_name="all-MiniLM-L6-v2")` |
| Auto importance | `MemoryConfig(auto_score_importance=True)` or `LLMImportanceScorer` |

**Further reading:** `README.md`, `demos/demo_openrouter.py`, `lessons/` (builds the library from scratch), `tests/` (executable spec).
"""),
]


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    build(quickstart, os.path.join(here, "quickstart.ipynb"))
    build(walkthrough, os.path.join(here, "walkthrough.ipynb"))
