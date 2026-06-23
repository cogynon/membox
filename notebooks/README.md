# Notebooks

Interactive examples for `remembox`. Part of the [repo documentation](../README.md#documentation) set.

| Notebook | Audience | Length |
|----------|----------|--------|
| [`quickstart.ipynb`](quickstart.ipynb) | New users — get to a memory-augmented prompt in 5 minutes | ~8 cells |
| [`walkthrough.ipynb`](walkthrough.ipynb) | Detailed simple → advanced tour of the whole library | ~18 cells |

## Running

From the repo root:

```bash
uv sync --extra dev
uv run jupyter lab notebooks/
```

The first code cell of each notebook adds `src/` to `sys.path`, so they run from the
repo without an installed package. Every example uses an in-memory database
(`":memory:"`), so nothing is written to disk — swap in a file path to persist.

> `walkthrough.ipynb` §16 (embeddings) optionally needs `uv sync --extra embeddings`.

## Rebuilding

The notebooks are generated from `notebooks/_build_notebooks.py` (keeps the cell
sources readable as plain Python). Edit that file and re-run:

```bash
uv run python notebooks/_build_notebooks.py
```
