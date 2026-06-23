"""Shared tokenization helpers.

A single source of truth for turning text into word tokens, used by both
retrieval scoring and reflection keyword detection. Keeping one tokenizer
ensures "coffee" and "coffee." tokenize identically everywhere.
"""

from __future__ import annotations

import re

# Word tokens: lowercase alphanumeric runs. Strips punctuation so that
# "coffee" and "coffee." tokenize identically.
_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Lowercase and split into a *set* of alphanumeric word tokens.

    Using a regex instead of str.split() means trailing/leading punctuation no
    longer creates spurious distinct tokens ("coffee" vs "coffee.").
    """
    return set(_WORD_RE.findall(text.lower()))


def tokenize_list(text: str) -> list[str]:
    """Like :func:`tokenize` but preserves order and duplicates."""
    return _WORD_RE.findall(text.lower())


def escape_like(term: str) -> str:
    r"""Escape SQL LIKE wildcards so user input is matched literally.

    ``%`` and ``_`` are LIKE metacharacters; without escaping, a search for
    ``"%"`` matches everything and ``"_"`` matches any single character. Use
    together with ``ESCAPE '\'`` in the query.
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
