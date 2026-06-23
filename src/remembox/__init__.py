"""remembox — Production-grade plug-and-play memory for AI agents."""

from remembox.models import Episode, Fact, RetrievalResult
from remembox.config import MemoryConfig
from remembox.memory import Remembox
from remembox.summarization import (
    RuleBasedSummarizer,
    Summarizer,
    ThreadSummaryResult,
)

__all__ = [
    "Remembox",
    "MemoryConfig",
    "Episode",
    "Fact",
    "RetrievalResult",
    "Summarizer",
    "RuleBasedSummarizer",
    "ThreadSummaryResult",
]

__version__ = "0.2.0"