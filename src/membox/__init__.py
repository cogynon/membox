"""membox — Production-grade plug-and-play memory for AI agents."""

from membox.models import Episode, Fact, RetrievalResult
from membox.config import MemoryConfig
from membox.memory import Membox
from membox.summarization import (
    RuleBasedSummarizer,
    Summarizer,
    ThreadSummaryResult,
)

__all__ = [
    "Membox",
    "MemoryConfig",
    "Episode",
    "Fact",
    "RetrievalResult",
    "Summarizer",
    "RuleBasedSummarizer",
    "ThreadSummaryResult",
]

__version__ = "0.2.0"