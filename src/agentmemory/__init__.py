"""agentmemory — Production-grade plug-and-play memory for AI agents."""

from agentmemory.models import Episode, Fact, RetrievalResult
from agentmemory.config import MemoryConfig
from agentmemory.memory import AgentMemory

__all__ = [
    "AgentMemory",
    "MemoryConfig",
    "Episode",
    "Fact",
    "RetrievalResult",
]

__version__ = "0.1.0"
