"""Research-pipeline agents (CL-h986).

Each agent is a thin wrapper around an ``LLMClient`` + a system prompt
loaded from disk. The base class handles prompt loading, knowledge-
retriever tool integration, and standardized invocation; concrete
agents (Bull, Bear, Idea, Implementer, Resolver) just specialize.
"""

from src.research.agents.base import Agent, AgentResponse

__all__ = ["Agent", "AgentResponse"]
