"""Pluggable LLM client (Claude + DeepSeek + Grok + extensible). See
client.py for the abstraction; per-provider modules (e.g. grok.py)
self-register at import time."""

from src.research.llm.claude_code import (  # noqa: F401 — side-effect: register
    ClaudeCodeDriver,
)
from src.research.llm.client import LLMClient, LLMResponse, Message, get_client
from src.research.llm.grok import GrokDriver  # noqa: F401 — side-effect: register

__all__ = ["LLMClient", "LLMResponse", "Message", "get_client"]
