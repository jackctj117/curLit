"""Grok provider driver (CL-dpw6) — third LLM provider alongside Claude
+ DeepSeek.

xAI's Grok serves the OpenAI chat-completions API at
``https://api.x.ai/v1`` with bearer auth, so the driver is the shared
``_OpenAICompatDriver`` from ``src/research/llm/client.py`` with the
xAI base URL set. Models: ``grok-4``, ``grok-3``, ``grok-3-mini``;
pricing lives in ``client.py:_PRICING_USD_PER_MTOK`` (operator should
verify against the current xAI rate card before trusting the cost
column on production runs).

Why a separate provider when DeepSeek is also OpenAI-compatible:

  * **Viewpoint diversity.** Two LLMs from the same training family
    (e.g. Claude-Bull + Claude-Bear) tend to converge — the asymmetric
    prompts help but don't fully overcome shared priors. A different
    provider with an independently-trained model brings genuinely
    distinct failure modes to the debate. Switching one reviewer to
    Grok is a one-line YAML edit and produces a structurally more
    adversarial debate.
  * **Provider-redundancy** if one vendor degrades. The agent config
    layer already supports per-agent provider selection so we can roll
    individual reviewers between providers without code changes.

This module exists because the bead acceptance specifies the path
``src/research/llm/grok.py``. The class itself is just the registered
subclass; ``_DRIVERS`` in ``client.py`` imports from here so
``get_client("grok")`` works the same as the other providers.
"""

from __future__ import annotations

from src.research.llm.client import _OpenAICompatDriver, register_driver


class GrokDriver(_OpenAICompatDriver):
    """xAI Grok driver. Connects to ``api.x.ai/v1`` via the shared
    OpenAI-compatible base class."""

    name = "grok"
    base_url = "https://api.x.ai/v1"


# Self-register on import so ``get_client("grok")`` works without the
# caller having to remember to import this module.
register_driver("grok", GrokDriver)
