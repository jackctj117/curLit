"""Strict, shared no-order shadow generation budget (CL-cnwz).

Counts CLI generation invocations, not internal Claude turns or dollar usage.
The existing strict transport owns authentication and included-usage checks.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from src.events._util import ThreadLocalClient
from src.events.niche_shadow import ModelReply
from src.research.llm import Message

# Same explicitly approved model as the deployed subscription-only discovery.
SUBSCRIPTION_MODEL = "claude-sonnet-4-6"
# Preserve the deployed two-discovery-plus-critic workload unless the operator
# explicitly declares a different bounded shadow experiment allocation.
DEFAULT_EVENT_CALL_LIMIT = 3
logger = logging.getLogger(__name__)


class ShadowBudgetError(RuntimeError):
    """No further model generation is permitted for this event."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class EventCallBudget:
    def __init__(self, limit: int) -> None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("event call limit must be a positive integer")
        self.limit = limit
        self._lock = threading.Lock()
        self._calls: dict[str, int] = {}
        self._reason = ""

    def reserve(self, role: str) -> None:
        with self._lock:
            if self._reason:
                raise ShadowBudgetError(self._reason)
            if sum(self._calls.values()) >= self.limit:
                self._reason = "event_model_call_limit"
                logger.info("Shadow event generation blocked: shared call limit=%d", self.limit)
                raise ShadowBudgetError(self._reason)
            self._calls[role] = self._calls.get(role, 0) + 1
            logger.info(
                "Reserving subscription shadow call role=%s used=%d limit=%d",
                role,
                sum(self._calls.values()),
                self.limit,
            )

    def stop(self, reason: str) -> None:
        with self._lock:
            if not self._reason:
                self._reason = reason
                logger.info("Stopping further shadow event generations reason=%s", reason)

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "unit": "cli_generation_invocations",
                "limit": self.limit,
                "calls": sum(self._calls.values()),
                "calls_by_role": dict(self._calls),
                "stop_reason": self._reason,
                "usd_cost": None,
                "cost_provenance": "subscription_unmetered",
                "output_token_cap_enforced": False,
            }


class SubscriptionBudgetClient:
    """Force strict transport kwargs on every role, including parallel edges."""

    def __init__(self, client: Any, budget: EventCallBudget, role: str) -> None:
        self.client, self.budget, self.role = client, budget, role

    def complete(self, **kwargs: Any) -> Any:
        if kwargs.get("model") != SUBSCRIPTION_MODEL:
            self.budget.stop("subscription_model_not_approved")
            raise ShadowBudgetError("subscription_model_not_approved")
        self.budget.reserve(self.role)
        kwargs.update(subscription_only=True, no_tools=True)
        try:
            client = (
                self.client.get() if isinstance(self.client, ThreadLocalClient) else self.client
            )
            reply = client.complete(**kwargs)
            if getattr(reply, "model", None) != SUBSCRIPTION_MODEL:
                raise ShadowBudgetError("subscription_model_substitution")
            return reply
        except Exception as exc:
            self.budget.stop(str(getattr(exc, "reason", type(exc).__name__)))
            raise


class SubscriptionResearchModel:
    """Equivalent-tool baseline through the same subscription-only transport."""

    provider = "claude-code"
    model = SUBSCRIPTION_MODEL

    def __init__(self, client: Any, budget: EventCallBudget | None = None) -> None:
        self.client = client
        self.budget = budget

    def bound(self, budget: EventCallBudget) -> SubscriptionResearchModel:
        return SubscriptionResearchModel(self.client, budget)

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply:
        if self.budget is None:
            raise ShadowBudgetError("subscription_baseline_budget_missing")
        response = SubscriptionBudgetClient(self.client, self.budget, "baseline").complete(
            messages=[Message(**message) for message in messages],
            model=self.model,
            max_tokens=max_tokens,
        )
        if type(response.input_tokens) is not int or type(response.output_tokens) is not int:
            self.budget.stop("subscription_usage_unavailable")
            raise ShadowBudgetError("subscription_usage_unavailable")
        return ModelReply(
            response.text, response.model, response.input_tokens, response.output_tokens
        )
