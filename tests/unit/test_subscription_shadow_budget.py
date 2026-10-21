"""Hard subscription call policy: synthetic clients, no provider calls."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from src.events.subscription_shadow_budget import (
    SUBSCRIPTION_MODEL,
    EventCallBudget,
    ShadowBudgetError,
    SubscriptionBudgetClient,
)
from tests.unit._hop_graph_fixtures import reply


class Client:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return reply("{}", kwargs["model"])


def test_parallel_roles_share_one_hard_limit() -> None:
    budget = EventCallBudget(3)
    inner = Client()

    def invoke(index: int) -> bool:
        try:
            SubscriptionBudgetClient(inner, budget, f"role{index % 3}").complete(
                model=SUBSCRIPTION_MODEL, subscription_only=False, no_tools=False
            )
            return True
        except ShadowBudgetError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(invoke, range(40)))
    assert sum(results) == 3
    assert len(inner.calls) == 3
    assert all(c["subscription_only"] is True and c["no_tools"] is True for c in inner.calls)
    assert budget.report()["calls"] == 3
    assert budget.report()["stop_reason"] == "event_model_call_limit"
    assert budget.report()["usd_cost"] is None


def test_auth_or_quota_failure_seals_budget_for_other_roles() -> None:
    class Unavailable(Client):
        def complete(self, **kwargs: Any) -> Any:
            raise ShadowBudgetError("subscription_quota_exhausted")

    budget = EventCallBudget(12)
    with pytest.raises(ShadowBudgetError, match="subscription_quota_exhausted"):
        SubscriptionBudgetClient(Unavailable(), budget, "traversal").complete(
            model=SUBSCRIPTION_MODEL
        )
    inner = Client()
    with pytest.raises(ShadowBudgetError, match="subscription_quota_exhausted"):
        SubscriptionBudgetClient(inner, budget, "critic").complete(model=SUBSCRIPTION_MODEL)
    assert inner.calls == []
    assert budget.report()["calls"] == 1


def test_unapproved_model_never_calls_transport() -> None:
    inner = Client()
    budget = EventCallBudget(3)
    with pytest.raises(ShadowBudgetError, match="subscription_model_not_approved"):
        SubscriptionBudgetClient(inner, budget, "entailment").complete(model="unapproved")
    assert inner.calls == []
    assert budget.report()["calls"] == 0


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_budget_rejected(limit: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        EventCallBudget(limit)
