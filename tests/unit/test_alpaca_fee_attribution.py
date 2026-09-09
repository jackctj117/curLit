"""Known fee cashflows and independent identity oracles (CL-z97c)."""

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.execution.alpaca_fee_attribution import fee_summary, project_fees

EXID = "8efc7b9a-8b2b-4000-9955-d36e7db0df74"
FILL = {
    "id": "20190524113406977::" + EXID,
    "activity_type": "FILL",
    "transaction_time": "2019-05-24T15:34:06.977Z",
    "symbol": "ABC",
}


def fee(identity="fee", amount="-0.03", **kw):
    return {
        "id": identity,
        "activity_type": "FEE",
        "currency": "USD",
        "date": "2019-05-24",
        "status": "executed",
        "net_amount": amount,
        "execution_id": EXID,
        **kw,
    }


def test_exact_execution_link_and_daily_fee_kept_unallocated():
    rows = project_fees([FILL, fee(), fee("daily", "-0.02", execution_id=None)])
    assert rows[0]["fill_activity_id"] == FILL["id"]
    assert rows[0]["known_cost"] == Decimal("0.03")
    assert rows[1]["fill_activity_id"] is None
    summary = fee_summary(rows)
    assert summary["confirmed_cost"] == Decimal("0.05")
    assert summary["unallocated_cost"] == Decimal("0.02")
    assert summary["completeness"] != "complete"


@pytest.mark.parametrize(
    "changes",
    [
        {"execution_id": "missing"},
        {"execution_id": "prefix-" + EXID},
        {"execution_id": EXID[:-1]},
    ],
)
def test_no_fuzzy_or_date_only_attribution(changes):
    row = project_fees([FILL, fee(**changes)])[0]
    assert row["status"] == "unallocated"
    assert row["fill_activity_id"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"currency": "EUR"},
        {"net_amount": "NaN"},
        {"net_amount": True},
        {"date": "2019-05-23"},
        {"symbol": "OTHER"},
    ],
)
def test_invalid_fee_cannot_be_verified(changes):
    rows = project_fees([FILL, fee(**changes)])
    assert rows[0]["status"] == "invalid"
    assert fee_summary(rows)["confirmed_cost"] == 0
    assert fee_summary(rows)["invalid_count"] == 1


def test_duplicate_delivery_conflict_and_ambiguous_execution():
    assert fee_summary(project_fees([FILL, fee(), fee()]))["linked_cost"] == Decimal("0.03")
    with pytest.raises(ValueError, match="conflicting"):
        project_fees([fee(), fee(amount="-9")])
    another = {**FILL, "id": "20190524123406977::" + EXID}
    assert project_fees([FILL, another, fee()])[0]["status"] == "unallocated"


def test_pending_charge_is_not_confirmed_and_missing_fill_can_arrive_later():
    assert project_fees([fee()])[0]["status"] == "unallocated"
    assert project_fees([fee(), FILL])[0]["status"] == "linked"
    rows = project_fees([FILL, fee(status="pending")])
    assert fee_summary(rows)["confirmed_cost"] == 0
    assert fee_summary(rows)["pending_count"] == 1


@given(
    st.decimals(min_value="-100", max_value="100", places=4),
    st.decimals(min_value="-100", max_value="100", places=4),
)
def test_cashflow_conservation_with_rebates(linked, aggregate):
    rows = project_fees(
        [FILL, fee(amount=str(linked)), fee("daily", str(aggregate), execution_id=None)]
    )
    result = fee_summary(rows)
    assert result["confirmed_cost"] == -(linked + aggregate)
    assert result["confirmed_cost"] == result["linked_cost"] + result["unallocated_cost"]
