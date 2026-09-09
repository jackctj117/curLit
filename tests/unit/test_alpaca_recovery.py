"""Independent broker fixtures for GET-only recovery; no broker/DB credentials."""

from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scripts.alpaca_recovery_report import write_private

from src.execution.alpaca_recovery import (
    EvidenceError,
    PaperEvidenceClient,
    build_report,
    equity_repair_candidate,
    fill_evidence,
    fingerprint,
    quantity,
)


def test_activity_pagination_uses_id_not_timestamp_and_allows_short_pages() -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "GET"
        assert request.url.host == "paper-api.alpaca.markets"
        # Same timestamp, distinct order IDs; a short page does not mean EOF.
        pages = [
            [{"id": "b", "submitted_at": "2026-09-09T13:30:00Z"}],
            [{"id": "a", "submitted_at": "2026-09-09T13:30:00Z"}],
            [],
        ]
        return httpx.Response(200, json=pages[len(calls) - 1])

    client = PaperEvidenceClient("fixture", "fixture", transport=httpx.MockTransport(respond))
    result = client.history("activities", max_pages=3)
    client.close()
    assert result.exhausted
    assert [r["id"] for r in result.records] == ["b", "a"]
    assert calls[1].url.params["page_token"] == "b"
    assert calls[2].url.params["page_token"] == "a"
    assert "until" not in calls[1].url.params


@pytest.mark.parametrize(
    "body,reason",
    [([{"id": "same"}], "nonadvancing_cursor"), ({"error": "not a list"}, "invalid_page")],
)
def test_incomplete_history_never_looks_empty_success(body: Any, reason: str) -> None:
    client = PaperEvidenceClient(
        "fixture",
        "fixture",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )
    result = client.history("activities", max_pages=3)
    client.close()
    assert not result.exhausted
    assert result.reason == reason


def test_http_failure_is_redacted_and_not_missing_order() -> None:
    client = PaperEvidenceClient(
        "fixture",
        "private-value",
        transport=httpx.MockTransport(lambda request: httpx.Response(429, text="private-value")),
    )
    with pytest.raises(EvidenceError, match="^http_429$"):
        client.order(client_id="original")
    history = client.history("activities")
    client.close()
    assert not history.exhausted and history.reason == "http_429"


def test_page_budget_is_explicit() -> None:
    client = PaperEvidenceClient(
        "fixture",
        "fixture",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[{"id": "one"}])),
    )
    result = client.history("activities", max_pages=1)
    client.close()
    assert not result.exhausted and result.reason == "page_budget_exhausted"
    assert len(result.records) == 1


def test_order_timestamp_paging_reaches_history_and_requires_empty_tail() -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        pages = [
            [{"id": "new", "submitted_at": "2026-09-09T14:00:00Z"}],
            [{"id": "old", "submitted_at": "2026-07-01T14:00:00Z"}],
            [],
        ]
        assert request.method == "GET"
        return httpx.Response(200, json=pages[len(calls) - 1])

    client = PaperEvidenceClient("fixture", "fixture", transport=httpx.MockTransport(respond))
    result = client.history("orders", max_pages=3)
    client.close()
    assert result.exhausted
    assert [r["id"] for r in result.records] == ["new", "old"]
    assert calls[1].url.params["until"] == "2026-09-09T14:00:00Z"
    assert calls[2].url.params["until"] == "2026-07-01T14:00:00Z"


def test_full_order_page_cannot_hide_unretrieved_timestamp_ties() -> None:
    pages = iter([[{"id": str(i), "submitted_at": "2026-09-09T14:00:00Z"} for i in range(500)], []])
    client = PaperEvidenceClient(
        "fixture",
        "fixture",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=next(pages))),
    )
    result = client.history("orders", max_pages=2)
    client.close()
    assert not result.exhausted
    assert result.reason == "timestamp_boundary_unverified"


def test_order_endpoint_ignoring_time_cursor_is_incomplete() -> None:
    pages = iter(
        [
            [{"id": "old", "submitted_at": "2026-07-01T14:00:00Z"}],
            [{"id": "new", "submitted_at": "2026-09-09T14:00:00Z"}],
        ]
    )
    client = PaperEvidenceClient(
        "fixture",
        "fixture",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=next(pages))),
    )
    result = client.history("orders", max_pages=2)
    client.close()
    assert not result.exhausted and result.reason == "invalid_order_time_boundary"


def order() -> dict[str, Any]:
    return {
        "id": "exit",
        "client_order_id": "curlit-exit-idea",
        "symbol": "OPTION",
        "side": "sell",
        "status": "canceled",
        "qty": "3",
        "filled_qty": "1",
    }


def fill() -> dict[str, Any]:
    return {
        "id": "fill",
        "activity_type": "FILL",
        "order_id": "exit",
        "symbol": "OPTION",
        "side": "sell",
        "qty": "1",
        "price": "2.50",
        "transaction_time": "2026-09-08T14:00:00Z",
    }


def test_partial_cancel_and_duplicate_fill_do_not_erase_or_double_inventory() -> None:
    result = fill_evidence(order(), [fill(), fill()])
    assert result == {
        "status": "matched",
        "order_filled_qty": "1",
        "activity_qty": "1",
        "unfilled_order_qty": "2",
        "fill_price_quantity_sum": "2.50",
        "activity_ids": ["fill"],
        "fees": None,
        "realized_pnl": None,
    }


def test_conflicting_fill_id_fails_closed() -> None:
    conflicting = {**fill(), "qty": "2"}
    assert fill_evidence(order(), [fill(), conflicting])["status"] == "invalid_evidence"


def test_missing_fill_is_unknown_not_last_quote() -> None:
    result = fill_evidence(order(), [])
    assert result["status"] == "quantity_mismatch"
    assert result["realized_pnl"] is None


def snapshot() -> dict[str, Any]:
    internal = {
        "options": [
            {
                "idea_id": "idea",
                "occ_symbol": "OPTION",
                "status": "submitted",
                "exit_status": "submitted",
                "premium_est": "999",
            }
        ],
        "equities": [
            {"idea_id": "old", "ticker": "ABC", "status": "submitted", "exit_status": "closed"}
        ],
    }
    positions = [{"asset_id": "asset", "symbol": "ABC", "qty": "-1.25", "asset_class": "us_equity"}]
    return {
        "account_scope": "paper:fixture",
        "finished_at": "2026-09-09T14:00:00Z",
        "internal": internal,
        "internal_after": deepcopy(internal),
        "positions_before": positions,
        "positions_after": deepcopy(positions),
        "orders": {"records": [], "exhausted": True},
        "activities": {"records": [fill()], "exhausted": True},
        "exit_lookups": {"idea": {"order": order()}},
    }


def test_report_never_adopts_by_symbol_or_clears_partial_cancel() -> None:
    report = build_report(snapshot())
    pending = report["pending_option_exits"][0]
    assert pending["classification"] == "terminal_canceled"
    assert pending["policy"] == "reconstruct_owned_residual_before_any_new_attempt"
    holding = report["unmatched_equity_holdings"][0]
    assert holding["signed_qty"] == "-1.25"
    assert holding["classification"] == "ownership_unresolved"
    assert "no_automatic_adoption" in holding["policy"]
    assert report["internal_stable_during_capture"]


def test_missing_order_or_position_does_not_prove_closure() -> None:
    data = snapshot()
    data["exit_lookups"] = {"idea": {"error": "http_404"}}
    finding = build_report(data)["pending_option_exits"][0]
    assert finding["classification"] == "reconciliation_required"
    assert finding["policy"] == "no_retry_no_accounting_finalization"


def test_identity_mismatch_blocks_terminal_order_disposition() -> None:
    data = snapshot()
    data["exit_lookups"]["idea"]["order"]["client_order_id"] = "someone-elses-order"
    assert build_report(data)["pending_option_exits"][0]["classification"] == "identity_mismatch"


def test_capture_race_is_visible_and_offline_replay_deterministic() -> None:
    data = snapshot()
    data["positions_after"][0]["qty"] = "-2.25"
    data["internal_after"]["options"][0]["exit_status"] = "closed"
    report = build_report(data)
    assert not report["inventory_stable_during_capture"]
    assert not report["internal_stable_during_capture"]
    assert report == build_report(json.loads(json.dumps(data)))
    assert fingerprint(data) != fingerprint(snapshot())


@pytest.mark.parametrize("value", [None, True, "NaN", "Infinity", "junk"])
def test_unknown_quantity_is_not_zero(value: object) -> None:
    with pytest.raises(ValueError):
        quantity(value)


@given(
    st.decimals(
        min_value="-100000", max_value="100000", places=6, allow_nan=False, allow_infinity=False
    )
)
def test_signed_fractional_quantity_round_trip(value: Decimal) -> None:
    assert quantity(str(value)) == value


def test_evidence_file_is_private_and_cannot_be_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    write_private(path, {"original": True})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        write_private(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}


def test_short_activity_side_is_matched_to_sell_order_without_losing_sign() -> None:
    entry = {
        **order(),
        "asset_id": "asset",
        "client_order_id": "curlit-eq-idea",
        "qty": "1.25",
        "filled_qty": "1.25",
        "status": "filled",
    }
    activity = {**fill(), "side": "sell_short", "qty": "1.25"}
    position = {"asset_id": "asset", "qty": "-1.25"}
    row = {
        "idea_id": "idea",
        "alpaca_order_id": "exit",
        "exit_status": "closed",
        "exit_reason": "closed_external",
        "exit_order_id": None,
        "side": "sell_short",
        "qty": "1.25",
        "submitted_at": "2026-09-08T13:59:00Z",
    }
    proposal = equity_repair_candidate(position, [row], [entry], [activity], ["idea"])
    assert proposal is not None and proposal["signed_qty"] == "-1.25"
    assert proposal["original_submitted_at"] == row["submitted_at"]
    # Symbol/prefix alone, or coincidentally similar quantities, is insufficient.
    assert (
        equity_repair_candidate(
            position, [{**row, "alpaca_order_id": "different"}], [entry], [activity], ["idea"]
        )
        is None
    )
    assert (
        equity_repair_candidate({**position, "qty": "-2"}, [row], [entry], [activity], ["idea"])
        is None
    )
    assert equity_repair_candidate(position, [row], [entry], [activity], []) is None
    assert equity_repair_candidate(position, [row], [entry], [activity, activity], ["idea"]) is None


@given(st.integers(min_value=1, max_value=1000), st.integers(min_value=0, max_value=1000))
def test_partial_fill_conservation_and_nonnegative_remainder(filled: int, remaining: int) -> None:
    broker = {**order(), "qty": str(filled + remaining), "filled_qty": str(filled)}
    result = fill_evidence(broker, [{**fill(), "qty": str(filled)}])
    assert result["status"] == "matched"
    assert (
        quantity(result["order_filled_qty"]) + quantity(result["unfilled_order_qty"])
        == filled + remaining
    )
    assert quantity(result["unfilled_order_qty"]) >= 0
    assert result["realized_pnl"] is None
