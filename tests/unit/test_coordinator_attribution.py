"""PortfolioCoordinator attribution + slippage policy (CL-5bwc).

Before CL-5bwc the coordinator rebuilt every netted intent as
``OrderIntent(strategy_id="portfolio", ...)`` with no metadata and the
default 2 bps slippage, so per-strategy identity, feature-snapshot ids
(CL-xpw9) and each strategy's own slippage bound were lost at the
aggregation boundary. Requirements under test:

* the aggregated intent's metadata lists every contribution
  (strategy_id, target, snapshot ids, slippage);
* slippage = tightest positive bound among this tick's contributors;
* the attribution reaches the TradeJournal INTENT_SUBMITTED payload via the
  REAL OrderManager, and ``reconstruct_features`` resolves each
  contributor's snapshot from the aggregated intent id;
* routing / aggregation math is UNCHANGED — a differential check against
  golden outputs captured by running these exact scenarios on the
  pre-change coordinator (commit 5d44d14).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import create_engine

from src.execution.oms import OrderIntent, OrderManager, Urgency
from src.execution.paper_broker import PaperBroker
from src.execution.trade_journal import EventType, TradeJournal
from src.models.feature_versioning import (
    FeatureSnapshot,
    FeatureSnapshotStore,
    reconstruct_features,
)
from src.portfolio.coordinator import PortfolioConstraints, PortfolioCoordinator


@dataclass
class _Strategy:
    id: str


class _RecordingOMS:
    def __init__(self) -> None:
        self.submitted: list[OrderIntent] = []

    def submit_intent(self, intent: OrderIntent, **kwargs: object) -> str:
        self.submitted.append(intent)
        return intent.intent_id


class _State:
    def __init__(self) -> None:
        self.portfolio_orders: list[tuple[str, float, dict[str, float]]] = []

    def record_reallocation(self, *args: Any) -> None:
        pass

    def record_portfolio_order(
        self,
        ts: datetime,
        symbol: str,
        target_position: float,
        strategy_contributions: dict[str, float],
    ) -> None:
        self.portfolio_orders.append((symbol, target_position, dict(strategy_contributions)))

    def get_positions_by_strategy(self, strategy_id: str) -> list[Any]:
        return []

    def load_strategy_returns_history(self, *args: Any) -> Any:
        import pandas as pd

        return pd.DataFrame()


def _broker() -> PaperBroker:
    b = PaperBroker(initial_capital=100_000.0)
    b.set_price("EURUSD", 1.0999, 1.1001)
    b.set_price("EUR_USD", 1.0999, 1.1001)
    b.set_price("USDJPY", 149.99, 150.01)
    b.set_price("GBPUSD", 1.2499, 1.2501)
    return b


def _coord(
    sids: list[str],
    weights: dict[str, float],
    oms: Any,
    broker: PaperBroker | None = None,
    constraints: PortfolioConstraints | None = None,
) -> tuple[PortfolioCoordinator, _State]:
    state = _State()
    coord = PortfolioCoordinator(
        strategies=[_Strategy(s) for s in sids],
        oms=oms,
        broker=broker or _broker(),
        state=state,  # type: ignore[arg-type]
        constraints=constraints,
    )
    coord.initialize_allocations(weights)
    return coord, state


# =============================================================================
# Differential: routing / aggregation math unchanged vs 5d44d14
# =============================================================================


def _i(sid: str, sym: str, tgt: float, urg: str = Urgency.NORMAL.value) -> OrderIntent:
    return OrderIntent(strategy_id=sid, symbol=sym, target_position=tgt, urgency=urg)


def _scenarios() -> dict[
    str,
    tuple[
        list[str], dict[str, float], list[dict[str, list[OrderIntent]]], PortfolioConstraints | None
    ],
]:  # noqa: E501
    return {
        "two_on_one": (
            ["s1", "s2"],
            {"s1": 0.5, "s2": 0.5},
            [{"s1": [_i("s1", "EURUSD", 1000)], "s2": [_i("s2", "EURUSD", 2000)]}],
            None,
        ),
        "conflict_mixed_dialect_escalation": (
            ["s1", "s2"],
            {"s1": 0.6, "s2": 0.4},
            [
                {
                    "s1": [_i("s1", "EURUSD", 10_000, Urgency.PASSIVE.value)],
                    "s2": [_i("s2", "EUR_USD", -4_000, Urgency.URGENT.value)],
                }
            ],
            None,
        ),
        "multi_symbol": (
            ["s1", "s2"],
            {"s1": 0.5, "s2": 0.5},
            [
                {
                    "s1": [_i("s1", "EURUSD", 1000), _i("s1", "USDJPY", 500)],
                    "s2": [_i("s2", "GBPUSD", -800)],
                }
            ],
            None,
        ),
        "remembered_share_on_exit": (
            ["s1", "s2"],
            {"s1": 0.5, "s2": 0.5},
            [
                {"s1": [_i("s1", "EURUSD", 1000)], "s2": [_i("s2", "EURUSD", 3000)]},
                {"s1": [_i("s1", "EURUSD", 0)]},
            ],
            None,
        ),
        "leverage_rescale": (
            ["s1", "s2"],
            {"s1": 0.5, "s2": 0.5},
            [
                {
                    "s1": [_i("s1", "EURUSD", 400_000)],
                    "s2": [_i("s2", "EURUSD", 300_000), _i("s2", "GBPUSD", 200_000)],
                }
            ],
            PortfolioConstraints(
                max_gross_leverage=1.0,
                max_notional_per_pair_pct=1.0,
                max_directional_exposure_per_currency=1.0,
                max_net_leverage=1.0,
            ),
        ),
    }


def _run_scenario(name: str) -> list[tuple[Any, ...]]:
    sids, weights, ticks, constraints = _scenarios()[name]
    oms = _RecordingOMS()
    coord, state = _coord(sids, weights, oms, constraints=constraints)
    for tick in ticks:
        asyncio.run(coord.process_intents(tick))
    out: list[tuple[Any, ...]] = [
        ("submit", i.strategy_id, i.symbol, round(i.target_position, 6), i.urgency)
        for i in oms.submitted
    ]
    out += [
        ("record", sym, round(tgt, 6), tuple(sorted((k, round(v, 6)) for k, v in c.items())))
        for sym, tgt, c in state.portfolio_orders
    ]
    return out


#: Captured by running ``_run_scenario`` on the pre-CL-5bwc coordinator
#: (5d44d14) — the reference path for the differential check.
_GOLDEN: dict[str, list[tuple[Any, ...]]] = {
    "conflict_mixed_dialect_escalation": [
        ("submit", "portfolio", "EURUSD", 4400.0, "urgent"),
        ("record", "EURUSD", 4400.0, (("s1", 6000.0), ("s2", -1600.0))),
    ],
    "leverage_rescale": [
        ("submit", "portfolio", "EURUSD", 68627.45098, "normal"),
        ("submit", "portfolio", "GBPUSD", 19607.843137, "normal"),
        ("record", "EURUSD", 68627.45098, (("s1", 39215.686275), ("s2", 29411.764706))),
        ("record", "GBPUSD", 19607.843137, (("s2", 19607.843137),)),
    ],
    "multi_symbol": [
        ("submit", "portfolio", "EURUSD", 500.0, "normal"),
        ("submit", "portfolio", "USDJPY", 233.333333, "normal"),
        ("submit", "portfolio", "GBPUSD", -400.0, "normal"),
        ("record", "EURUSD", 500.0, (("s1", 500.0),)),
        ("record", "USDJPY", 233.333333, (("s1", 233.333333),)),
        ("record", "GBPUSD", -400.0, (("s2", -400.0),)),
    ],
    "remembered_share_on_exit": [
        ("submit", "portfolio", "EURUSD", 2000.0, "normal"),
        ("submit", "portfolio", "EURUSD", 1500.0, "normal"),
        ("record", "EURUSD", 2000.0, (("s1", 500.0), ("s2", 1500.0))),
        ("record", "EURUSD", 1500.0, (("s1", 0.0), ("s2", 1500.0))),
    ],
    "two_on_one": [
        ("submit", "portfolio", "EURUSD", 1500.0, "normal"),
        ("record", "EURUSD", 1500.0, (("s1", 500.0), ("s2", 1000.0))),
    ],
}


@pytest.mark.parametrize("name", sorted(_scenarios()))
def test_routing_and_aggregation_unchanged_vs_base(name: str) -> None:
    assert name in _GOLDEN, f"no golden captured for {name}"
    assert _run_scenario(name) == _GOLDEN[name]


# =============================================================================
# Attribution + slippage policy
# =============================================================================


class TestAttribution:
    def test_two_strategies_one_intent_with_both_contributions(self) -> None:
        oms = _RecordingOMS()
        coord, _ = _coord(["s1", "s2"], {"s1": 0.5, "s2": 0.5}, oms)
        a = OrderIntent(
            strategy_id="s1",
            symbol="EURUSD",
            target_position=1000,
            max_slippage_bps=10.0,
            metadata={"snapshot_id": "snap-a", "trigger": "entry"},
        )
        b = OrderIntent(
            strategy_id="s2",
            symbol="EUR_USD",
            target_position=2000,
            max_slippage_bps=3.0,
            metadata={"snapshot_id": "snap-b"},
        )
        asyncio.run(coord.process_intents({"s1": [a], "s2": [b]}))

        assert len(oms.submitted) == 1
        out = oms.submitted[0]
        assert out.strategy_id == "portfolio"
        assert out.target_position == pytest.approx(1500.0)
        # Tightest positive bound wins (3 < 10), not the 2 bps default.
        assert out.max_slippage_bps == pytest.approx(3.0)
        contribs = {c["strategy_id"]: c for c in out.metadata["contributions"]}
        assert set(contribs) == {"s1", "s2"}
        assert contribs["s1"]["target_position"] == pytest.approx(500.0)
        assert contribs["s2"]["target_position"] == pytest.approx(1000.0)
        assert contribs["s1"]["snapshot_ids"] == ["snap-a"]
        assert contribs["s2"]["snapshot_ids"] == ["snap-b"]
        assert contribs["s1"]["max_slippage_bps"] == pytest.approx(10.0)
        assert contribs["s2"]["max_slippage_bps"] == pytest.approx(3.0)
        assert contribs["s1"]["intent_ids"] == [a.intent_id]
        assert contribs["s2"]["intent_ids"] == [b.intent_id]
        assert contribs["s1"]["source"] == contribs["s2"]["source"] == "intent"
        assert contribs["s1"]["metadata"] == [{"snapshot_id": "snap-a", "trigger": "entry"}]
        # Multi-contributor: no single snapshot_id is promoted to top level.
        assert "snapshot_id" not in out.metadata
        assert out.metadata["slippage_policy"] == "tightest_positive"

    def test_unbounded_contributor_never_counts_as_tightest(self) -> None:
        oms = _RecordingOMS()
        coord, _ = _coord(["s1", "s2"], {"s1": 0.5, "s2": 0.5}, oms)
        asyncio.run(
            coord.process_intents(
                {
                    "s1": [OrderIntent("s1", "EURUSD", 1000, max_slippage_bps=0.0)],
                    "s2": [OrderIntent("s2", "EURUSD", 1000, max_slippage_bps=7.5)],
                }
            )
        )
        assert oms.submitted[0].max_slippage_bps == pytest.approx(7.5)

    def test_single_strategy_keeps_its_own_metadata_and_slippage(self) -> None:
        oms = _RecordingOMS()
        coord, _ = _coord(["s1"], {"s1": 1.0}, oms)
        a = OrderIntent(
            strategy_id="s1",
            symbol="EURUSD",
            target_position=1000,
            max_slippage_bps=10.0,
            metadata={"snapshot_id": "snap-a", "trigger": "entry"},
        )
        asyncio.run(coord.process_intents({"s1": [a]}))
        out = oms.submitted[0]
        assert out.target_position == pytest.approx(1000.0)
        assert out.symbol == "EURUSD"
        assert out.max_slippage_bps == pytest.approx(10.0)
        # Same top-level payload the strategy would have journaled directly.
        assert out.metadata["snapshot_id"] == "snap-a"
        assert out.metadata["trigger"] == "entry"
        assert [c["strategy_id"] for c in out.metadata["contributions"]] == ["s1"]

    def test_remembered_share_is_attributed_without_fabricated_slippage(self) -> None:
        oms = _RecordingOMS()
        coord, _ = _coord(["s1", "s2"], {"s1": 0.5, "s2": 0.5}, oms)
        asyncio.run(
            coord.process_intents(
                {
                    "s1": [OrderIntent("s1", "EURUSD", 1000, max_slippage_bps=4.0)],
                    "s2": [OrderIntent("s2", "EURUSD", 3000, max_slippage_bps=1.0)],
                }
            )
        )
        asyncio.run(
            coord.process_intents({"s1": [OrderIntent("s1", "EURUSD", 0, max_slippage_bps=4.0)]})
        )
        out = oms.submitted[-1]
        assert out.target_position == pytest.approx(1500.0)  # s2's remembered share
        contribs = {c["strategy_id"]: c for c in out.metadata["contributions"]}
        assert contribs["s2"]["source"] == "remembered"
        assert contribs["s2"]["max_slippage_bps"] is None
        assert contribs["s2"]["intent_ids"] == []
        assert contribs["s2"]["target_position"] == pytest.approx(1500.0)
        # Only this tick's speaker (s1, 4 bps) sets the bound.
        assert out.max_slippage_bps == pytest.approx(4.0)


class TestJournaledAttribution:
    """End-to-end through the REAL OrderManager + TradeJournal + snapshot store."""

    def test_attribution_journaled_and_features_reconstructable(self, tmp_path: Any) -> None:
        # File-backed: the coordinator runs the OMS off-loop (to_thread), and
        # each thread would get its own empty sqlite :memory: database.
        engine = create_engine(f"sqlite:///{tmp_path / 'journal.db'}")
        journal = TradeJournal(engine)
        store = FeatureSnapshotStore(engine)
        snap_a = FeatureSnapshot.create(
            feature_set_name="fa",
            feature_set_version="1",
            data_snapshot_id="live",
            model_version="m1",
            ts=datetime(2026, 10, 6, tzinfo=UTC),
            values={"z": 2.1},
        )
        snap_b = FeatureSnapshot.create(
            feature_set_name="fb",
            feature_set_version="1",
            data_snapshot_id="live",
            model_version="m2",
            ts=datetime(2026, 10, 6, tzinfo=UTC),
            values={"sent": -0.4},
        )
        store.store(snap_a)
        store.store(snap_b)

        broker = _broker()
        oms = OrderManager(broker, journal=journal)
        coord, _ = _coord(["s1", "s2"], {"s1": 0.5, "s2": 0.5}, oms, broker=broker)
        asyncio.run(
            coord.process_intents(
                {
                    "s1": [
                        OrderIntent(
                            "s1",
                            "EURUSD",
                            1000,
                            max_slippage_bps=10.0,
                            metadata={"snapshot_id": snap_a.snapshot_id},
                        )
                    ],
                    "s2": [
                        OrderIntent(
                            "s2",
                            "EURUSD",
                            2000,
                            max_slippage_bps=10.0,
                            metadata={"snapshot_id": snap_b.snapshot_id},
                        )
                    ],
                }
            )
        )

        submitted = [e for e in journal.all_events() if e.event_type == EventType.INTENT_SUBMITTED]
        assert len(submitted) == 1
        ev = submitted[0]
        assert ev.strategy_id == "portfolio"
        assert ev.payload["max_slippage_bps"] == pytest.approx(10.0)
        journaled = {c["strategy_id"]: c for c in ev.payload["contributions"]}
        assert journaled["s1"]["snapshot_ids"] == [snap_a.snapshot_id]
        assert journaled["s2"]["snapshot_ids"] == [snap_b.snapshot_id]
        assert journaled["s1"]["target_position"] == pytest.approx(500.0)
        assert journaled["s2"]["target_position"] == pytest.approx(1000.0)

        rec_a = reconstruct_features(journal, store, ev.intent_id, strategy_id="s1")
        rec_b = reconstruct_features(journal, store, ev.intent_id, strategy_id="s2")
        assert rec_a is not None and rec_a.snapshot_id == snap_a.snapshot_id
        assert rec_b is not None and rec_b.snapshot_id == snap_b.snapshot_id
        assert rec_a.values == {"z": 2.1}
        # Unfiltered lookup still resolves (first contributor's snapshot).
        assert reconstruct_features(journal, store, ev.intent_id) is not None
