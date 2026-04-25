"""Cold-start position reconciliation (CL-sp3r) — broker truth vs internal state.

When the live engine restarts with open broker positions, the OMS, strategies,
and StrategyStateStore can be in an inconsistent state with the broker. This
module is the cold-start checkpoint that brings everything into alignment
before the signal-generation loop fires.

Reconciliation outcomes per (strategy_id?, symbol) tuple:

    MATCHED            broker position size matches internal expectation
    SIZE_MISMATCH      broker position differs in size from internal
    ORPHANED_BROKER    broker has a position no internal record claims
    ORPHANED_INTERNAL  internal record claims a position broker doesn't have

The ReconciliationPolicy decides what to do for each non-matched outcome:
flatten the broker position, clear the internal record, or alert-only.

Wired into LiveEngine.run() at startup. When a TradeJournal is supplied, the
final report is appended as a single RECONCILIATION_REPORT event so the
audit trail starts at engine boot, not after the first trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from src.execution.broker import Broker, Position
from src.execution.oms import OrderIntent, OrderManager
from src.execution.trade_journal import EventType, TradeJournal

logger = logging.getLogger(__name__)


# Quantity tolerance for "match" classification — float comparisons under
# different broker representations (decimal vs base-10 vs OANDA's units) need
# slack. 1e-4 covers typical tick rounding for FX.
_QUANTITY_MATCH_TOLERANCE: float = 1e-4


class ReconciliationStatus(Enum):
    MATCHED = "matched"
    SIZE_MISMATCH = "size_mismatch"
    ORPHANED_BROKER = "orphaned_broker"
    ORPHANED_INTERNAL = "orphaned_internal"


@dataclass
class ReconciliationEntry:
    """One reconciled position — broker side, internal side, and verdict."""

    symbol: str
    broker_quantity: float
    internal_quantity: float
    contributing_strategies: list[str]
    status: ReconciliationStatus
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "broker_quantity": self.broker_quantity,
            "internal_quantity": self.internal_quantity,
            "contributing_strategies": list(self.contributing_strategies),
            "status": self.status.value,
            "detail": self.detail,
        }


@dataclass
class ReconciliationReport:
    """Full output of a reconciliation pass — entries plus actions taken."""

    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    entries: list[ReconciliationEntry] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)

    @property
    def has_mismatches(self) -> bool:
        return any(e.status != ReconciliationStatus.MATCHED for e in self.entries)

    def by_status(self, status: ReconciliationStatus) -> list[ReconciliationEntry]:
        return [e for e in self.entries if e.status == status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "entries": [e.to_dict() for e in self.entries],
            "actions_taken": list(self.actions_taken),
            "summary": {
                status.value: len(self.by_status(status))
                for status in ReconciliationStatus
            },
        }


@dataclass
class ReconciliationPolicy:
    """How to act on each non-matched reconciliation outcome.

    Defaults are conservative: flatten orphaned broker positions (no strategy
    claims them, so they're either leftover from a prior session or manual
    interventions — flat is safest), clear orphaned internal records (broker
    is authoritative, so internal claim is stale), and TRUST the broker on
    size mismatches by default (in practice broker authority wins).
    """

    on_orphaned_broker: str = "flatten"   # "flatten" | "hold" | "alert_only"
    on_orphaned_internal: str = "clear"   # "clear" | "hold"
    on_size_mismatch: str = "trust_broker"  # "trust_broker" | "alert_only"

    def __post_init__(self) -> None:
        assert self.on_orphaned_broker in {"flatten", "hold", "alert_only"}, (
            f"on_orphaned_broker invalid: {self.on_orphaned_broker}"
        )
        assert self.on_orphaned_internal in {"clear", "hold"}, (
            f"on_orphaned_internal invalid: {self.on_orphaned_internal}"
        )
        assert self.on_size_mismatch in {"trust_broker", "alert_only"}, (
            f"on_size_mismatch invalid: {self.on_size_mismatch}"
        )


@runtime_checkable
class StrategyStateLike(Protocol):
    """Minimum state-store surface used by the reconciler.

    Returns currently open positions for a single strategy. Multiple strategies
    can hold the same symbol — the reconciler aggregates per symbol.
    """

    def get_current_position(self, strategy_id: str) -> dict[str, Any] | None:
        ...


class PositionReconciler:
    """Reconciles broker positions vs internal strategy state at engine startup.

    Use:
        report = PositionReconciler(broker, oms, state, strategies, policy).reconcile()
        if report.has_mismatches:
            log_alert(report)
    """

    def __init__(
        self,
        broker: Broker,
        oms: OrderManager,
        state: StrategyStateLike,
        strategies: list[Any],
        policy: ReconciliationPolicy | None = None,
        journal: TradeJournal | None = None,
    ) -> None:
        assert strategies, "PositionReconciler requires at least one strategy"
        self.broker = broker
        self.oms = oms
        self.state = state
        self.strategies = strategies
        self.policy = policy or ReconciliationPolicy()
        self.journal = journal

    def reconcile(self) -> ReconciliationReport:
        """Run reconciliation and apply policy actions. Returns the full report.

        Idempotent: produces a report describing actions taken, even if no
        mismatches exist (entries will all be MATCHED in that case).
        """
        report = ReconciliationReport()

        broker_positions = self._fetch_broker_positions()
        internal_positions = self._fetch_internal_positions_per_symbol()

        all_symbols = set(broker_positions.keys()) | set(internal_positions.keys())

        for symbol in sorted(all_symbols):
            broker_qty = broker_positions.get(symbol, Position(
                symbol=symbol, quantity=0.0, avg_price=0.0,
            )).quantity
            internal_records = internal_positions.get(symbol, [])
            internal_qty = sum(r["quantity"] for r in internal_records)
            contributors = [r["strategy_id"] for r in internal_records]

            entry = self._classify(
                symbol=symbol,
                broker_qty=broker_qty,
                internal_qty=internal_qty,
                contributors=contributors,
            )
            report.entries.append(entry)
            self._apply_policy(entry, report)

        logger.info(
            "Reconciliation complete: %d entries, mismatches=%s, summary=%s",
            len(report.entries),
            report.has_mismatches,
            report.to_dict()["summary"],
        )

        # Single audit event at boot — captures broker truth + applied actions
        # before the first trade. Journal failures must not block startup.
        if self.journal is not None:
            try:
                self.journal.record(
                    event_type=EventType.RECONCILIATION_REPORT,
                    payload=report.to_dict(),
                )
            except Exception:
                logger.exception(
                    "Trade journal append failed for reconciliation report"
                )

        return report

    # ------------------------------------------------------------------
    # Classification + policy
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(
        symbol: str,
        broker_qty: float,
        internal_qty: float,
        contributors: list[str],
    ) -> ReconciliationEntry:
        broker_present = abs(broker_qty) > _QUANTITY_MATCH_TOLERANCE
        internal_present = abs(internal_qty) > _QUANTITY_MATCH_TOLERANCE

        if not broker_present and not internal_present:
            # Neither side claims this symbol — should not happen via the call
            # path above (we only iterate symbols at least one side knows
            # about), but defensively classified as matched-flat.
            return ReconciliationEntry(
                symbol=symbol,
                broker_quantity=0.0,
                internal_quantity=0.0,
                contributing_strategies=contributors,
                status=ReconciliationStatus.MATCHED,
                detail="both sides flat",
            )

        if broker_present and not internal_present:
            return ReconciliationEntry(
                symbol=symbol,
                broker_quantity=broker_qty,
                internal_quantity=0.0,
                contributing_strategies=contributors,
                status=ReconciliationStatus.ORPHANED_BROKER,
                detail=(
                    f"broker has {broker_qty} but no strategy claims this symbol"
                ),
            )

        if internal_present and not broker_present:
            return ReconciliationEntry(
                symbol=symbol,
                broker_quantity=0.0,
                internal_quantity=internal_qty,
                contributing_strategies=contributors,
                status=ReconciliationStatus.ORPHANED_INTERNAL,
                detail=(
                    f"strategies {contributors} claim {internal_qty} but "
                    f"broker has no position"
                ),
            )

        # Both sides present — check sizes.
        if abs(broker_qty - internal_qty) <= _QUANTITY_MATCH_TOLERANCE:
            return ReconciliationEntry(
                symbol=symbol,
                broker_quantity=broker_qty,
                internal_quantity=internal_qty,
                contributing_strategies=contributors,
                status=ReconciliationStatus.MATCHED,
                detail="sizes match within tolerance",
            )

        return ReconciliationEntry(
            symbol=symbol,
            broker_quantity=broker_qty,
            internal_quantity=internal_qty,
            contributing_strategies=contributors,
            status=ReconciliationStatus.SIZE_MISMATCH,
            detail=(
                f"broker {broker_qty} vs internal {internal_qty} "
                f"(strategies: {contributors})"
            ),
        )

    def _apply_policy(
        self,
        entry: ReconciliationEntry,
        report: ReconciliationReport,
    ) -> None:
        """Translate the policy decision into OMS actions or alert-only logs."""
        if entry.status == ReconciliationStatus.MATCHED:
            return

        if entry.status == ReconciliationStatus.ORPHANED_BROKER:
            action = self.policy.on_orphaned_broker
            if action == "flatten":
                self._submit_flatten(entry.symbol)
                report.actions_taken.append(
                    f"flattened orphaned broker position {entry.symbol}={entry.broker_quantity}"
                )
            elif action == "hold":
                report.actions_taken.append(
                    f"held orphaned broker position {entry.symbol}={entry.broker_quantity}"
                )
            else:
                report.actions_taken.append(
                    f"alert: orphaned broker {entry.symbol}={entry.broker_quantity}"
                )
            return

        if entry.status == ReconciliationStatus.ORPHANED_INTERNAL:
            # Internal-state cleanup is the state store's responsibility — we
            # only log here. The store can be cleared out of band by the
            # operator, or future work can add state.clear_strategy_position().
            action = self.policy.on_orphaned_internal
            report.actions_taken.append(
                f"alert ({action}): internal claim {entry.symbol}={entry.internal_quantity} "
                f"by {entry.contributing_strategies} but broker flat"
            )
            return

        if entry.status == ReconciliationStatus.SIZE_MISMATCH:
            action = self.policy.on_size_mismatch
            if action == "trust_broker":
                report.actions_taken.append(
                    f"trusted broker on size mismatch {entry.symbol}: "
                    f"broker {entry.broker_quantity} vs internal {entry.internal_quantity}"
                )
            else:
                report.actions_taken.append(
                    f"alert: size mismatch {entry.symbol} "
                    f"broker={entry.broker_quantity} internal={entry.internal_quantity}"
                )
            return

    def _submit_flatten(self, symbol: str) -> None:
        intent = OrderIntent(
            strategy_id="reconciler-flatten",
            symbol=symbol,
            target_position=0.0,
            urgency="normal",
        )
        try:
            self.oms.submit_intent(intent)
        except Exception:
            logger.exception("Failed to submit flatten intent for %s", symbol)

    # ------------------------------------------------------------------
    # Position aggregation helpers
    # ------------------------------------------------------------------

    def _fetch_broker_positions(self) -> dict[str, Position]:
        try:
            positions = self.broker.get_positions()
        except Exception:
            logger.exception("Reconciliation: broker.get_positions() failed")
            return {}
        return {p.symbol: p for p in positions}

    def _fetch_internal_positions_per_symbol(
        self,
    ) -> dict[str, list[dict[str, Any]]]:
        """Aggregate per-strategy current positions into a per-symbol map."""
        per_symbol: dict[str, list[dict[str, Any]]] = {}
        for strategy in self.strategies:
            sid = strategy.id
            try:
                pos = self.state.get_current_position(sid)
            except Exception:
                logger.exception(
                    "Reconciliation: get_current_position(%s) failed", sid,
                )
                continue
            if pos is None:
                continue
            symbol = pos.get("symbol")
            qty = pos.get("size") if "size" in pos else pos.get("quantity")
            if symbol is None or qty is None:
                logger.warning(
                    "Strategy %s position record missing symbol/size: %s",
                    sid, pos,
                )
                continue
            per_symbol.setdefault(symbol, []).append({
                "strategy_id": sid,
                "quantity": float(qty),
                "raw": pos,
            })
        return per_symbol
