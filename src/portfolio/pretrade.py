"""Pre-trade validation gate (CL-srsy) — last checkpoint before OMS submit.

The PortfolioCoordinator scales and aggregates intents to satisfy gross-leverage
and per-pair constraints. PreTradeValidator runs ON TOP of that, validating each
final intent against ACCOUNT-level state (margin, current positions, broker
tradability) which the coordinator can't see.

Rejection layers (ordered by responsibility):
    Strategy intent  →  PortfolioCoordinator (scale + aggregate + constraints)
                    →  PreTradeValidator     (margin, post-trade leverage,
                                              concentration, tradability)
                    →  OMS / broker

Rejected intents emit a RejectionEvent with structured reason; counter metric
fx_pretrade_rejections_total{pair, reason} surfaces volume to Prometheus.

Reasons handled:
    INSUFFICIENT_MARGIN     account.margin_available < required for this trade
    GROSS_LEVERAGE_BREACH   post-trade gross > constraints.max_gross_leverage
    NET_LEVERAGE_BREACH     post-trade net   > constraints.max_net_leverage
    PER_PAIR_CONCENTRATION  post-trade pair notional > constraints cap
    PER_CURRENCY_EXPOSURE   post-trade per-CCY exposure > constraints cap
    INSTRUMENT_UNSUPPORTED  symbol not in optional whitelist
    INSTRUMENT_HALTED       optional broker tradability check returned False

Halt detection requires a TradabilityChecker; without one, halt rejection is
delegated to broker-side rejects (handled by CL-yteo order rejection policy).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from src.execution.broker import Broker, Position
from src.execution.oms import OrderIntent
from src.monitoring.metrics import pretrade_rejections
from src.portfolio.coordinator import PortfolioConstraints

logger = logging.getLogger(__name__)


# Required margin as a fraction of notional. OANDA retail leverage is up to
# 50:1 (2% margin); we conservatively require 5% to leave headroom for adverse
# moves before margin call. Tunable per deployment.
_DEFAULT_MARGIN_REQUIREMENT_PCT: float = 0.05


class RejectionReason(Enum):
    INSUFFICIENT_MARGIN = "insufficient_margin"
    GROSS_LEVERAGE_BREACH = "gross_leverage_breach"
    NET_LEVERAGE_BREACH = "net_leverage_breach"
    PER_PAIR_CONCENTRATION = "per_pair_concentration"
    PER_CURRENCY_EXPOSURE = "per_currency_exposure"
    INSTRUMENT_UNSUPPORTED = "instrument_unsupported"
    INSTRUMENT_HALTED = "instrument_halted"


@dataclass
class RejectionEvent:
    """Structured event recorded when an intent is rejected pre-trade."""

    intent: OrderIntent
    reason: RejectionReason
    detail: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "intent_id": self.intent.intent_id,
            "strategy_id": self.intent.strategy_id,
            "symbol": self.intent.symbol,
            "target_position": self.intent.target_position,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@runtime_checkable
class TradabilityChecker(Protocol):
    """Optional broker-side check for instrument halt / tradability state."""

    def is_tradable(self, symbol: str) -> bool:
        ...


class PreTradeValidator:
    """Validates intents against account-level state before OMS submission.

    The coordinator's portfolio-level constraints (gross leverage cap, per-pair
    cap) operate on intents-only. This validator considers the *resulting*
    post-trade state including currently held positions and account margin.

    Usage:
        validator = PreTradeValidator(broker, constraints)
        for intent in coordinator_output:
            rejection = validator.validate(intent)
            if rejection is not None:
                handle_rejection(rejection)
                continue
            oms.submit_intent(intent)
    """

    def __init__(
        self,
        broker: Broker,
        constraints: PortfolioConstraints,
        supported_instruments: set[str] | None = None,
        tradability: TradabilityChecker | None = None,
        margin_requirement_pct: float = _DEFAULT_MARGIN_REQUIREMENT_PCT,
    ) -> None:
        self.broker = broker
        self.constraints = constraints
        self.supported_instruments = supported_instruments
        self.tradability = tradability
        self.margin_requirement_pct = margin_requirement_pct

        assert margin_requirement_pct > 0, (
            f"margin_requirement_pct must be positive, got {margin_requirement_pct}"
        )

        self._rejections_log: list[RejectionEvent] = []

    def validate(
        self,
        intent: OrderIntent,
        current_positions: list[Position] | None = None,
    ) -> RejectionEvent | None:
        """Validate one intent. Returns RejectionEvent if rejected, else None.

        current_positions: optional pre-fetched positions (avoids re-fetching
        when validating a batch). If None, queries broker.
        """
        # Cheapest checks first — fail fast.
        rejection = self._check_instrument_supported(intent)
        if rejection is not None:
            return self._record(rejection)

        rejection = self._check_instrument_tradable(intent)
        if rejection is not None:
            return self._record(rejection)

        positions = current_positions if current_positions is not None else self.broker.get_positions()
        position_map = {p.symbol: p for p in positions}

        price = self._get_price(intent.symbol)
        new_pair_qty = intent.target_position
        existing_pair_qty = (
            position_map[intent.symbol].quantity
            if intent.symbol in position_map
            else 0.0
        )
        # Trade delta — this is what will actually be sent to the broker.
        # If new_pair_qty == existing, no order; treat as valid (no-op).
        delta_qty = new_pair_qty - existing_pair_qty
        if abs(delta_qty) < 1e-9:
            return None

        rejection = self._check_per_pair_concentration(intent, new_pair_qty, price)
        if rejection is not None:
            return self._record(rejection)

        account_equity = self._broker_equity()
        rejection = self._check_margin(intent, delta_qty, price, account_equity)
        if rejection is not None:
            return self._record(rejection)

        # Post-trade leverage requires aggregating the new state across all
        # symbols. We compute it assuming this is the only changing position;
        # if the validator is called for several intents in the same tick, the
        # caller should pass an updated current_positions reflecting prior
        # accepted intents.
        rejection = self._check_post_trade_leverage(
            intent, position_map, new_pair_qty, price, account_equity,
        )
        if rejection is not None:
            return self._record(rejection)

        rejection = self._check_currency_exposure(
            intent, position_map, new_pair_qty, account_equity,
        )
        if rejection is not None:
            return self._record(rejection)

        return None

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_instrument_supported(self, intent: OrderIntent) -> RejectionEvent | None:
        if self.supported_instruments is None:
            return None
        if intent.symbol not in self.supported_instruments:
            return RejectionEvent(
                intent=intent,
                reason=RejectionReason.INSTRUMENT_UNSUPPORTED,
                detail=f"{intent.symbol} not in supported set",
            )
        return None

    def _check_instrument_tradable(self, intent: OrderIntent) -> RejectionEvent | None:
        if self.tradability is None:
            return None
        try:
            if not self.tradability.is_tradable(intent.symbol):
                return RejectionEvent(
                    intent=intent,
                    reason=RejectionReason.INSTRUMENT_HALTED,
                    detail=f"{intent.symbol} not tradable per broker",
                )
        except Exception:
            # Tradability check failure is not by itself a rejection — log and
            # let other checks proceed; broker-side reject (CL-yteo) is the
            # final safety net.
            logger.exception("Tradability check raised for %s", intent.symbol)
        return None

    def _check_per_pair_concentration(
        self,
        intent: OrderIntent,
        new_pair_qty: float,
        price: float,
    ) -> RejectionEvent | None:
        equity = self._broker_equity()
        max_pair_notional = equity * self.constraints.max_notional_per_pair_pct
        post_pair_notional = abs(new_pair_qty) * price
        if post_pair_notional > max_pair_notional + 1e-6:
            return RejectionEvent(
                intent=intent,
                reason=RejectionReason.PER_PAIR_CONCENTRATION,
                detail=(
                    f"{intent.symbol} post-trade notional {post_pair_notional:.2f} "
                    f"exceeds cap {max_pair_notional:.2f}"
                ),
            )
        return None

    def _check_margin(
        self,
        intent: OrderIntent,
        delta_qty: float,
        price: float,
        equity: float,
    ) -> RejectionEvent | None:
        account = self.broker.get_account()
        # Required margin for the *additional* exposure (delta only).
        delta_notional = abs(delta_qty) * price
        required = delta_notional * self.margin_requirement_pct
        # margin_available may be 0 on some broker stubs; if so, fall back to
        # checking against equity as a soft proxy (avoids false rejects when
        # the broker doesn't expose margin precisely).
        available = (
            account.margin_available
            if account.margin_available > 0
            else equity - account.margin_used
        )
        if available < required - 1e-6:
            return RejectionEvent(
                intent=intent,
                reason=RejectionReason.INSUFFICIENT_MARGIN,
                detail=(
                    f"required margin {required:.2f} exceeds available {available:.2f}"
                ),
            )
        return None

    def _check_post_trade_leverage(
        self,
        intent: OrderIntent,
        position_map: dict[str, Position],
        new_pair_qty: float,
        price: float,
        equity: float,
    ) -> RejectionEvent | None:
        # Build post-trade notional map across all symbols.
        post_notional_signed: dict[str, float] = {}
        for sym, pos in position_map.items():
            sym_price = self._get_price(sym)
            post_notional_signed[sym] = pos.quantity * sym_price
        post_notional_signed[intent.symbol] = new_pair_qty * price

        gross = sum(abs(v) for v in post_notional_signed.values())
        net = abs(sum(post_notional_signed.values()))

        gross_leverage = gross / equity if equity > 0 else float("inf")
        net_leverage = net / equity if equity > 0 else float("inf")

        if gross_leverage > self.constraints.max_gross_leverage + 1e-6:
            return RejectionEvent(
                intent=intent,
                reason=RejectionReason.GROSS_LEVERAGE_BREACH,
                detail=(
                    f"post-trade gross leverage {gross_leverage:.2f}x exceeds "
                    f"max {self.constraints.max_gross_leverage:.2f}x"
                ),
            )
        if net_leverage > self.constraints.max_net_leverage + 1e-6:
            return RejectionEvent(
                intent=intent,
                reason=RejectionReason.NET_LEVERAGE_BREACH,
                detail=(
                    f"post-trade net leverage {net_leverage:.2f}x exceeds "
                    f"max {self.constraints.max_net_leverage:.2f}x"
                ),
            )
        return None

    def _check_currency_exposure(
        self,
        intent: OrderIntent,
        position_map: dict[str, Position],
        new_pair_qty: float,
        equity: float,
    ) -> RejectionEvent | None:
        # Aggregate post-trade per-currency exposure.
        exposures: dict[str, float] = defaultdict(float)
        for sym, pos in position_map.items():
            if len(sym) < 6:
                continue
            base, quote = sym[:3], sym[3:6]
            sym_price = self._get_price(sym)
            notional = pos.quantity * sym_price
            exposures[base] += notional
            exposures[quote] -= notional
        # Overlay this trade.
        if len(intent.symbol) >= 6:
            base, quote = intent.symbol[:3], intent.symbol[3:6]
            old_notional = (
                position_map[intent.symbol].quantity * self._get_price(intent.symbol)
                if intent.symbol in position_map
                else 0.0
            )
            new_notional = new_pair_qty * self._get_price(intent.symbol)
            delta = new_notional - old_notional
            exposures[base] += delta
            exposures[quote] -= delta

        max_exposure = equity * self.constraints.max_directional_exposure_per_currency
        for ccy, exp in exposures.items():
            if abs(exp) > max_exposure + 1e-6:
                return RejectionEvent(
                    intent=intent,
                    reason=RejectionReason.PER_CURRENCY_EXPOSURE,
                    detail=(
                        f"{ccy} post-trade exposure {exp:.2f} exceeds cap {max_exposure:.2f}"
                    ),
                )
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record(self, rejection: RejectionEvent) -> RejectionEvent:
        self._rejections_log.append(rejection)
        try:
            pretrade_rejections.labels(
                pair=rejection.intent.symbol,
                reason=rejection.reason.value,
            ).inc()
        except Exception:
            # Metric failures must not block trading decisions.
            logger.exception("pretrade_rejections metric increment failed")
        logger.warning(
            "Pre-trade reject %s symbol=%s reason=%s detail=%s",
            rejection.intent.intent_id,
            rejection.intent.symbol,
            rejection.reason.value,
            rejection.detail,
        )
        return rejection

    def _get_price(self, symbol: str) -> float:
        try:
            bid, ask = self.broker.get_price(symbol)
            mid = (bid + ask) / 2
            assert mid > 0, f"non-positive mid for {symbol}"
            return mid
        except Exception:
            logger.exception("Could not fetch price for %s; using 1.0 fallback", symbol)
            return 1.0

    def _broker_equity(self) -> float:
        equity = self.broker.get_account().equity
        assert equity > 0, f"non-positive equity {equity}"
        return equity

    @property
    def rejections_log(self) -> list[RejectionEvent]:
        return list(self._rejections_log)
