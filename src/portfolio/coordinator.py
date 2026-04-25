"""Portfolio Coordinator — combines strategy intents, applies portfolio constraints.

Sits above strategies, below OMS:

    Strategies → emit OrderIntent
       ↓
    PortfolioCoordinator.process_intents()
       — scale by allocation weight × correlation regime exposure × performance override
       — aggregate same-symbol intents (netting)
       — apply portfolio constraints (gross leverage, per-pair cap, currency exposure)
       — log conflicts (opposite signs on same symbol)
       ↓
    OMS.submit_intent()

Allocation weights are recomputed via inverse-volatility risk parity at most every
~3 weeks. The correlation regime check downscales total exposure when strategies
start moving together (crisis indicator).

Strategy lifecycle: add_strategy() introduces in paper mode; remove_strategy()
liquidates attributed positions and forces a rebalance.

Reference: reference/06_portfolio.md.
Related: D2 (CL-6tu) replaces inline risk parity with src/portfolio/risk_parity.py.
         D7 (CL-amf) wires this into LiveEngine.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.execution.broker import Broker
from src.execution.oms import OrderIntent, OrderManager
from src.monitoring.logging_setup import LogContext

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration constants
# =============================================================================

# Per-strategy weight bounds for risk parity. Source: reference/06_portfolio.md.
# 5% floor: prevents the optimizer from asymptotically zeroing out a strategy that
#           is on the books — flat-line strategies should be removed via
#           remove_strategy(), not shrunk to nothing.
# 40% cap:  prevents single-strategy concentration even when one very-low-vol
#           strategy would otherwise dominate the inverse-vol allocation.
_RISK_PARITY_BOUNDS_PER_STRATEGY: tuple[float, float] = (0.05, 0.40)

# Re-fit risk parity at most every ~3 weeks. Daily refits add transaction cost
# (rebalancing trades) without proportional benefit; covariance estimates are
# stable on a multi-week horizon. Per reference/06_portfolio.md.
_REBALANCE_MIN_INTERVAL_DAYS: int = 21

# Use 2 years of strategy returns for risk parity fitting. One year captures at
# least one regime transition (rate cycle, vol regime); two gives the EWMA enough
# horizon to discount stale data through the halflife (used by D2 risk_parity.py).
_RISK_PARITY_LOOKBACK_DAYS: int = 252 * 2

# Below 1 year of returns history risk parity is statistically unreliable; fall
# back to equal weight rather than over-fit to short windows.
_RISK_PARITY_MIN_HISTORY_DAYS: int = 252

# Recent-vs-baseline correlation z-score thresholds. Above 2.5σ flags "crisis"
# (cut exposure to 50%); above 1.5σ flags "stressed" (75%). Tuned in
# reference/06_portfolio.md from 2008 / 2020 / 2022 stress-test calibration.
_CORR_REGIME_RECENT_WINDOW: int = 40
_CORR_REGIME_BASELINE_WINDOW: int = 252
_CORR_REGIME_CRISIS_Z: float = 2.5
_CORR_REGIME_STRESSED_Z: float = 1.5
_CORR_REGIME_EXPOSURE_CRISIS: float = 0.50
_CORR_REGIME_EXPOSURE_STRESSED: float = 0.75
_CORR_REGIME_EXPOSURE_NORMAL: float = 1.00

# Trading days per year for annualized covariance. Convention.
_TRADING_DAYS_PER_YEAR: int = 252

# Urgency rank for escalation when multiple strategies share a symbol.
_URGENCY_RANK: dict[str, int] = {"passive": 0, "normal": 1, "urgent": 2}


# =============================================================================
# Protocol for state persistence
# =============================================================================


@runtime_checkable
class PortfolioStateProtocol(Protocol):
    """State persistence interface required by PortfolioCoordinator.

    Production wiring (D7 / CL-amf) supplies a concrete implementation.
    Tests inject a fake conforming to this protocol.
    """

    def record_reallocation(
        self,
        ts: datetime,
        weights: dict[str, float],
        regime: dict[str, Any],
    ) -> None:
        ...

    def record_portfolio_order(
        self,
        ts: datetime,
        symbol: str,
        target_position: float,
        strategy_contributions: dict[str, float],
    ) -> None:
        ...

    def get_positions_by_strategy(self, strategy_id: str) -> list[Any]:
        ...

    def load_strategy_returns_history(
        self,
        strategy_ids: list[str],
        lookback_days: int,
    ) -> pd.DataFrame:
        ...


# =============================================================================
# Allocation + constraints dataclasses
# =============================================================================


@dataclass
class StrategyAllocation:
    """Per-strategy allocation state.

    target_weight: fraction of portfolio risk budget allocated to this strategy.
                   Sums to 1.0 across all live strategies.
    current_exposure_mult: scalar applied on top of target_weight, e.g. 0.5 in
                           crisis regime to cut total exposure.
    paper_mode: True means intents are logged but not submitted to OMS.
    performance_override: per-strategy multiplier for ad-hoc downsize without
                          waiting for a rebalance (e.g. drawdown alarm hit).
    """

    strategy_id: str
    target_weight: float
    current_exposure_mult: float = 1.0
    paper_mode: bool = False
    performance_override: float = 1.0

    def __post_init__(self) -> None:
        assert 0.0 <= self.target_weight <= 1.0, (
            f"target_weight must be in [0,1], got {self.target_weight}"
        )
        assert 0.0 <= self.current_exposure_mult <= 1.0, (
            f"current_exposure_mult must be in [0,1], got {self.current_exposure_mult}"
        )
        assert 0.0 <= self.performance_override <= 1.0, (
            f"performance_override must be in [0,1], got {self.performance_override}"
        )

    @property
    def effective_scale(self) -> float:
        """Composite scale applied to a strategy's intents in process_intents."""
        scale = (
            self.target_weight
            * self.current_exposure_mult
            * self.performance_override
        )
        assert 0.0 <= scale <= 1.0, "effective_scale invariant violated"
        return scale


@dataclass
class PortfolioConstraints:
    """Hard portfolio-level limits applied in _apply_portfolio_constraints.

    Defaults sourced from reference/06_portfolio.md and ARCHITECTURE.md.
    Each value is a deliberate risk budget choice, NOT a regulatory or broker
    limit — intentionally tighter than OANDA retail leverage.
    """

    # Sum |notional| / equity. 3x means 30% margin if all positions max out,
    # leaves 70% headroom to absorb adverse moves before a margin call.
    max_gross_leverage: float = 3.0

    # |Sum notional| / equity. Limits directional exposure when strategies
    # happen to align (most days they are partially offsetting).
    max_net_leverage: float = 2.0

    # Operational ceiling — beyond ~10 active pairs the operator can no longer
    # hold the book in their head when manually reviewing dashboards.
    max_total_positions: int = 10

    # Single-pair concentration cap as fraction of equity. At 35%, a single bad
    # pair cannot lose more than ~10% of equity assuming a 30% adverse move.
    max_notional_per_pair_pct: float = 0.35

    # Per-CCY exposure cap (e.g. all USD-funded longs ≤ 40%). Distinct from
    # per-pair cap: a basket of EUR longs through different counter-currencies
    # can still concentrate EUR risk.
    max_directional_exposure_per_currency: float = 0.40

    # Diversification floor: avoid 5+ trades all betting the same regime story.
    max_concurrent_same_direction: int = 5

    # Annualized portfolio vol target (10%) and ceiling (12%). Risk parity
    # rescales weights to hit target; ceiling is a soft alarm boundary.
    max_portfolio_vol: float = 0.12
    portfolio_vol_target: float = 0.10

    def __post_init__(self) -> None:
        assert self.max_gross_leverage > 0, "max_gross_leverage must be positive"
        assert self.max_net_leverage > 0, "max_net_leverage must be positive"
        assert self.max_net_leverage <= self.max_gross_leverage, (
            "max_net_leverage cannot exceed max_gross_leverage"
        )
        assert self.max_notional_per_pair_pct > 0, (
            "max_notional_per_pair_pct must be positive"
        )
        assert self.max_directional_exposure_per_currency > 0, (
            "max_directional_exposure_per_currency must be positive"
        )
        assert self.portfolio_vol_target > 0, "portfolio_vol_target must be positive"
        assert self.max_portfolio_vol >= self.portfolio_vol_target, (
            "max_portfolio_vol must be >= portfolio_vol_target"
        )
        assert self.max_total_positions >= 1, "max_total_positions must be >= 1"
        assert self.max_concurrent_same_direction >= 1, (
            "max_concurrent_same_direction must be >= 1"
        )


# =============================================================================
# PortfolioCoordinator
# =============================================================================


class PortfolioCoordinator:
    """Combines strategy intents, applies portfolio constraints, manages allocations.

    Wired into LiveEngine by D7 (CL-amf). Strategies emit OrderIntent; the
    coordinator scales, aggregates, constrains, and forwards the final per-symbol
    intent to OMS.

    Concurrency: process_intents holds an asyncio.Lock so concurrent signal-gen
    cycles cannot interleave aggregation and constraint application.
    """

    def __init__(
        self,
        strategies: list[Any],
        oms: OrderManager,
        broker: Broker,
        state: PortfolioStateProtocol,
        constraints: PortfolioConstraints | None = None,
    ) -> None:
        assert strategies, "PortfolioCoordinator requires at least one strategy"
        ids = [s.id for s in strategies]
        assert len(ids) == len(set(ids)), f"duplicate strategy ids: {ids}"

        self.strategies: dict[str, Any] = {s.id: s for s in strategies}
        self.oms = oms
        self.broker = broker
        self.state = state
        self.constraints = constraints or PortfolioConstraints()

        self.allocations: dict[str, StrategyAllocation] = {}
        self._last_rebalance: datetime | None = None
        self._intent_lock = asyncio.Lock()
        self._conflicts_log: list[dict[str, Any]] = []

        logger.info(
            "PortfolioCoordinator initialized with %d strategies: %s",
            len(self.strategies),
            sorted(self.strategies.keys()),
        )

    # ------------------------------------------------------------------
    # Allocation initialization
    # ------------------------------------------------------------------

    def initialize_allocations(self, initial_weights: dict[str, float]) -> None:
        """Seed allocations from a pre-computed weight dict.

        Weights are normalized to sum to 1.0 across known strategies. Unknown
        strategies in the input are warned and skipped. Strategies not in the
        input start in paper mode at zero weight.
        """
        recognized: dict[str, float] = {}
        for sid, weight in initial_weights.items():
            if sid not in self.strategies:
                logger.warning("Initial weight for unknown strategy %s ignored", sid)
                continue
            assert weight >= 0.0, f"weight for {sid} must be non-negative, got {weight}"
            recognized[sid] = weight

        total = sum(recognized.values())
        if total <= 0:
            n = len(self.strategies)
            for sid in self.strategies:
                self.allocations[sid] = StrategyAllocation(
                    strategy_id=sid, target_weight=1.0 / n,
                )
            logger.warning("No positive initial weights; defaulting to equal weight")
            return

        for sid, weight in recognized.items():
            self.allocations[sid] = StrategyAllocation(
                strategy_id=sid, target_weight=weight / total,
            )

        for sid in self.strategies:
            if sid not in self.allocations:
                self.allocations[sid] = StrategyAllocation(
                    strategy_id=sid,
                    target_weight=0.0,
                    current_exposure_mult=0.0,
                    paper_mode=True,
                )

        logger.info(
            "Initialized allocations: %s",
            {sid: round(a.target_weight, 4) for sid, a in self.allocations.items()},
        )

    # ------------------------------------------------------------------
    # Intent processing — public entry point
    # ------------------------------------------------------------------

    async def process_intents(
        self,
        raw_intents: dict[str, list[OrderIntent]],
    ) -> dict[str, dict[str, Any]]:
        """Process intents from all strategies and forward final orders to OMS.

        Returns the final aggregated, constrained intents per symbol so callers
        can persist or surface them for observability.
        """
        async with self._intent_lock:
            scaled = self._scale_intents(raw_intents)
            aggregated = self._aggregate_by_symbol(scaled)
            feasible = self._apply_portfolio_constraints(aggregated)

            ts = datetime.now(UTC)
            for symbol, info in feasible.items():
                with LogContext(symbol=symbol):
                    final = OrderIntent(
                        strategy_id="portfolio",
                        symbol=symbol,
                        target_position=info["target_position"],
                        urgency=info["urgency"],
                    )
                    logger.info(
                        "Submitting %s target=%.4f contributions=%s",
                        symbol,
                        info["target_position"],
                        info["strategy_contributions"],
                    )
                    self.oms.submit_intent(final)
                    self.state.record_portfolio_order(
                        ts,
                        symbol,
                        info["target_position"],
                        dict(info["strategy_contributions"]),
                    )

            return feasible

    # ------------------------------------------------------------------
    # Intent processing — internal stages
    # ------------------------------------------------------------------

    def _scale_intents(
        self,
        raw_intents: dict[str, list[OrderIntent]],
    ) -> list[OrderIntent]:
        """Scale each strategy's intents by its allocation; drop paper + unknown."""
        out: list[OrderIntent] = []
        for sid, intents in raw_intents.items():
            if sid not in self.allocations:
                logger.warning("Intent from unknown strategy %s ignored", sid)
                continue

            alloc = self.allocations[sid]

            if alloc.paper_mode:
                for intent in intents:
                    logger.info(
                        "[PAPER] %s: %s target=%.4f",
                        sid,
                        intent.symbol,
                        intent.target_position,
                    )
                continue

            scale = alloc.effective_scale
            for intent in intents:
                out.append(
                    OrderIntent(
                        strategy_id=intent.strategy_id,
                        symbol=intent.symbol,
                        target_position=intent.target_position * scale,
                        urgency=intent.urgency,
                        max_slippage_bps=intent.max_slippage_bps,
                    )
                )
        return out

    def _aggregate_by_symbol(
        self,
        intents: list[OrderIntent],
    ) -> dict[str, dict[str, Any]]:
        """Net intents on the same symbol; escalate urgency to most urgent contributor.

        Conflicts (opposite signs on the same symbol) are recorded in
        self._conflicts_log but allowed to net.
        """
        by_symbol: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "target_position": 0.0,
                "urgency": "passive",
                "strategy_contributions": {},
            }
        )

        for intent in intents:
            agg = by_symbol[intent.symbol]
            agg["target_position"] += intent.target_position
            agg["strategy_contributions"][intent.strategy_id] = intent.target_position
            current_rank = _URGENCY_RANK.get(agg["urgency"], 0)
            new_rank = _URGENCY_RANK.get(intent.urgency, 0)
            if new_rank > current_rank:
                agg["urgency"] = intent.urgency

        for symbol, agg in by_symbol.items():
            contribs = agg["strategy_contributions"]
            signs = {int(np.sign(v)) for v in contribs.values() if v != 0}
            if len(signs) > 1:
                self._conflicts_log.append(
                    {
                        "ts": datetime.now(UTC),
                        "symbol": symbol,
                        "contributions": dict(contribs),
                        "net": agg["target_position"],
                    }
                )
                logger.info(
                    "Conflict on %s: net=%.4f contribs=%s",
                    symbol,
                    agg["target_position"],
                    contribs,
                )

        return dict(by_symbol)

    def _apply_portfolio_constraints(
        self,
        aggregated: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Apply gross leverage cap, per-pair cap; warn on per-currency exceedance.

        Mutates aggregated in place AND returns it. Per-strategy contributions
        are scaled in lockstep with the symbol-level scaling so attribution
        downstream remains accurate.

        Per-currency exposure is warn-only here; the pre-trade gate (CL-srsy)
        will reject orders that would breach per-currency caps once implemented.
        """
        if not aggregated:
            return aggregated

        account = self.broker.get_account()
        equity = account.equity
        assert equity > 0, (
            f"broker reported non-positive equity {equity}; cannot compute leverage"
        )

        # 1) Gross leverage cap.
        gross_notional = sum(
            abs(a["target_position"]) * self._get_price(sym)
            for sym, a in aggregated.items()
        )
        gross_leverage = gross_notional / equity
        if gross_leverage > self.constraints.max_gross_leverage:
            scale = self.constraints.max_gross_leverage / gross_leverage
            logger.warning(
                "Gross leverage %.2fx exceeds max %.2fx — scaling all positions by %.4f",
                gross_leverage,
                self.constraints.max_gross_leverage,
                scale,
            )
            self._rescale_all(aggregated, scale)

        # 2) Per-pair cap.
        max_pair_notional = equity * self.constraints.max_notional_per_pair_pct
        for symbol, agg in aggregated.items():
            price = self._get_price(symbol)
            notional = abs(agg["target_position"]) * price
            if notional > max_pair_notional:
                scale = max_pair_notional / notional
                logger.warning(
                    "%s notional %.2f exceeds per-pair cap %.2f — scaling by %.4f",
                    symbol,
                    notional,
                    max_pair_notional,
                    scale,
                )
                self._rescale_symbol(agg, scale)

        # 3) Per-currency exposure (warn-only).
        currency_exposure = self._compute_currency_exposure(aggregated)
        max_ccy_exposure = equity * self.constraints.max_directional_exposure_per_currency
        for ccy, exposure in currency_exposure.items():
            if abs(exposure) > max_ccy_exposure:
                logger.warning(
                    "Currency %s net exposure %.2f exceeds cap %.2f",
                    ccy,
                    exposure,
                    max_ccy_exposure,
                )

        return aggregated

    @staticmethod
    def _rescale_all(aggregated: dict[str, dict[str, Any]], scale: float) -> None:
        for agg in aggregated.values():
            PortfolioCoordinator._rescale_symbol(agg, scale)

    @staticmethod
    def _rescale_symbol(agg: dict[str, Any], scale: float) -> None:
        agg["target_position"] *= scale
        for sid in agg["strategy_contributions"]:
            agg["strategy_contributions"][sid] *= scale

    def _compute_currency_exposure(
        self,
        aggregated: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Sum notional exposure per currency. EUR/USD long → +EUR, -USD."""
        exposures: dict[str, float] = defaultdict(float)
        for symbol, agg in aggregated.items():
            assert len(symbol) >= 6, (
                f"symbol {symbol!r} too short to extract currency pair (need 6 chars)"
            )
            base, quote = symbol[:3], symbol[3:6]
            notional = agg["target_position"] * self._get_price(symbol)
            exposures[base] += notional
            exposures[quote] -= notional
        return dict(exposures)

    def _get_price(self, symbol: str) -> float:
        """Mid-price from broker. Falls back to 1.0 with WARN if broker fails.

        Fallback behavior is a safety net; in practice price stream availability
        is enforced by the live engine's price_stream_task before signals fire.
        """
        try:
            bid, ask = self.broker.get_price(symbol)
            mid = (bid + ask) / 2
            assert mid > 0, f"non-positive mid for {symbol}: bid={bid} ask={ask}"
            return mid
        except Exception:
            logger.exception("Could not fetch price for %s; using 1.0 fallback", symbol)
            return 1.0

    # ------------------------------------------------------------------
    # Rebalance — risk parity + correlation regime
    # ------------------------------------------------------------------

    async def rebalance_allocations(self, force: bool = False) -> None:
        """Recompute risk parity weights from recent strategy returns.

        Idempotent for rapid invocations: skips if last rebalance was within
        _REBALANCE_MIN_INTERVAL_DAYS unless force=True.
        """
        now = datetime.now(UTC)
        if (
            not force
            and self._last_rebalance is not None
            and (now - self._last_rebalance).days < _REBALANCE_MIN_INTERVAL_DAYS
        ):
            logger.debug(
                "Rebalance skipped: last %s days ago",
                (now - self._last_rebalance).days,
            )
            return

        live_ids = [sid for sid, a in self.allocations.items() if not a.paper_mode]
        if not live_ids:
            logger.warning("No live strategies; skipping rebalance")
            return

        returns_df = self.state.load_strategy_returns_history(
            live_ids, _RISK_PARITY_LOOKBACK_DAYS,
        )

        if len(returns_df) < _RISK_PARITY_MIN_HISTORY_DAYS:
            logger.warning(
                "Insufficient history (%d rows < %d); falling back to equal weight",
                len(returns_df),
                _RISK_PARITY_MIN_HISTORY_DAYS,
            )
            new_weights = {sid: 1.0 / len(live_ids) for sid in live_ids}
            corr_regime: dict[str, Any] = {
                "regime": "unknown",
                "exposure_multiplier": _CORR_REGIME_EXPOSURE_NORMAL,
                "avg_corr": 0.0,
                "z": 0.0,
            }
        else:
            new_weights = self._compute_risk_parity(returns_df)
            corr_regime = self._check_correlation_regime(returns_df)

        for sid, weight in new_weights.items():
            if sid not in self.allocations:
                continue
            alloc = self.allocations[sid]
            old_weight = alloc.target_weight
            alloc.target_weight = weight
            alloc.current_exposure_mult = corr_regime["exposure_multiplier"]
            logger.info(
                "Reallocated %s: %.4f → %.4f (exposure=%.2f, regime=%s)",
                sid,
                old_weight,
                weight,
                alloc.current_exposure_mult,
                corr_regime["regime"],
            )

        self._last_rebalance = now
        self.state.record_reallocation(now, dict(new_weights), corr_regime)

    @staticmethod
    def _compute_risk_parity(returns: pd.DataFrame) -> dict[str, float]:
        """Solve for weights where each strategy contributes equal portfolio risk.

        SLSQP on the mean-deviation objective with per-strategy bounds and a
        sum-to-one equality. D2 (CL-6tu) will replace this with the shared
        src/portfolio/risk_parity.py module.
        """
        cov = returns.cov().values * _TRADING_DAYS_PER_YEAR
        n = len(cov)
        strategy_ids = returns.columns.tolist()

        def objective(w: np.ndarray[Any, Any]) -> float:
            port_vol = float(np.sqrt(w @ cov @ w))
            if port_vol == 0:
                return 0.0
            marginal = cov @ w
            rc = w * marginal / port_vol
            return float(np.sum((rc - np.mean(rc)) ** 2))

        x0 = np.ones(n) / n
        bounds = [_RISK_PARITY_BOUNDS_PER_STRATEGY] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            logger.warning(
                "Risk parity SLSQP failed: %s; falling back to equal weight",
                result.message,
            )
            return {sid: 1.0 / n for sid in strategy_ids}

        # Renormalize to sum to exactly 1.0 (SLSQP equality is satisfied to ~1e-8).
        raw = {sid: float(x) for sid, x in zip(strategy_ids, result.x, strict=True)}
        total = sum(raw.values())
        assert total > 0, "risk parity returned non-positive weight sum"
        return {sid: w / total for sid, w in raw.items()}

    @staticmethod
    def _check_correlation_regime(returns: pd.DataFrame) -> dict[str, Any]:
        """Detect stressed/crisis regime via recent-vs-baseline strategy correlation.

        Returns dict with keys: regime ("normal"|"stressed"|"crisis"),
        exposure_multiplier (1.0|0.75|0.5), avg_corr (recent), z.
        """
        ncols = len(returns.columns)
        if ncols < 2:
            # Single-strategy correlation is undefined.
            return {
                "regime": "normal",
                "exposure_multiplier": _CORR_REGIME_EXPOSURE_NORMAL,
                "avg_corr": 0.0,
                "z": 0.0,
            }
        if len(returns) < _CORR_REGIME_RECENT_WINDOW * 2:
            return {
                "regime": "normal",
                "exposure_multiplier": _CORR_REGIME_EXPOSURE_NORMAL,
                "avg_corr": 0.0,
                "z": 0.0,
            }

        recent = returns.iloc[-_CORR_REGIME_RECENT_WINDOW:]
        baseline_end = -_CORR_REGIME_RECENT_WINDOW
        baseline_start = -(_CORR_REGIME_BASELINE_WINDOW + _CORR_REGIME_RECENT_WINDOW)
        baseline = (
            returns.iloc[baseline_start:baseline_end]
            if len(returns) >= _CORR_REGIME_BASELINE_WINDOW + _CORR_REGIME_RECENT_WINDOW
            else returns.iloc[:baseline_end]
        )

        mask = np.triu(np.ones((ncols, ncols), dtype=bool), k=1)
        recent_avg = float(recent.corr().where(mask).stack().mean())
        baseline_avg = float(baseline.corr().where(mask).stack().mean())

        # Bootstrap baseline σ from rolling windows of the baseline period.
        rolling_corrs: list[float] = []
        for i in range(_CORR_REGIME_RECENT_WINDOW, len(baseline)):
            window = baseline.iloc[i - _CORR_REGIME_RECENT_WINDOW : i]
            c = window.corr().where(mask).stack().mean()
            if not pd.isna(c):
                rolling_corrs.append(float(c))

        # 0.1 default σ if no rolling history: a conservative spread that won't
        # trigger crisis from noise alone (matches reference doc behavior).
        std = float(np.std(rolling_corrs)) if rolling_corrs else 0.1
        z = (recent_avg - baseline_avg) / (std + 1e-9)

        if z > _CORR_REGIME_CRISIS_Z:
            return {
                "regime": "crisis",
                "exposure_multiplier": _CORR_REGIME_EXPOSURE_CRISIS,
                "avg_corr": recent_avg,
                "z": z,
            }
        if z > _CORR_REGIME_STRESSED_Z:
            return {
                "regime": "stressed",
                "exposure_multiplier": _CORR_REGIME_EXPOSURE_STRESSED,
                "avg_corr": recent_avg,
                "z": z,
            }
        return {
            "regime": "normal",
            "exposure_multiplier": _CORR_REGIME_EXPOSURE_NORMAL,
            "avg_corr": recent_avg,
            "z": z,
        }

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def add_strategy(self, strategy: Any, initial_paper_days: int = 30) -> None:
        """Add a strategy in paper mode at zero weight.

        Promotion to live happens via the lifecycle workflow (D5 / CL-6vv) and
        the upcoming allocation policy (CL-15r4).
        """
        sid = strategy.id
        if sid in self.strategies:
            raise ValueError(f"Strategy {sid} already registered")
        assert initial_paper_days >= 0, "initial_paper_days must be non-negative"

        self.strategies[sid] = strategy
        self.allocations[sid] = StrategyAllocation(
            strategy_id=sid,
            target_weight=0.0,
            current_exposure_mult=0.0,
            paper_mode=True,
        )
        logger.info(
            "Added %s in paper mode for %d days",
            sid,
            initial_paper_days,
        )

    def remove_strategy(self, strategy_id: str) -> None:
        """Liquidate attributed positions then remove from registry.

        Triggers a forced rebalance so the freed weight is redistributed.
        """
        if strategy_id not in self.strategies:
            logger.warning("remove_strategy: %s not registered, no-op", strategy_id)
            return

        positions = self.state.get_positions_by_strategy(strategy_id)
        for pos in positions:
            self.oms.submit_intent(
                OrderIntent(
                    strategy_id=f"removal-{strategy_id}",
                    symbol=pos.symbol,
                    target_position=0.0,
                    urgency="normal",
                )
            )

        del self.strategies[strategy_id]
        del self.allocations[strategy_id]
        logger.info(
            "Removed strategy %s; %d positions liquidated",
            strategy_id,
            len(positions),
        )

        # Forced rebalance to redistribute freed weight. Schedule on the running
        # loop if available; otherwise leave the weight in place until the next
        # scheduled rebalance.
        try:
            asyncio.get_running_loop()
            asyncio.create_task(self.rebalance_allocations(force=True))
        except RuntimeError:
            logger.debug(
                "No running event loop; rebalance after %s removal deferred",
                strategy_id,
            )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def conflicts_log(self) -> list[dict[str, Any]]:
        """Copy of detected intent conflicts (caller can persist or display)."""
        return list(self._conflicts_log)
