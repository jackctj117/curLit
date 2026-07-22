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

from src.data.economic_calendar import BlackoutAction, BlackoutEvaluator
from src.execution.broker import Broker, canonical_symbol, currency_pair
from src.execution.oms import OrderIntent, OrderManager, Urgency
from src.monitoring.logging_setup import LogContext
from src.monitoring.metrics import blackout_size_down
from src.portfolio.risk_parity import risk_parity_weights

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

# Urgency rank for escalation when multiple strategies share a symbol —
# derived from the canonical Urgency enum's declaration order (CL-ikz2),
# so the vocabulary has ONE source of truth at the OMS boundary. Unknown
# values fall back to rank 0 in the lookups below and never escalate.
_URGENCY_RANK: dict[str, int] = {u.value: rank for rank, u in enumerate(Urgency)}


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
        pre_trade_validator: Any | None = None,
        blackout_evaluator: BlackoutEvaluator | None = None,
    ) -> None:
        assert strategies, "PortfolioCoordinator requires at least one strategy"
        ids = [s.id for s in strategies]
        assert len(ids) == len(set(ids)), f"duplicate strategy ids: {ids}"

        self.strategies: dict[str, Any] = {s.id: s for s in strategies}
        self.oms = oms
        self.broker = broker
        self.state = state
        self.constraints = constraints or PortfolioConstraints()
        self.pre_trade_validator = pre_trade_validator
        # Blackout evaluator handles SIZE_DOWN_50PCT here at the coordinator
        # level (intent mutation); PAUSE/EXIT_FLAT are handled separately by
        # the validator (rejection only). Splitting by responsibility:
        # validator's contract is reject-or-accept, coordinator owns intent
        # construction so it can mutate target_position cleanly.
        self.blackout_evaluator = blackout_evaluator

        self.allocations: dict[str, StrategyAllocation] = {}
        self._last_rebalance: datetime | None = None
        self._intent_lock = asyncio.Lock()
        self._conflicts_log: list[dict[str, Any]] = []
        # Cross-tick per-strategy target memory (CL-8lv6 P0): OMS computes
        # delta against the ABSOLUTE broker position, but strategies only
        # express their own slice — and book-based strategies don't re-emit
        # every tick. Without remembering each strategy's last target, one
        # strategy's exit-to-0 (or lone rebalance) on a shared symbol would
        # flatten EVERYONE's position. {strategy_id: {canonical_symbol:
        # last scaled target}}; zeros pruned after aggregation; seeded from
        # strategy open_positions books on the first tick after a restart.
        self._strategy_targets: dict[str, dict[str, float]] = {}
        self._targets_seeded = False
        # Strong references to fire-and-forget rebalance tasks (CL-xdnh):
        # bare create_task results were previously dropped, so the tasks
        # could be garbage-collected mid-flight and any exception vanished
        # ("Task exception was never retrieved" at best). Each task is
        # retained here and observed via _on_background_task_done.
        self._background_tasks: set[asyncio.Task[Any]] = set()

        logger.info(
            "PortfolioCoordinator initialized with %d strategies: %s "
            "(pre_trade_validator=%s, blackout_evaluator=%s)",
            len(self.strategies),
            sorted(self.strategies.keys()),
            "enabled" if pre_trade_validator is not None else "disabled",
            "enabled" if blackout_evaluator is not None else "disabled",
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
        can persist or surface them for observability. Pre-trade rejections
        are logged via the validator and do NOT appear in the returned dict.
        """
        async with self._intent_lock:
            if not self._targets_seeded:
                self._seed_targets_from_books()
                self._targets_seeded = True
            scaled = self._scale_intents(raw_intents)
            aggregated = self._aggregate_by_symbol(scaled)
            # Blocking broker I/O (get_account + get_price per symbol) runs
            # off-loop (CL-xdnh): with the httpx-based OANDA broker these are
            # synchronous HTTP round-trips that froze the event loop — and
            # with it the price stream, health ticks, and kill switches —
            # for the duration of each call.
            feasible = await asyncio.to_thread(
                self._apply_portfolio_constraints, aggregated,
            )

            # Cross-tick memory updates AFTER constraints (CL-8cw1 P1):
            # contributions here are post-leverage-cut. Remembering the
            # PRE-constraint values made later ticks re-expand toward the
            # unconstrained size, silently undoing risk cuts. Symbols the
            # constraints dropped (unpriceable) keep last tick's memory —
            # the broker didn't change, so neither did the truth.
            for canon, info in feasible.items():
                for sid, tgt in info["strategy_contributions"].items():
                    self._strategy_targets.setdefault(sid, {})[canon] = tgt
            for sid in list(self._strategy_targets):
                self._strategy_targets[sid] = {
                    k: v for k, v in self._strategy_targets[sid].items() if v
                }
                if not self._strategy_targets[sid]:
                    del self._strategy_targets[sid]

            ts = datetime.now(UTC)
            # Pre-trade rejections drop the offending intent from `feasible` so
            # the returned dict reflects what actually went to OMS.
            accepted: dict[str, dict[str, Any]] = {}
            current_positions: list[Any] | None = None
            if self.pre_trade_validator is not None:
                # Single broker query reused across all per-symbol checks.
                current_positions = await asyncio.to_thread(
                    self.broker.get_positions,
                )

            for symbol, info in feasible.items():
                # Aggregation keys are canonical; ROUTE with the original
                # dialect the strategy spoke (CL-8cw1) — brokers and the
                # pricing path expect it.
                route_symbol = str(info.get("symbol") or symbol)
                with LogContext(symbol=route_symbol):
                    target = float(info["target_position"])
                    # Blackout SIZE_DOWN_50PCT — halve the intent BEFORE
                    # passing to the validator. PAUSE/EXIT_FLAT cases are
                    # rejected by the validator separately (CL-c8th); we
                    # only mutate here for the size-down case.
                    target = self._apply_blackout_size_down(route_symbol, target)

                    final = OrderIntent(
                        strategy_id="portfolio",
                        symbol=route_symbol,
                        target_position=target,
                        urgency=info["urgency"],
                    )

                    if self.pre_trade_validator is not None:
                        # validate() hits the broker for prices/account —
                        # blocking HTTP, keep it off the loop (CL-xdnh).
                        rejection = await asyncio.to_thread(
                            self.pre_trade_validator.validate,
                            final, current_positions=current_positions,
                        )
                        if rejection is not None:
                            # Validator already logged + recorded metric.
                            continue

                    logger.info(
                        "Submitting %s target=%.4f contributions=%s",
                        symbol,
                        info["target_position"],
                        info["strategy_contributions"],
                    )
                    # submit_intent does broker get_positions + place_order
                    # (sync HTTP, plus RejectionHandler retry sleeps) —
                    # off-loop via to_thread (CL-xdnh). Called through
                    # to_thread rather than OrderManager.submit_intent_async
                    # so OMS doubles that only implement submit_intent keep
                    # working. When the validator already fetched a positions
                    # snapshot, reuse it (CL-a0sv) — post-aggregation there is
                    # one intent per symbol, so the snapshot stays
                    # delta-accurate across the batch.
                    if current_positions is not None:
                        await asyncio.to_thread(
                            self.oms.submit_intent, final,
                            positions=current_positions,
                        )
                    else:
                        await asyncio.to_thread(self.oms.submit_intent, final)
                    # DB write — also blocking I/O.
                    await asyncio.to_thread(
                        self.state.record_portfolio_order,
                        ts,
                        symbol,
                        info["target_position"],
                        dict(info["strategy_contributions"]),
                    )
                    accepted[symbol] = info

            return accepted

    def _apply_blackout_size_down(
        self, symbol: str, target_position: float,
    ) -> float:
        """Halve target_position when the calendar is in SIZE_DOWN_50PCT.

        FULL_SIZE / PAUSE_NEW_ENTRIES / EXIT_FLAT are not handled here:
        FULL_SIZE is a no-op, the other two are rejected by the validator.

        Currency derived via currency_pair() (CL-rybp) — canonicalized, so
        OANDA-form 'EUR_USD' and compact 'EURUSD' both yield 'EUR'; same
        convention as PreTradeValidator._currency_from_symbol.
        """
        if self.blackout_evaluator is None:
            return target_position
        pair = currency_pair(symbol)
        currency = pair[0] if pair else None
        decision = self.blackout_evaluator.evaluate(
            now=datetime.now(UTC), currency=currency,
        )
        if decision.action != BlackoutAction.SIZE_DOWN_50PCT:
            return target_position
        # Mutate to half size, log + emit metric.
        new_target = target_position * 0.5
        logger.info(
            "Blackout SIZE_DOWN_50PCT for %s — halving target %.4f → %.4f (%s)",
            symbol, target_position, new_target, decision.reason,
        )
        try:
            blackout_size_down.labels(pair=symbol).inc()
        except Exception:
            logger.exception("blackout_size_down metric increment failed")
        return new_target

    # ------------------------------------------------------------------
    # Intent processing — internal stages
    # ------------------------------------------------------------------

    def _seed_targets_from_books(self) -> None:
        """Rebuild cross-tick target memory from strategy books after a
        restart (CL-8lv6). Book-based strategies (event_driven) hold
        positions without re-emitting intents; until they emit again, their
        share would be invisible to aggregation and any other strategy's
        trade on the same symbol would stomp it. Self-sized books store
        broker-scale quantities, so the seed is exact for them; scaled
        strategies re-emit every tick anyway."""
        for sid, strategy in self.strategies.items():
            book = getattr(strategy, "open_positions", None)
            if not isinstance(book, dict):
                continue
            for sym, pos in book.items():
                qty = (
                    pos.get("quantity") if isinstance(pos, dict)
                    else getattr(pos, "quantity", None)
                )
                if qty:
                    self._strategy_targets.setdefault(sid, {})[
                        canonical_symbol(sym)
                    ] = float(qty)
        if self._strategy_targets:
            logger.info(
                "Seeded cross-tick targets from books: %s",
                {sid: dict(t) for sid, t in self._strategy_targets.items()},
            )

    def _scale_intents(
        self,
        raw_intents: dict[str, list[OrderIntent]],
    ) -> list[OrderIntent]:
        """Scale each strategy's intents by its allocation; drop paper + unknown.

        Strategies that declare ``self_sized = True`` (event_driven — sizes
        by risk internally, 50bps at the stop) pass through UNSCALED
        (CL-8lv6): applying the allocation weight on top double-applied
        sizing — the broker held 1/n of what the strategy's book recorded,
        which is exactly the persistent boot-time size_mismatch.
        """
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

            if getattr(self.strategies.get(sid), "self_sized", False):
                scale = 1.0
            else:
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
        # Keyed by CANONICAL symbol (CL-8cw1 P0): the live portfolio mixes
        # dialects (event legs EUR_USD, rate-diff EURUSD) — raw-symbol keys
        # produced TWO aggregate rows for one economic pair, each re-merging
        # remembered shares → double OMS delta. "symbol" carries the
        # first-seen original dialect for ROUTING/pricing.
        by_symbol: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "symbol": None,
                "target_position": 0.0,
                "urgency": Urgency.PASSIVE.value,
                "strategy_contributions": {},
            }
        )

        for intent in intents:
            canon = canonical_symbol(intent.symbol)
            agg = by_symbol[canon]
            if agg["symbol"] is None:
                agg["symbol"] = intent.symbol
            agg["target_position"] += intent.target_position
            agg["strategy_contributions"][intent.strategy_id] = intent.target_position
            current_rank = _URGENCY_RANK.get(agg["urgency"], 0)
            new_rank = _URGENCY_RANK.get(intent.urgency, 0)
            if new_rank > current_rank:
                agg["urgency"] = intent.urgency

        # Cross-tick merge (CL-8lv6 P0): the OMS deltas against the ABSOLUTE
        # broker position, so a touched symbol's aggregate must include every
        # OTHER strategy's remembered share — otherwise one strategy's
        # exit-to-0 closes everyone's position on that symbol. Memory holds
        # LAST-TICK (post-constraint) values; this tick's speakers were
        # counted above from their fresh intents (CL-8cw1).
        for canon, agg in by_symbol.items():
            for sid, remembered in self._strategy_targets.items():
                if sid in agg["strategy_contributions"]:
                    continue  # spoke this tick — already counted
                held = remembered.get(canon)
                if held:
                    agg["target_position"] += held
                    agg["strategy_contributions"][sid] = held
                    logger.debug(
                        "Aggregation on %s includes %s's remembered "
                        "target %.4f", canon, sid, held,
                    )

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

        # FAIL CLOSED on missing prices (CL-e8ze, corrected per ultrareview
        # #1). The original fell back to mid=1.0, understating notional by
        # orders of magnitude. The FIRST fix rescaled the target to 0 — which
        # was its own bug: a zero-target intent still flowed to the OMS, whose
        # delta math (0 - current_qty, positions come from a DIFFERENT
        # endpoint than /pricing) would FORCE-LIQUIDATE a held position on a
        # mere pricing flap — carry_vol re-emits held-currency targets every
        # rebalance, so this was live exposure. "Fail closed" means REFUSE TO
        # ACT: delete the symbol from the aggregate entirely so no intent —
        # entry OR fabricated exit — reaches the OMS this cycle.
        unpriceable = [
            symbol for symbol, agg in aggregated.items()
            if agg.get("target_position")
            and self._get_price(str(agg.get("symbol") or symbol)) is None
        ]
        for symbol in unpriceable:
            logger.error(
                "price unavailable for %s — WITHHOLDING its intent entirely "
                "this cycle (fail closed: cannot validate risk without a "
                "price; existing position, if any, is left untouched)",
                symbol,
            )
            del aggregated[symbol]
        if not aggregated:
            return aggregated

        account = self.broker.get_account()
        equity = account.equity
        assert equity > 0, (
            f"broker reported non-positive equity {equity}; cannot compute leverage"
        )

        # 1) Gross leverage cap.
        gross_notional = sum(
            abs(a["target_position"])
            * (self._get_price(str(a.get("symbol") or sym)) or 0.0)
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
            price = self._get_price(str(agg.get("symbol") or symbol)) or 0.0
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
            pair = currency_pair(symbol)
            if pair is None:
                # Not an FX pair (index CFD, malformed) — skipping is honest;
                # slicing raw fabricated legs like '_US' / 'X50' (CL-rybp).
                logger.debug(
                    "currency exposure: %r is not an FX pair — skipped", symbol,
                )
                continue
            base, quote = pair
            notional = agg["target_position"] * (
                self._get_price(str(agg.get("symbol") or symbol)) or 0.0
            )
            exposures[base] += notional
            exposures[quote] -= notional
        return dict(exposures)

    def _get_price(self, symbol: str) -> float | None:
        """Mid-price from broker, or None when unavailable (CL-e8ze).

        No 1.0 fallback: a fabricated price understates notional by orders of
        magnitude for gold/indices and lets leverage/concentration checks pass
        when they should reject. Callers must treat None as "cannot validate
        → drop/zero the target" (see _apply_portfolio_constraints pre-pass).
        """
        for candidate in dict.fromkeys((symbol, canonical_symbol(symbol))):
            try:
                bid, ask = self.broker.get_price(candidate)
                mid = (bid + ask) / 2
                assert mid > 0, (
                    f"non-positive mid for {candidate}: bid={bid} ask={ask}"
                )
                return mid
            except Exception:  # noqa: PERF203 — try next dialect (CL-8cw1)
                continue
        logger.error(
            "Could not fetch price for %s (either dialect) — treating as "
            "UNPRICEABLE", symbol,
        )
        return None

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

        # DB read (potentially 2y × n strategies of returns) — off-loop so
        # the rebalance tick never stalls the engine (CL-xdnh).
        returns_df = await asyncio.to_thread(
            self.state.load_strategy_returns_history,
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
        await asyncio.to_thread(
            self.state.record_reallocation, now, dict(new_weights), corr_regime,
        )

    @staticmethod
    def _compute_risk_parity(returns: pd.DataFrame) -> dict[str, float]:
        """Solve for weights where each strategy contributes equal portfolio risk.

        Delegates to src/portfolio/risk_parity.py. Returns sum-to-1 weights —
        exposure scaling is applied separately via current_exposure_mult.

        Bounds are widened automatically for small portfolios where the default
        (0.05, 0.40) would make sum-to-1 infeasible (e.g. n=2 → max sum 0.80).
        """
        n = len(returns.columns)
        bounds = PortfolioCoordinator._effective_risk_parity_bounds(n)
        weights_series = risk_parity_weights(
            returns,
            bounds=bounds,
            target_total_vol=None,
        )
        return {sid: float(w) for sid, w in weights_series.items()}

    @staticmethod
    def _effective_risk_parity_bounds(n: int) -> tuple[float, float]:
        """Pick per-strategy bounds that are feasible for n strategies summing to 1.

        Default (0.05, 0.40) targets a typical 4-6 strategy portfolio. For very
        small n where the defaults are infeasible (n=2 → max sum 0.80 < 1.0)
        the upper bound widens; for very large n where lower would over-saturate
        (n > 1/lower → lower budget > 1.0) the lower bound shrinks. Otherwise
        the defaults pass through unchanged.
        """
        lower, upper = _RISK_PARITY_BOUNDS_PER_STRATEGY
        if n <= 0:
            return (lower, upper)
        # Widen upper only when n*upper < 1.0 (genuinely infeasible).
        if n * upper < 1.0:
            upper = min(1.0 / n + 0.10, 1.0)
        # Tighten lower only when n*lower > 1.0 (genuinely infeasible).
        if n * lower > 1.0:
            lower = max(0.0, 0.5 / n)
        return (lower, upper)

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
    # Background tasks (CL-xdnh)
    # ------------------------------------------------------------------

    def _spawn_forced_rebalance(self, reason: str) -> asyncio.Task[Any] | None:
        """Schedule rebalance_allocations(force=True) on the running loop.

        Returns the retained task, or None when no loop is running (caller
        logs the deferral). The task reference is held in
        self._background_tasks and its outcome is ALWAYS observed — the old
        bare asyncio.create_task() dropped the reference, so a failing
        rebalance died silently (and the task object itself could be GC'd
        mid-flight).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return None
        task: asyncio.Task[Any] = asyncio.create_task(
            self.rebalance_allocations(force=True),
            name=f"forced-rebalance-{reason}",
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, task: asyncio.Task[Any]) -> None:
        """Observe a finished background task; log (never raise) on failure."""
        self._background_tasks.discard(task)
        if task.cancelled():
            logger.warning("Background task %s cancelled", task.get_name())
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "Background task %s failed", task.get_name(), exc_info=exc,
            )

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
        # Liquidate THIS strategy's share only (CL-8cw1 P1): a raw target=0
        # per symbol would flatten co-holders too — the OMS deltas against
        # the absolute broker position. Submit the sum of the OTHER
        # strategies' remembered shares instead, and clear this strategy's
        # memory so aggregation stops counting it.
        removed_targets = self._strategy_targets.pop(strategy_id, {})
        symbols = {pos.symbol for pos in positions} | {
            sym for sym in removed_targets
        }
        for symbol in symbols:
            canon = canonical_symbol(symbol)
            others_total = sum(
                remembered.get(canon, 0.0)
                for sid, remembered in self._strategy_targets.items()
            )
            self.oms.submit_intent(
                OrderIntent(
                    strategy_id=f"removal-{strategy_id}",
                    symbol=symbol,
                    target_position=others_total,
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

        # Forced rebalance to redistribute freed weight. Scheduled on the
        # running loop if available (retained + observed via
        # _spawn_forced_rebalance); otherwise the weight stays in place until
        # the next scheduled rebalance.
        if self._spawn_forced_rebalance(f"remove-{strategy_id}") is None:
            logger.debug(
                "No running event loop; rebalance after %s removal deferred",
                strategy_id,
            )

    def promote_strategy_to_live(
        self,
        strategy_id: str,
        initial_weight: float = 0.05,
    ) -> None:
        """Flip ``strategy_id`` from paper mode to live with ``initial_weight``.

        Redistributes from existing strategies proportionally so the sum
        of weights stays at 1.0. Triggers a forced rebalance to push the
        new target through risk parity. Caller (CL-15r4 AllocationPolicy)
        is responsible for deciding *whether* to promote and what the
        initial weight should be.
        """
        if strategy_id not in self.allocations:
            raise ValueError(f"Cannot promote {strategy_id}: not registered")
        alloc = self.allocations[strategy_id]
        if not alloc.paper_mode:
            logger.warning(
                "promote_strategy_to_live: %s already live, no-op",
                strategy_id,
            )
            return

        assert 0 < initial_weight <= 1, (
            f"initial_weight must be in (0, 1], got {initial_weight}"
        )

        # Scale existing live weights down to make room for the new strategy.
        live_ids = [
            sid for sid, a in self.allocations.items()
            if not a.paper_mode and sid != strategy_id
        ]
        if live_ids:
            existing_total = sum(self.allocations[s].target_weight for s in live_ids)
            if existing_total > 0:
                scale = (1.0 - initial_weight) / existing_total
                for sid in live_ids:
                    self.allocations[sid].target_weight *= scale

        alloc.paper_mode = False
        alloc.target_weight = initial_weight
        logger.info(
            "Promoted %s to live at %.1f%% — %d existing strategies rescaled",
            strategy_id, initial_weight * 100, len(live_ids),
        )

        if self._spawn_forced_rebalance(f"promote-{strategy_id}") is None:
            logger.debug(
                "No running event loop; rebalance after promotion deferred",
            )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def conflicts_log(self) -> list[dict[str, Any]]:
        """Copy of detected intent conflicts (caller can persist or display)."""
        return list(self._conflicts_log)
