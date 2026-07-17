"""Kill switch manager — automatic circuit breakers for risk protection.

CL-ep0c adds two self-contained switches:

* ``equity_trailing_stop`` — halt new trades once equity falls
  ``trailing_stop_pct`` below its persisted all-time peak, and STAY
  halted for ``trailing_stop_cooldown_days`` even across restarts
  (state in ``data/equity_trailing_stop_state.json``, atomic
  tmp+rename like ``polymarket_loss_caps``).
* ``open_position_correlation`` — pull recent daily closes for the
  symbols the broker actually holds and fire when the mean
  direction-adjusted pairwise return correlation says the book is one
  crowded bet (two shorts in correlated pairs ARE the same bet; a
  long+short in correlated pairs hedge).
"""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any

from src.monitoring.metrics import kill_switch_triggered

logger = logging.getLogger(__name__)


# CL-ep0c defaults — mirror the conservative profile in
# configs/risk_profile.yaml (KillSwitchConfig in risk_profile.py).
_DEFAULT_TRAILING_STOP_PCT: float = 0.10
_DEFAULT_TRAILING_STOP_COOLDOWN_DAYS: float = 7
_DEFAULT_OPEN_POSITION_CORR_THRESHOLD: float = 0.85
_DEFAULT_OPEN_POSITION_CORR_LOOKBACK_DAYS: int = 60
# Below this many overlapping daily returns the correlation estimate is
# noise — the open-position switch refuses to fire on thin data.
_MIN_CORR_OBSERVATIONS: int = 20

# Trailing-stop persistence. Lives under the gitignored ``data/``
# runtime tree next to the other engine state (polymarket_loss_caps).
_DEFAULT_TRAILING_STATE_PATH: Path = Path("data/equity_trailing_stop_state.json")
_TRAILING_STATE_VERSION: int = 1


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class KillSwitch:
    name: str
    condition: Callable[[dict[str, Any]], bool]
    action: str
    armed: bool = True


@dataclass
class EquityTrailingStop:
    """CL-ep0c equity-curve trailing stop with persistent peak + cooldown.

    Semantics:
      * The all-time equity peak persists in ``state_path`` (atomic
        tmp+rename JSON, mirroring ``polymarket_loss_caps``) so it
        survives restarts — a redeploy must NOT grant a fresh peak.
      * Trigger when ``(equity - peak) / peak <= -trailing_stop_pct``.
        Triggering starts a cooldown of ``cooldown_days``; the cooldown
        deadline persists too.
      * While the cooldown is active ``evaluate`` keeps returning True
        (idempotent — the manager re-fires at most once per day via its
        daily dedup, and immediately again after any restart) and logs
        when trading resumes. The manager's ``_triggered_today`` dedup
        is NOT the cooldown mechanism — this state file is.
      * The peak is frozen during cooldown and resets ONLY after the
        cooldown has expired AND a fresh equity mark has been observed:
        the first post-expiry ``evaluate`` call that receives a real
        equity value sets ``peak = equity`` and clears the cooldown.
        Without that reset the old peak would immediately re-trigger.
    """

    trailing_stop_pct: float = _DEFAULT_TRAILING_STOP_PCT
    cooldown_days: float = _DEFAULT_TRAILING_STOP_COOLDOWN_DAYS
    state_path: Path | None = _DEFAULT_TRAILING_STATE_PATH
    clock: Callable[[], datetime] = field(default=_utc_now)

    def __post_init__(self) -> None:
        if self.state_path is not None and not isinstance(self.state_path, Path):
            self.state_path = Path(self.state_path)
        self.peak_equity: float | None = None
        self.cooldown_until: datetime | None = None
        self._load()

    def evaluate(self, equity: float | None) -> bool:
        """One mark-to-market observation. Returns True while halted."""
        now = self.clock()
        if self.cooldown_until is not None:
            if now < self.cooldown_until:
                logger.warning(
                    "equity_trailing_stop: cooldown active — new trades stay "
                    "halted; trading resumes at %s (frozen peak=%s)",
                    self.cooldown_until.isoformat(), self.peak_equity,
                )
                return True
            if equity is None:
                # Expired, but no fresh mark yet — peak reset waits for
                # the next marked-to-market equity observation.
                logger.info(
                    "equity_trailing_stop: cooldown expired at %s — awaiting "
                    "a fresh equity mark before resetting the peak",
                    self.cooldown_until.isoformat(),
                )
                return False
            logger.warning(
                "equity_trailing_stop: cooldown expired at %s — peak reset "
                "%s -> %.2f; trading may resume",
                self.cooldown_until.isoformat(), self.peak_equity, float(equity),
            )
            self.peak_equity = float(equity)
            self.cooldown_until = None
            self._save()
            return False

        if equity is None:
            return False
        eq = float(equity)
        if eq <= 0:
            return False
        if self.peak_equity is None or eq > self.peak_equity:
            self.peak_equity = eq
            self._save()
            return False
        drawdown = (eq - self.peak_equity) / self.peak_equity
        if drawdown <= -self.trailing_stop_pct:
            self.cooldown_until = now + timedelta(days=self.cooldown_days)
            self._save()
            logger.critical(
                "equity_trailing_stop: equity %.2f is %.2f%% below peak %.2f "
                "(limit %.2f%%) — halting new trades; trading resumes at %s",
                eq, drawdown * 100, self.peak_equity,
                -self.trailing_stop_pct * 100, self.cooldown_until.isoformat(),
            )
            return True
        return False

    def _save(self) -> None:
        """Atomic (tmp+rename) JSON snapshot — same pattern as
        ``polymarket_loss_caps``."""
        path = self.state_path
        if path is None:
            return
        payload = {
            "version": _TRAILING_STATE_VERSION,
            "peak_equity": self.peak_equity,
            "cooldown_until": (
                self.cooldown_until.isoformat()
                if self.cooldown_until is not None else None
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True))
        os.replace(tmp, path)

    def _load(self) -> None:
        """Missing file -> fresh state; a corrupt file raises — a
        silently reset trailing stop (fresh peak, dropped cooldown) is
        worse than a crash at boot."""
        path = self.state_path
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text())
            if raw.get("version") != _TRAILING_STATE_VERSION:
                msg = f"unsupported state version {raw.get('version')!r}"
                raise ValueError(msg)
            peak = raw.get("peak_equity")
            until = raw.get("cooldown_until")
            self.peak_equity = float(peak) if peak is not None else None
            self.cooldown_until = (
                datetime.fromisoformat(until) if until is not None else None
            )
        except (ValueError, KeyError, TypeError) as exc:
            msg = (
                f"equity trailing-stop state at {path} is corrupt or "
                f"unreadable ({exc}) — refusing to start with a reset peak; "
                f"repair or remove the file"
            )
            raise ValueError(msg) from exc
        logger.info(
            "equity_trailing_stop: loaded state from %s (peak=%s cooldown_until=%s)",
            path, self.peak_equity, self.cooldown_until,
        )


@dataclass
class OpenPositionCorrelationEvaluator:
    """CL-ep0c self-contained open-position correlation check.

    Unlike ``portfolio_correlation_crisis`` / ``strategy_correlation_spike``
    (which rely on externally supplied context keys that nothing populates
    today), this evaluator pulls its own inputs: ``broker.get_positions()``
    for the held symbols and ``DataProvider.get_aligned_series`` for
    ``lookback_days`` of daily closes.

    Metric: mean direction-adjusted pairwise return correlation,
    ``mean(corr(a, b) * dir(a) * dir(b))`` over held pairs, where
    ``dir`` is +1 long / -1 short (net per symbol). Two shorts in
    positively correlated pairs adjust to +corr (same bet); a
    long+short pair adjusts to -corr (hedge). Fires when the mean
    exceeds ``threshold``.

    Never fires when: no data provider, fewer than two net positions,
    missing/thin price data (< ``min_observations`` overlapping daily
    returns), or no computable pair correlation.
    """

    broker: Any
    data_provider: Any | None = None
    threshold: float = _DEFAULT_OPEN_POSITION_CORR_THRESHOLD
    lookback_days: int = _DEFAULT_OPEN_POSITION_CORR_LOOKBACK_DAYS
    min_observations: int = _MIN_CORR_OBSERVATIONS
    clock: Callable[[], datetime] = field(default=_utc_now)

    def __post_init__(self) -> None:
        # Evidence from the most recent evaluation — read by
        # KillSwitchManager._execute_action to log the matrix when
        # reduce_50pct fires.
        self.last_matrix: Any | None = None
        self.last_directions: dict[str, int] = {}
        self.last_mean_adjusted: float | None = None

    def evaluate(self) -> bool:
        if self.data_provider is None:
            return False
        net_qty: dict[str, float] = {}
        for pos in self.broker.get_positions() or []:
            symbol = getattr(pos, "symbol", None)
            qty = getattr(pos, "quantity", None)
            if not symbol or qty is None:
                continue
            net_qty[symbol] = net_qty.get(symbol, 0.0) + float(qty)
        directions = {s: (1 if q > 0 else -1) for s, q in net_qty.items() if q != 0}
        if len(directions) < 2:
            return False

        end = self.clock()
        start = end - timedelta(days=self.lookback_days)
        df = self.data_provider.get_aligned_series(sorted(directions), start, end)
        if df is None or df.empty:
            logger.info(
                "open_position_correlation: no aligned price data for %s — skipping",
                sorted(directions),
            )
            return False
        held = [s for s in sorted(directions) if s in df.columns]
        if len(held) < 2:
            return False
        # ffill bridges sparse union indices across sources; dropna trims
        # the leading rows a symbol hasn't started yet.
        closes = df[held].sort_index().ffill().dropna()
        returns = closes.pct_change().dropna()
        if len(returns) < self.min_observations:
            logger.info(
                "open_position_correlation: only %d overlapping daily returns "
                "(< %d required) — skipping", len(returns), self.min_observations,
            )
            return False

        corr = returns.corr()
        adjusted: list[float] = []
        for a, b in combinations(held, 2):
            c = corr.loc[a, b]
            if c != c:  # noqa: PLR0124 — NaN check without importing pandas
                continue
            adjusted.append(float(c) * directions[a] * directions[b])
        if not adjusted:
            return False
        mean_adjusted = sum(adjusted) / len(adjusted)
        self.last_matrix = corr
        self.last_directions = {s: directions[s] for s in held}
        self.last_mean_adjusted = mean_adjusted
        if mean_adjusted > self.threshold:
            logger.critical(
                "open_position_correlation: mean direction-adjusted pairwise "
                "correlation %.3f > %.3f across %s — the open book is one bet",
                mean_adjusted, self.threshold, self.last_directions,
            )
            return True
        logger.debug(
            "open_position_correlation: OK (mean_adjusted=%.3f threshold=%.3f)",
            mean_adjusted, self.threshold,
        )
        return False


class KillSwitchManager:
    def __init__(
        self,
        broker: Any,
        oms: Any,
        config: dict[str, Any] | None = None,
        data_provider: Any | None = None,
        clock: Callable[[], datetime] | None = None,
        trailing_state_path: Path | str | None = _DEFAULT_TRAILING_STATE_PATH,
    ) -> None:
        """``data_provider`` and ``clock`` are optional (CL-ep0c): without
        a provider the open_position_correlation switch simply never
        fires; ``clock`` is injectable for tests. ``trailing_state_path``
        is where the equity trailing stop persists its peak/cooldown
        (None -> in-memory only)."""
        self.broker = broker
        self.oms = oms
        self.config = config or {}
        self._triggered_today: set[str] = set()
        clk = clock or _utc_now
        self.trailing_stop = EquityTrailingStop(
            trailing_stop_pct=float(
                self.config.get("trailing_stop_pct", _DEFAULT_TRAILING_STOP_PCT),
            ),
            cooldown_days=float(
                self.config.get(
                    "trailing_stop_cooldown_days",
                    _DEFAULT_TRAILING_STOP_COOLDOWN_DAYS,
                ),
            ),
            state_path=trailing_state_path,
            clock=clk,
        )
        self.open_position_corr = OpenPositionCorrelationEvaluator(
            broker=broker,
            data_provider=data_provider,
            threshold=float(
                self.config.get(
                    "open_position_corr_threshold",
                    _DEFAULT_OPEN_POSITION_CORR_THRESHOLD,
                ),
            ),
            lookback_days=int(
                self.config.get(
                    "open_position_corr_lookback_days",
                    _DEFAULT_OPEN_POSITION_CORR_LOOKBACK_DAYS,
                ),
            ),
            clock=clk,
        )
        self.switches = self._build_switches()

    def _equity_from(self, ctx: dict[str, Any]) -> float | None:
        """Equity for the trailing stop: context first, broker fallback."""
        equity = ctx.get("equity")
        if equity is not None:
            return float(equity)
        get_account = getattr(self.broker, "get_account", None)
        if callable(get_account):
            try:
                return float(get_account().equity)
            except Exception:
                logger.exception("equity_trailing_stop: broker equity fetch failed")
        return None

    def _build_switches(self) -> list[KillSwitch]:
        return [
            # -3% daily loss: at this threshold, the strategy is likely in a regime
            # where its assumptions are broken (not just noise). Halt new trades to
            # preserve capital until human review. (Arch §5.5)
            KillSwitch(name="daily_loss_limit",
                       condition=lambda ctx: ctx.get("daily_pnl_pct", 0) < -0.03,
                       action="halt_new"),
            # -20% drawdown: beyond this level, recovery requires 25%+ to get back
            # to even, which may exceed strategy Sharpe over relevant horizon.
            # Flatten all open positions. (Arch §5.5)
            KillSwitch(name="drawdown_limit",
                       condition=lambda ctx: ctx.get("portfolio_dd", 0) < -0.20,
                       action="flatten_all"),
            # VIX >35 AND +50% intraday: historically signals panic selling or
            # extreme risk-off (March 2020, August 2024). VIX above 35 is top
            # 5% historically; +50% intraday is ~3σ event. Reduce 50%. (Arch §5.5)
            KillSwitch(name="vix_spike",
                       condition=lambda ctx: ctx.get("vix_level", 0) > 35 and ctx.get("vix_change_1d", 0) > 0.5,
                       action="reduce_50pct"),
            # CVIX Z-score >3: extreme FX vol relative to its own distribution.
            # 3σ event — about 0.3% probability. Halt new trades. (Arch §5.5)
            KillSwitch(name="fx_vol_spike",
                       condition=lambda ctx: ctx.get("cvix_zscore", 0) > 3.0,
                       action="halt_new"),
            # Position reconciliation mismatch: internal state disagrees with broker.
            # Trading with stale state risks duplicates or missed exits. Halt.
            KillSwitch(name="reconciliation_failure",
                       condition=lambda ctx: ctx.get("position_mismatch", False),
                       action="halt_new"),
            # 600s (10 minutes) stale prices: beyond this, the engine is blind to
            # market reality. Broker connection likely lost. Halt new trades.
            # (Arch §5.5)
            KillSwitch(name="stale_prices",
                       condition=lambda ctx: ctx.get("max_price_age_sec", 0) > 600,
                       action="halt_new"),
            # CL-7j9 portfolio-level switches.
            #
            # Correlation regime "crisis" comes from the correlation monitor —
            # signals that historically uncorrelated strategies are now moving
            # together (e.g. a crowded macro factor blew up). Halve gross
            # exposure rather than halting outright; partial reduction lets us
            # ride out short crises without forced liquidation slippage.
            KillSwitch(name="portfolio_correlation_crisis",
                       condition=lambda ctx: ctx.get("correlation_regime", "") == "crisis",
                       action="reduce_50pct"),
            # 0.9 pairwise corr across strategies means we effectively own one
            # bet, not a portfolio. Reduce 50% — same logic as the regime
            # version but driven directly off the live correlation matrix.
            KillSwitch(name="strategy_correlation_spike",
                       condition=lambda ctx: ctx.get("max_pair_corr", 0) > 0.9,
                       action="reduce_50pct"),
            # Single strategy at -25% DD: the portfolio still trades, but this
            # particular strategy goes to halt. Action argument carries the
            # offending strategy_id for the OMS to halt selectively.
            KillSwitch(name="single_strategy_drawdown",
                       condition=lambda ctx: any(
                           dd <= -0.25
                           for dd in (ctx.get("strategy_drawdowns") or {}).values()
                       ),
                       action="halt_strategy"),
            # CL-ep0c equity-curve trailing stop. Peak equity persists in
            # data/equity_trailing_stop_state.json; a breach halts new
            # trades for trailing_stop_cooldown_days ACROSS restarts.
            # Fires idempotently every day of the cooldown (the manager's
            # daily dedup rate-limits the action, the evaluator's own
            # persisted state carries the cooldown).
            KillSwitch(name="equity_trailing_stop",
                       condition=lambda ctx: self.trailing_stop.evaluate(
                           self._equity_from(ctx),
                       ),
                       action="halt_new"),
            # CL-ep0c open-position correlation. Self-contained (broker
            # positions + DataProvider closes) — unlike the two context-fed
            # correlation switches above, this one does not depend on any
            # caller wiring correlation numbers into the context.
            KillSwitch(name="open_position_correlation",
                       condition=lambda ctx: self.open_position_corr.evaluate(),
                       action="reduce_50pct"),
        ]

    def check(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        triggered = []
        log_keys = (
            "daily_pnl_pct", "portfolio_dd", "vix_level",
            "correlation_regime", "max_pair_corr", "strategy_drawdowns",
            "equity",  # CL-ep0c
        )
        for sw in self.switches:
            if not sw.armed or sw.name in self._triggered_today:
                continue
            try:
                if sw.condition(context):
                    log_ctx = {k: v for k, v in context.items() if k in log_keys}
                    logger.critical(
                        "KILL SWITCH: %s triggered — action=%s context=%s",
                        sw.name, sw.action, log_ctx,
                    )
                    self._execute_action(sw.action, context)
                    self._triggered_today.add(sw.name)
                    kill_switch_triggered.labels(
                        switch_name=sw.name, action=sw.action,
                    ).inc()
                    triggered.append({
                        "switch": sw.name, "action": sw.action, "context": context,
                    })
                else:
                    logger.debug("Kill switch %s: OK (value=%s)", sw.name,
                                  {k: v for k, v in context.items() if k in log_keys})
            except Exception:
                logger.exception("Kill switch %s check failed", sw.name)
        return triggered

    def _execute_action(self, action: str, context: dict[str, Any]) -> None:
        if action == "halt_new":
            self.oms.halt_new_trades()
        elif action == "halt_strategy":
            # CL-7j9: surgical halt of just the offending strategies. The OMS
            # is expected to expose halt_strategy(sid) — falls back to a
            # logged warning when the broker integration isn't there yet.
            offending = [
                sid for sid, dd in (context.get("strategy_drawdowns") or {}).items()
                if dd <= -0.25
            ]
            for sid in offending:
                halt_fn = getattr(self.oms, "halt_strategy", None)
                if callable(halt_fn):
                    halt_fn(sid)
                else:
                    logger.warning(
                        "halt_strategy not implemented on OMS — would halt %s",
                        sid,
                    )
        elif action == "reduce_50pct":
            # CL-ep0c minimal-safe implementation: a true 50% gross
            # reduction needs broker order integration, and placing
            # orders from the risk layer is deliberately out of scope —
            # a bugged auto-reducer is its own risk event. Until then:
            # stop NEW risk and page the operator with the evidence.
            # TODO(CL-ep0c follow-up): broker-integrated reduction —
            # snapshot positions, submit idempotent partial closes via
            # the OMS, verify fills. Do NOT place orders here.
            self.oms.halt_new_trades()
            logger.critical(
                "reduce_50pct requested — broker-integrated reduction not "
                "implemented; halted new trades instead. OPERATOR ACTION "
                "REQUIRED: manually reduce gross exposure ~50%%. "
                "Open-position correlation evidence: directions=%s "
                "mean_adjusted=%s matrix=\n%s",
                self.open_position_corr.last_directions,
                self.open_position_corr.last_mean_adjusted,
                self.open_position_corr.last_matrix,
            )
        elif action == "flatten_all":
            logger.warning("Action '%s' requires broker integration — stub", action)

    def reset_daily(self) -> None:
        self._triggered_today.clear()
