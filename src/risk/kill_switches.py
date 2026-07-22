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

CL-i4tx de-facades the rest of the subsystem (code review 2026-07-21
§2.2/§6.1.3):

* Context-fed switches now receive REAL inputs each health tick from
  ``src.risk.risk_context.RiskContextBuilder`` (daily PnL, portfolio
  drawdown, VIX/CVIX, price-stream age, position mismatch).
* ``daily_loss_limit`` / ``drawdown_limit`` thresholds come from the
  active risk profile's kill_switches block — no more hardcoded
  -0.03/-0.20 lambdas that ignored ``CURLIT_RISK_PROFILE``.
* ``flatten_all`` really flattens: a target-0 OrderIntent per open
  broker position through the OMS (canonical-symbol netted), then new
  trades halt. ``reduce_50pct`` really halves each position the same
  way, then halts new trades and pages the operator.
* Three switches whose inputs genuinely do not exist in the engine were
  DELETED rather than left fake: ``portfolio_correlation_crisis`` and
  ``strategy_correlation_spike`` (per-strategy returns history is never
  populated in production — the coordinator's corr-regime path always
  reports "unknown"), and ``single_strategy_drawdown`` (no per-strategy
  equity series exists anywhere in the live engine).
* Evaluation errors fail CLOSED: each failure is logged with traceback;
  three consecutive failures of the SAME switch halt new trades — a
  broken safety net must not quietly keep trading.
* ``log_arming(provided_keys)`` prints one ARMED/UNARMED line per
  switch at engine boot so operators see the truth, not the docs.
* ``reset_daily()`` is now actually invoked (health tick, on UTC-day
  rollover detected by the context builder) so a fired switch can
  re-arm the next day after ``/api/system/resume``.
"""

import json
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any

from src.execution.broker import canonical_symbol
from src.execution.oms import OrderIntent
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

# CL-i4tx profile-driven thresholds — defaults mirror the conservative
# profile (configs/risk_profile.yaml kill_switches block); the live
# aggressive profile overrides them via the config dict at construction.
_DEFAULT_DAILY_LOSS_LIMIT_PCT: float = -0.03
_DEFAULT_DRAWDOWN_LIMIT_PCT: float = -0.20

# CL-i4tx fixed thresholds (documented magic numbers, Arch §5.5 — not
# profile knobs because their calibration is historical, not appetite):
# VIX >35 is top-5% historically; +50% over the prior close is a ~3σ
# event (March 2020, August 2024). CVIX z>3 is ~0.3% probability.
_VIX_SPIKE_LEVEL: float = 35.0
_VIX_SPIKE_CHANGE_1D: float = 0.5
_CVIX_ZSCORE_LIMIT: float = 3.0
# 600s (10 min) without ANY tick inside the trading window: the engine
# is blind to market reality; the broker stream is likely dead.
_PRICE_STREAM_STALE_SEC: float = 600.0

# Below this net quantity a broker position is dust — flatten/reduce
# intents are not worth a market order.
_MIN_ACTIONABLE_QTY: float = 1e-6

# CL-i4tx fail-closed policy: after this many CONSECUTIVE evaluation
# failures of the same switch, halt new trades. A brake that cannot be
# evaluated must not be treated as a brake that is fine.
_EVAL_FAILURES_BEFORE_HALT: int = 3


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class KillSwitch:
    name: str
    condition: Callable[[dict[str, Any]], bool]
    action: str
    armed: bool = True
    #: Context keys this switch needs to ever fire (CL-i4tx). Empty means
    #: self-feeding (pulls its own inputs). Consumed by ``log_arming``.
    required_keys: tuple[str, ...] = ()


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

    This evaluator pulls its own inputs: ``broker.get_positions()``
    for the held symbols and ``DataProvider.get_aligned_series`` for
    ``lookback_days`` of daily closes. (The context-fed strategy-level
    correlation switches it once sat beside were deleted in CL-i4tx —
    their per-strategy returns feed never existed in production.)

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
        # Exposed for the live engine's RiskContextBuilder wiring (CL-i4tx).
        self.data_provider = data_provider
        self._triggered_today: set[str] = set()
        # CL-i4tx fail-closed accounting: consecutive evaluation failures
        # per switch. Reset to 0 on any successful evaluation.
        self._consecutive_failures: dict[str, int] = {}
        # CL-i4tx profile-driven thresholds (no hardcoded lambdas).
        self.daily_loss_limit_pct = float(
            self.config.get("daily_loss_limit_pct", _DEFAULT_DAILY_LOSS_LIMIT_PCT),
        )
        self.drawdown_limit_pct = float(
            self.config.get("drawdown_limit_pct", _DEFAULT_DRAWDOWN_LIMIT_PCT),
        )
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
            # Path|str|None accepted at this boundary; EquityTrailingStop
            # stores Path|None (pre-existing mypy error fixed in CL-i4tx).
            state_path=(
                Path(trailing_state_path)
                if trailing_state_path is not None else None
            ),
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
        # CL-i4tx: three switches were DELETED here rather than left fake —
        # portfolio_correlation_crisis / strategy_correlation_spike (their
        # feed, per-strategy returns history, is never populated in
        # production, so the coordinator's corr regime is always "unknown")
        # and single_strategy_drawdown (no per-strategy equity series exists
        # in the live engine). open_position_correlation covers the crowded-
        # book risk with inputs it pulls itself.
        return [
            # Daily loss beyond the profile limit: the strategy is likely in
            # a regime where its assumptions are broken (not just noise).
            # Halt new trades to preserve capital until human review.
            # (Arch §5.5; threshold from risk_profile.yaml — CL-i4tx)
            KillSwitch(name="daily_loss_limit",
                       condition=lambda ctx: (
                           ctx.get("daily_pnl_pct", 0.0)
                           < self.daily_loss_limit_pct
                       ),
                       action="halt_new",
                       required_keys=("daily_pnl_pct",)),
            # Drawdown beyond the profile limit: recovery from -20% needs
            # +25% just to get back to even, which may exceed strategy Sharpe
            # over the relevant horizon. Flatten all open positions.
            # (Arch §5.5; threshold from risk_profile.yaml — CL-i4tx)
            KillSwitch(name="drawdown_limit",
                       condition=lambda ctx: (
                           ctx.get("portfolio_dd", 0.0) < self.drawdown_limit_pct
                       ),
                       action="flatten_all",
                       required_keys=("portfolio_dd",)),
            # VIX >35 AND +50% vs prior close: historically signals panic
            # selling or extreme risk-off (March 2020, August 2024). Daily
            # closes via the context builder — fires up to a day late, which
            # is honest about the data we actually have. Reduce 50%.
            # (Arch §5.5)
            KillSwitch(name="vix_spike",
                       condition=lambda ctx: (
                           ctx.get("vix_level", 0.0) > _VIX_SPIKE_LEVEL
                           and ctx.get("vix_change_1d", 0.0) > _VIX_SPIKE_CHANGE_1D
                       ),
                       action="reduce_50pct",
                       required_keys=("vix_level", "vix_change_1d")),
            # CVIX Z-score >3: extreme FX vol relative to its own
            # distribution. 3σ event — about 0.3% probability. Halt new
            # trades. (Arch §5.5)
            KillSwitch(name="fx_vol_spike",
                       condition=lambda ctx: (
                           ctx.get("cvix_zscore", 0.0) > _CVIX_ZSCORE_LIMIT
                       ),
                       action="halt_new",
                       required_keys=("cvix_zscore",)),
            # Position reconciliation mismatch: internal state disagrees with
            # broker (fed by the engine's periodic PositionReconciler
            # alignment check — CL-i4tx). Trading with stale state risks
            # duplicates or missed exits. Halt.
            KillSwitch(name="reconciliation_failure",
                       condition=lambda ctx: bool(
                           ctx.get("position_mismatch", False),
                       ),
                       action="halt_new",
                       required_keys=("position_mismatch",)),
            # No tick from the price stream for 10 minutes inside the
            # trading window: the engine is blind to market reality; the
            # broker connection is likely lost. Halt new trades. (Arch §5.5)
            KillSwitch(name="stale_prices",
                       condition=lambda ctx: (
                           ctx.get("price_stream_age_sec", 0.0)
                           > _PRICE_STREAM_STALE_SEC
                       ),
                       action="halt_new",
                       required_keys=("price_stream_age_sec",)),
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
            "equity", "daily_pnl_pct", "portfolio_dd", "vix_level",
            "vix_change_1d", "cvix_zscore", "position_mismatch",
            "price_stream_age_sec",
        )
        for sw in self.switches:
            if not sw.armed or sw.name in self._triggered_today:
                continue
            try:
                fired = sw.condition(context)
            except Exception:
                # CL-i4tx fail-closed policy: an unevaluable brake is not a
                # brake. Log every failure; after
                # _EVAL_FAILURES_BEFORE_HALT consecutive failures of the
                # SAME switch, halt new trades and page the operator.
                failures = self._consecutive_failures.get(sw.name, 0) + 1
                self._consecutive_failures[sw.name] = failures
                logger.exception(
                    "Kill switch %s evaluation failed (consecutive=%d)",
                    sw.name, failures,
                )
                if failures == _EVAL_FAILURES_BEFORE_HALT:
                    logger.critical(
                        "Kill switch %s failed %d consecutive evaluations — "
                        "failing CLOSED: halting new trades. OPERATOR ACTION "
                        "REQUIRED: fix the data path, then resume via "
                        "/api/system/resume.",
                        sw.name, failures,
                    )
                    self.oms.halt_new_trades()
                continue
            self._consecutive_failures[sw.name] = 0
            if fired:
                log_ctx = {k: v for k, v in context.items() if k in log_keys}
                logger.critical(
                    "KILL SWITCH: %s triggered — action=%s context=%s",
                    sw.name, sw.action, log_ctx,
                )
                effective = True
                try:
                    effective = self._execute_action(sw.action)
                except Exception:
                    # A PARTIALLY applied action still spends the trigger —
                    # endless re-fires of a half-working action are worse
                    # than one audit-trailed attempt. Only a could-not-even-
                    # attempt (position fetch failed → effective=False)
                    # leaves the trigger unspent so it re-fires next tick
                    # while open risk persists (CL-8cw1).
                    logger.exception(
                        "Kill switch %s action %s failed", sw.name, sw.action,
                    )
                if effective:
                    self._triggered_today.add(sw.name)
                else:
                    logger.warning(
                        "Kill switch %s: action ineffective — trigger NOT "
                        "spent, will re-fire next evaluation", sw.name,
                    )
                kill_switch_triggered.labels(
                    switch_name=sw.name, action=sw.action,
                ).inc()
                triggered.append({
                    "switch": sw.name, "action": sw.action, "context": context,
                    "effective": effective,
                })
            else:
                logger.debug(
                    "Kill switch %s: OK (value=%s)", sw.name,
                    {k: v for k, v in context.items() if k in log_keys},
                )
        return triggered

    def _execute_action(self, action: str) -> bool:
        """Execute a fired switch's action. Returns EFFECTIVE: False means
        the de-risk could not even be attempted (position fetch failed) —
        the caller must NOT spend the once-per-day trigger, so the switch
        re-fires next tick and keeps trying while the condition holds
        (CL-8cw1). Halting always succeeds and always accompanies the
        attempt, so open risk can never grow while we retry."""
        if action == "halt_new":
            self.oms.halt_new_trades()
            return True
        elif action == "reduce_50pct":
            # CL-i4tx: real broker-integrated reduction — one halved-target
            # intent per net open position through the OMS, then halt new
            # trades and page the operator with the evidence.
            submitted = self._submit_position_intents(
                target_fraction=0.5, strategy_id="kill_switch_reduce",
            )
            self.oms.halt_new_trades()
            if submitted is None:
                logger.critical(
                    "reduce_50pct could NOT enumerate positions — halted "
                    "only; trigger NOT spent, switch will re-fire next tick",
                )
                return False
            logger.critical(
                "reduce_50pct executed: %d halving intents submitted via OMS; "
                "new trades halted. OPERATOR ACTION REQUIRED: verify fills "
                "and residual exposure. Open-position correlation evidence "
                "(if corr-triggered): directions=%s mean_adjusted=%s "
                "matrix=\n%s",
                submitted,
                self.open_position_corr.last_directions,
                self.open_position_corr.last_mean_adjusted,
                self.open_position_corr.last_matrix,
            )
            return True
        elif action == "flatten_all":
            # CL-i4tx: real flatten — one target-0 intent per net open
            # broker position through the OMS, then halt new trades. Was a
            # log-stub before ("requires broker integration").
            submitted = self._submit_position_intents(
                target_fraction=0.0, strategy_id="kill_switch_flatten",
            )
            self.oms.halt_new_trades()
            if submitted is None:
                logger.critical(
                    "flatten_all could NOT enumerate positions — halted "
                    "only; trigger NOT spent, switch will re-fire next tick",
                )
                return False
            logger.critical(
                "flatten_all executed: %d closing intents submitted via OMS; "
                "new trades halted. OPERATOR ACTION REQUIRED: verify all "
                "positions closed at the broker.",
                submitted,
            )
            return True
        else:
            # Unknown action string is a wiring bug — refuse silently doing
            # nothing about a fired kill switch.
            self.oms.halt_new_trades()
            logger.critical(
                "Kill switch action %r is not implemented — halted new "
                "trades as the fail-closed fallback", action,
            )
        return True

    def _submit_position_intents(
        self, target_fraction: float, strategy_id: str,
    ) -> int | None:
        """Submit ``target = net_qty * target_fraction`` intents for every
        net open broker position. Returns the number of intents submitted;
        ``None`` when the position fetch failed (nothing was even attempted
        — distinct from 0, which means genuinely flat).

        Positions are netted by :func:`canonical_symbol` (CL-qqra) so the
        broker's compact form and OANDA-underscore event legs cannot
        produce duplicate orders for the same instrument; routing keeps
        the broker's own symbol string. Intents bypass the OMS halt gate
        (``bypass_halt=True``) because a prior halt_new (e.g. the trailing
        stop) must never block emergency de-risking — these intents only
        ever REDUCE exposure.
        """
        try:
            positions = self.broker.get_positions() or []
        except Exception:
            logger.exception(
                "%s: broker.get_positions() failed — no intents submitted; "
                "halting only", strategy_id,
            )
            # None (not 0) so the caller can distinguish "couldn't
            # enumerate positions" from "genuinely flat" — an ineffective
            # flatten must NOT spend the once-per-day trigger (CL-8cw1).
            return None
        net_qty: dict[str, float] = {}
        route_symbol: dict[str, str] = {}
        for pos in positions:
            symbol = getattr(pos, "symbol", None)
            qty = getattr(pos, "quantity", None)
            if not symbol or qty is None:
                continue
            key = canonical_symbol(symbol)
            net_qty[key] = net_qty.get(key, 0.0) + float(qty)
            route_symbol.setdefault(key, str(symbol))
        submitted = 0
        for key in sorted(net_qty):
            qty = net_qty[key]
            if abs(qty) < _MIN_ACTIONABLE_QTY:
                continue
            intent = OrderIntent(
                strategy_id=strategy_id,
                symbol=route_symbol[key],
                target_position=qty * target_fraction,
                urgency="urgent",
            )
            try:
                self.oms.submit_intent(intent, bypass_halt=True)
                submitted += 1
                logger.warning(
                    "%s: intent %s target %.4f (was %.4f)",
                    strategy_id, route_symbol[key], qty * target_fraction, qty,
                )
            except Exception:
                logger.exception(
                    "%s: submit_intent failed for %s", strategy_id,
                    route_symbol[key],
                )
        return submitted

    def log_arming(self, provided_keys: Iterable[str]) -> None:
        """Boot-time honesty (CL-i4tx): one line per switch, ARMED vs UNARMED.

        ``provided_keys`` is what the caller's context builder can actually
        supply (``RiskContextBuilder.provided_keys()``). A switch whose
        required keys are not all provided WILL NEVER FIRE — that is logged
        CRITICAL so nobody trusts a brake that is not connected.
        """
        provided = set(provided_keys)
        for sw in self.switches:
            missing = [k for k in sw.required_keys if k not in provided]
            if sw.name == "open_position_correlation" and self.data_provider is None:
                missing.append("data_provider")
            if missing:
                logger.critical(
                    "kill switch %s: UNARMED — missing inputs %s; this "
                    "switch will NEVER fire", sw.name, missing,
                )
            else:
                logger.info(
                    "kill switch %s: ARMED (action=%s, inputs=%s)",
                    sw.name, sw.action,
                    ", ".join(sw.required_keys) or "self-feeding",
                )

    def reset_daily(self) -> None:
        """Re-arm the once-per-day trigger dedup. Called by the live engine
        health tick at UTC-day rollover (CL-i4tx) — before that, nothing
        called this and a fired switch stayed deduped until restart."""
        if self._triggered_today:
            logger.info(
                "Kill switches re-armed for the new UTC day (had fired: %s)",
                sorted(self._triggered_today),
            )
        self._triggered_today.clear()
