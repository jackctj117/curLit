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

CL-d7ex makes auto-resume lift ONLY halts it owns: operator / startup /
cold-start-reconciliation halts are recorded as sticky ``external:*``
causes via ``record_external_halt``, and a halt already in force with no
recorded cause is adopted as ``external:unattributed_prior_halt`` before a
data gate piles on, so a cleared ``stale_prices`` can never silently lift
an entry-paused engine. Only ``/api/system/resume`` clears them.

CL-o9sq/CL-pksi make emergency closes evidence-driven and restart-safe:
every flatten/reduce order is persisted as an :class:`EmergencyAttempt`
(migration 027) BEFORE it is sent; a WORKING/UNKNOWN/partial outcome fences
the symbol (here AND in the OMS, so no other writer can size off a book that
may not include it) and records the sticky external halt cause
``unresolved_emergency_orders``. Only verified broker evidence resolves a
fence (``resolve_derisk``): a streamed fill for the attempt's client id
(``OrderManager.on_fill`` listener) or a broker order lookup
(``refresh_unresolved_derisk``, each health tick and at startup). A position
snapshot never does. A verified zero-fill rejection/cancel permits one retry
per tick against the ORIGINAL fixed target. Startup recovery
(``recover_emergency_attempts``) re-fences every unresolved attempt from the
durable store. A partial fill that went terminal stays fenced until an
operator reconciles it (``release_derisk_fence``).
"""

import json
import logging
import math
import os
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from itertools import combinations
from pathlib import Path
from typing import Any

from src.execution.broker import BrokerOrderNotFoundError, Order, canonical_symbol
from src.execution.oms import OrderIntent, SubmissionResult, SubmissionStatus
from src.monitoring.metrics import kill_switch_triggered
from src.portfolio.reconciler import SnapshotUnavailableError, validate_broker_snapshot
from src.risk.emergency_attempts import (
    EVIDENCE_FILL_EVENT,
    EVIDENCE_OPERATOR,
    EVIDENCE_ORDER_LOOKUP,
    EVIDENCE_SUBMISSION,
    RETRYABLE_STATUSES,
    AttemptStatus,
    DeriskEvidence,
    EmergencyAttempt,
    EmergencyAttemptStore,
    InMemoryEmergencyAttemptStore,
    apply_evidence,
)

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

# CL-nxjx — DATA-AVAILABILITY gates: switches that halt because the engine
# temporarily can't SEE the market, not because a RISK event occurred. When
# their condition clears (data returns) they AUTO-LIFT the halt, unlike risk
# switches (drawdown/VIX/reconciliation) which stay sticky for human review.
# stale_prices is the canonical case: a laptop wake / network blip halts on
# stale prices; the stream reconnects; trading should resume on its own
# instead of needing a manual /api/system/resume.
_DATA_GATE_SWITCHES: frozenset[str] = frozenset({"stale_prices"})

# CL-d7ex — halt causes that did NOT come from a kill switch (operator
# entry-paused startup, cold-start reconciliation mismatch/failure,
# /api/system/halt). They are recorded in ``_active_halt_causes`` under this
# prefix so the cause set is never empty while such a halt is in force; they
# are never data gates, so ``attempt_auto_resume`` can never drain them and
# only a manual /api/system/resume (``reset_daily()``) clears them. The
# prefix keeps them disjoint from switch names.
EXTERNAL_HALT_PREFIX: str = "external:"

# CL-d7ex — recorded when a switch fires while the OMS is ALREADY halted with
# no recorded cause (a halt from a path that did not register itself). That
# halt was not ours, so it must outlive the data gate that piled on top.
_UNATTRIBUTED_PRIOR_HALT: str = EXTERNAL_HALT_PREFIX + "unattributed_prior_halt"

# CL-pksi — sticky external halt causes for emergency-order ambiguity. Held
# while any emergency attempt is unresolved (or its store is unreadable);
# never a data gate, so auto-resume cannot lift them, and reset_daily keeps
# them while an unresolved attempt remains even on a manual resume.
UNRESOLVED_EMERGENCY_CAUSE: str = "unresolved_emergency_orders"
EMERGENCY_STORE_UNAVAILABLE_CAUSE: str = "emergency_attempts_unavailable"

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
                    self.cooldown_until.isoformat(),
                    self.peak_equity,
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
                self.cooldown_until.isoformat(),
                self.peak_equity,
                float(equity),
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
                eq,
                drawdown * 100,
                self.peak_equity,
                -self.trailing_stop_pct * 100,
                self.cooldown_until.isoformat(),
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
                self.cooldown_until.isoformat() if self.cooldown_until is not None else None
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
            self.cooldown_until = datetime.fromisoformat(until) if until is not None else None
        except (ValueError, KeyError, TypeError) as exc:
            msg = (
                f"equity trailing-stop state at {path} is corrupt or "
                f"unreadable ({exc}) — refusing to start with a reset peak; "
                f"repair or remove the file"
            )
            raise ValueError(msg) from exc
        logger.info(
            "equity_trailing_stop: loaded state from %s (peak=%s cooldown_until=%s)",
            path,
            self.peak_equity,
            self.cooldown_until,
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
                "(< %d required) — skipping",
                len(returns),
                self.min_observations,
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
                mean_adjusted,
                self.threshold,
                self.last_directions,
            )
            return True
        logger.debug(
            "open_position_correlation: OK (mean_adjusted=%.3f threshold=%.3f)",
            mean_adjusted,
            self.threshold,
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
        attempt_store: EmergencyAttemptStore | None = None,
    ) -> None:
        """``data_provider`` and ``clock`` are optional (CL-ep0c): without
        a provider the open_position_correlation switch simply never
        fires; ``clock`` is injectable for tests. ``trailing_state_path``
        is where the equity trailing stop persists its peak/cooldown
        (None -> in-memory only). ``attempt_store`` persists emergency order
        attempts (CL-pksi); None -> process-local only (tests, legacy)."""
        self.broker = broker
        self.oms = oms
        self.config = config or {}
        # Exposed for the live engine's RiskContextBuilder wiring (CL-i4tx).
        self.data_provider = data_provider
        self._triggered_today: set[str] = set()
        # CL-o9sq: fixed per-action targets avoid repeated halving when only
        # some legs complete. Uncertain/working submissions fence the symbol
        # across ALL de-risk actions until explicitly reconciled.
        self._derisk_targets: dict[str, dict[str, tuple[str, float]]] = {}
        self._derisk_results: dict[tuple[str, str], SubmissionResult] = {}
        # CL-pksi: every emergency attempt this process knows (made here, or
        # recovered unresolved from the store at startup), by intent id. The
        # unresolved ones are the fences. Guarded by _attempt_lock; lock
        # order is _attempt_lock -> _cause_lock -> OMS lock, and no broker
        # call ever runs while it is held.
        self.attempt_store: EmergencyAttemptStore = (
            attempt_store if attempt_store is not None else InMemoryEmergencyAttemptStore()
        )
        self._attempts: dict[str, EmergencyAttempt] = {}
        self._attempt_lock = threading.RLock()
        # action -> durable episode id holding its FIXED targets (CL-pksi).
        self._episodes: dict[str, str] = {}
        # Legs whose emergency close is VERIFIED filled (by any evidence path,
        # in-process or at startup) but the position feed has not yet shown
        # it. Fenced for EVERY writer — the cold-start reconciler and other
        # kill-switch actions included — until a snapshot agrees with the
        # target, so a lagging book can never trigger a second close. Its
        # episode stays OPEN (durable) until then, so a restart restores it.
        # canonical symbol -> (route symbol, target, action).
        self._confirm_pending: dict[str, tuple[str, float, str]] = {}
        # True while startup recovery could not read the attempt store: every
        # OMS submission stays blocked until a retry succeeds.
        self._recovery_failed = False
        # Canonical symbols already sent an emergency order during the current
        # check() — the bound on retries: at most ONE per symbol per tick.
        self._tick_submitted: set[str] = set()
        # CL-nxjx: switches currently responsible for a halt. A DATA-gate
        # cause is removed (and re-armed) by attempt_auto_resume when its
        # condition clears; a RISK cause stays until a manual resume. When
        # this becomes empty via data-gate clearing, the OMS auto-resumes.
        # Empty while the OMS is halted ⇒ a NON-switch halt (manual
        # /api/system/halt) — never auto-resumed.
        self._active_halt_causes: set[str] = set()
        # CL-d7ex: serializes the auto-resume drain-and-resume decision
        # against record_external_halt (called from the API thread) so an
        # operator halt landing mid-drain is never resumed away. RLock so a
        # nested call from the same thread cannot self-deadlock.
        self._cause_lock = threading.RLock()
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
        self._clock = clk
        # CL-pksi: streamed fills resolve emergency fences. Registered on the
        # real OMS only (feature-detected) — the listener runs after the OMS's
        # own transaction-id dedup, so a redelivered fill never re-fires.
        register = getattr(oms, "add_fill_listener", None)
        if callable(register):
            register(self.on_broker_fill)
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
            state_path=(Path(trailing_state_path) if trailing_state_path is not None else None),
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
            KillSwitch(
                name="daily_loss_limit",
                condition=lambda ctx: ctx.get("daily_pnl_pct", 0.0) < self.daily_loss_limit_pct,
                action="halt_new",
                required_keys=("daily_pnl_pct",),
            ),
            # Drawdown beyond the profile limit: recovery from -20% needs
            # +25% just to get back to even, which may exceed strategy Sharpe
            # over the relevant horizon. Flatten all open positions.
            # (Arch §5.5; threshold from risk_profile.yaml — CL-i4tx)
            KillSwitch(
                name="drawdown_limit",
                condition=lambda ctx: ctx.get("portfolio_dd", 0.0) < self.drawdown_limit_pct,
                action="flatten_all",
                required_keys=("portfolio_dd",),
            ),
            # VIX >35 AND +50% vs prior close: historically signals panic
            # selling or extreme risk-off (March 2020, August 2024). Daily
            # closes via the context builder — fires up to a day late, which
            # is honest about the data we actually have. Reduce 50%.
            # (Arch §5.5)
            KillSwitch(
                name="vix_spike",
                condition=lambda ctx: (
                    ctx.get("vix_level", 0.0) > _VIX_SPIKE_LEVEL
                    and ctx.get("vix_change_1d", 0.0) > _VIX_SPIKE_CHANGE_1D
                ),
                action="reduce_50pct",
                required_keys=("vix_level", "vix_change_1d"),
            ),
            # CVIX Z-score >3: extreme FX vol relative to its own
            # distribution. 3σ event — about 0.3% probability. Halt new
            # trades. (Arch §5.5)
            KillSwitch(
                name="fx_vol_spike",
                condition=lambda ctx: ctx.get("cvix_zscore", 0.0) > _CVIX_ZSCORE_LIMIT,
                action="halt_new",
                required_keys=("cvix_zscore",),
            ),
            # Position reconciliation mismatch: internal state disagrees with
            # broker (fed by the engine's periodic PositionReconciler
            # alignment check — CL-i4tx). Trading with stale state risks
            # duplicates or missed exits. Halt.
            KillSwitch(
                name="reconciliation_failure",
                condition=lambda ctx: bool(
                    ctx.get("position_mismatch", False),
                ),
                action="halt_new",
                required_keys=("position_mismatch",),
            ),
            # No tick from the price stream for 10 minutes inside the
            # trading window: the engine is blind to market reality; the
            # broker connection is likely lost. Halt new trades. (Arch §5.5)
            KillSwitch(
                name="stale_prices",
                condition=lambda ctx: (
                    ctx.get("price_stream_age_sec", 0.0) > _PRICE_STREAM_STALE_SEC
                ),
                action="halt_new",
                required_keys=("price_stream_age_sec",),
            ),
            # CL-ep0c equity-curve trailing stop. Peak equity persists in
            # data/equity_trailing_stop_state.json; a breach halts new
            # trades for trailing_stop_cooldown_days ACROSS restarts.
            # Fires idempotently every day of the cooldown (the manager's
            # daily dedup rate-limits the action, the evaluator's own
            # persisted state carries the cooldown).
            KillSwitch(
                name="equity_trailing_stop",
                condition=lambda ctx: self.trailing_stop.evaluate(
                    self._equity_from(ctx),
                ),
                action="halt_new",
            ),
            # CL-ep0c open-position correlation. Self-contained (broker
            # positions + DataProvider closes) — unlike the two context-fed
            # correlation switches above, this one does not depend on any
            # caller wiring correlation numbers into the context.
            KillSwitch(
                name="open_position_correlation",
                condition=lambda ctx: self.open_position_corr.evaluate(),
                action="reduce_50pct",
            ),
        ]

    def check(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        triggered = []
        # CL-pksi: a new evaluation tick — each symbol may receive at most one
        # emergency order (first attempt or bounded retry) during it.
        self._tick_submitted.clear()
        log_keys = (
            "equity",
            "daily_pnl_pct",
            "portfolio_dd",
            "vix_level",
            "vix_change_1d",
            "cvix_zscore",
            "position_mismatch",
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
                    sw.name,
                    failures,
                )
                if failures >= _EVAL_FAILURES_BEFORE_HALT:
                    # CL-7zwp (P1): >= not == so a broken evaluator RE-halts
                    # every subsequent tick (halt_new is idempotent) — with
                    # == the halt only ever fired exactly at the 3rd failure
                    # and a later data-gate auto-resume could un-halt it
                    # permanently. Register it as a sticky (non-data-gate)
                    # cause so attempt_auto_resume never lifts a fail-closed
                    # halt: only a manual /api/system/resume clears it.
                    logger.critical(
                        "Kill switch %s failed %d consecutive evaluations — "
                        "failing CLOSED: halting new trades. OPERATOR ACTION "
                        "REQUIRED: fix the data path, then resume via "
                        "/api/system/resume.",
                        sw.name,
                        failures,
                    )
                    self._adopt_unattributed_halt()
                    self.oms.halt_new_trades()
                    self._active_halt_causes.add(sw.name)
                continue
            self._consecutive_failures[sw.name] = 0
            if fired:
                log_ctx = {k: v for k, v in context.items() if k in log_keys}
                logger.critical(
                    "KILL SWITCH: %s triggered — action=%s context=%s",
                    sw.name,
                    sw.action,
                    log_ctx,
                )
                effective = False
                # CL-d7ex: snapshot a pre-existing cause-less halt BEFORE this
                # switch's action halts the OMS and records its own cause.
                self._adopt_unattributed_halt()
                try:
                    effective = self._execute_action(sw.action)
                except Exception:
                    # An exception is never evidence of completed de-risking.
                    self.oms.halt_new_trades()
                    logger.exception(
                        "Kill switch %s action %s failed",
                        sw.name,
                        sw.action,
                    )
                if effective:
                    self._triggered_today.add(sw.name)
                    # Track the halt cause (CL-nxjx) so a data-gate can
                    # later auto-lift while risk causes stay sticky.
                    self._active_halt_causes.add(sw.name)
                else:
                    logger.warning(
                        "Kill switch %s: action ineffective — trigger NOT "
                        "spent, will re-fire next evaluation",
                        sw.name,
                    )
                kill_switch_triggered.labels(
                    switch_name=sw.name,
                    action=sw.action,
                ).inc()
                triggered.append(
                    {
                        "switch": sw.name,
                        "action": sw.action,
                        "context": context,
                        "effective": effective,
                    }
                )
            else:
                logger.debug(
                    "Kill switch %s: OK (value=%s)",
                    sw.name,
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
            result = self._submit_position_intents(
                target_fraction=0.5,
                strategy_id="kill_switch_reduce",
            )
            self.oms.halt_new_trades()
            if not result or result[0] < result[1] or result[0] == 0:
                # CL-xh6g (P1): re-fire (trigger NOT spent) unless EVERY
                # actionable leg was submitted. A partial reduce (some legs
                # rejected) leaves residual open risk, so retry next tick;
                # None=fetch failed, (0,0)=flat/degraded — both re-fire per
                # CL-9dhg ("never spend the brake on nothing").
                sub = 0 if not result else result[0]
                act = 0 if not result else result[1]
                logger.critical(
                    "reduce_50pct INCOMPLETE (%d/%s legs) — halted only; "
                    "trigger NOT spent, switch re-fires next tick to retry",
                    sub,
                    "fetch-failed" if result is None else act,
                )
                return False
            logger.critical(
                "reduce_50pct completed: %d original reduction targets confirmed; "
                "new trades halted. OPERATOR ACTION REQUIRED: verify fills "
                "and residual exposure. Open-position correlation evidence "
                "(if corr-triggered): directions=%s mean_adjusted=%s "
                "matrix=\n%s",
                result[0],
                self.open_position_corr.last_directions,
                self.open_position_corr.last_mean_adjusted,
                self.open_position_corr.last_matrix,
            )
            return True
        elif action == "flatten_all":
            # CL-i4tx: real flatten — one target-0 intent per net open
            # broker position through the OMS, then halt new trades. Was a
            # log-stub before ("requires broker integration").
            result = self._submit_position_intents(
                target_fraction=0.0,
                strategy_id="kill_switch_flatten",
            )
            self.oms.halt_new_trades()
            if not result or result[0] < result[1] or result[0] == 0:
                # CL-xh6g (P1): a PARTIAL flatten (some legs rejected) leaves
                # unstopped risk — do NOT spend the once-per-day trigger, so
                # the switch re-fires next tick and keeps retrying the failed
                # legs (target-0 is idempotent — already-flat legs are skipped
                # as sub-actionable, so it converges). None=fetch failed,
                # (0,0)=flat/degraded also re-fire (CL-9dhg).
                sub = 0 if not result else result[0]
                act = 0 if not result else result[1]
                logger.critical(
                    "flatten_all INCOMPLETE (%d/%s legs) — halted only; "
                    "trigger NOT spent, switch re-fires next tick to retry "
                    "the unclosed legs",
                    sub,
                    "fetch-failed" if result is None else act,
                )
                return False
            logger.critical(
                "flatten_all completed: %d closing targets confirmed; "
                "new trades halted. OPERATOR ACTION REQUIRED: verify all "
                "positions closed at the broker.",
                result[0],
            )
            return True
        else:
            # Unknown action string is a wiring bug — refuse silently doing
            # nothing about a fired kill switch.
            self.oms.halt_new_trades()
            logger.critical(
                "Kill switch action %r is not implemented — halted new "
                "trades as the fail-closed fallback",
                action,
            )
        return True

    def _submit_position_intents(
        self,
        target_fraction: float,
        strategy_id: str,
    ) -> tuple[int, int] | None:
        """Submit ``target = net_qty * target_fraction`` intents for every
        net open broker position. Returns ``(completed, actionable)`` — how
        many targets were confirmed vs how many positions needed one — so
        the caller can distinguish a COMPLETE de-risk from a PARTIAL one
        (some legs rejected, CL-xh6g). ``None`` when the position fetch failed
        (nothing was even attempted — distinct from (0, 0), genuinely flat).

        Positions are netted by :func:`canonical_symbol` (CL-qqra) so the
        broker's compact form and OANDA-underscore event legs cannot
        produce duplicate orders for the same instrument; routing keeps
        the broker's own symbol string. Intents bypass the OMS halt gate
        (``bypass_halt=True``) because a prior halt_new (e.g. the trailing
        stop) must never block emergency de-risking — these intents only
        ever REDUCE exposure.
        """
        try:
            positions = self.broker.get_positions()
            if not isinstance(positions, list):
                raise ValueError("Broker position snapshot is not a list")
            # Validate the WHOLE snapshot before acting on any leg; a partial
            # parse must not make omitted exposure look flat.
            for pos in positions:
                if (
                    not isinstance(pos.symbol, str)
                    or not canonical_symbol(pos.symbol)
                    or isinstance(pos.quantity, bool)
                    or not math.isfinite(float(pos.quantity))
                ):
                    raise ValueError("Broker position snapshot contains an invalid leg")
        except Exception:
            logger.exception(
                "%s: broker position snapshot unavailable/invalid — no orders; halting only",
                strategy_id,
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
        if self._recovery_failed:
            # CL-pksi: outstanding emergency orders from before the restart are
            # unknown; any close could duplicate one. Halted only (re-fires).
            logger.critical(
                "%s: emergency-attempt recovery has not succeeded — no orders; halting only",
                strategy_id,
            )
            return None
        self._confirm_positions(positions)
        current = self._derisk_targets.get(strategy_id, {})
        targets = dict(current)
        for key in sorted(net_qty):
            qty = net_qty[key]
            if abs(qty) < _MIN_ACTIONABLE_QTY:
                continue
            targets.setdefault(key, (route_symbol[key], qty * target_fraction))
        if targets != current or strategy_id not in self._episodes:
            # CL-pksi: the FIXED targets are durable before any order, so a
            # restart never re-derives a completed leg's target from its
            # smaller post-fill position (which would reduce it again). The
            # in-memory targets/episode are committed ONLY after the write
            # succeeds, so a failed write is retried (and blocks) next tick.
            episode_id = self._episodes.get(strategy_id) or str(uuid.uuid4())
            try:
                logger.warning(
                    "%s: persisting episode %s fixed targets %s", strategy_id, episode_id, targets
                )
                self.attempt_store.save_episode(episode_id, strategy_id, targets, self._clock())
            except Exception:
                logger.critical(
                    "%s: fixed targets could NOT be persisted — no orders (fail closed)",
                    strategy_id,
                    exc_info=True,
                )
                self.record_external_halt(EMERGENCY_STORE_UNAVAILABLE_CAUSE)
                return None
            self._episodes[strategy_id] = episode_id
        self._derisk_targets[strategy_id] = targets
        submitted = 0
        actionable = len(targets)
        for key, (symbol, target) in targets.items():
            previous = self._derisk_results.get((strategy_id, key))
            pending = self._unresolved_for(key)
            if pending:
                # CL-pksi: only broker evidence (resolve_derisk) or an operator
                # release lifts this — never this snapshot, which cannot tell a
                # filled order from a working one or a stale book.
                logger.critical(
                    "%s: %s has %d unresolved emergency order(s) %s; no duplicate close; "
                    "awaiting broker evidence or operator reconciliation",
                    strategy_id,
                    symbol,
                    len(pending),
                    [f"{a.client_order_id}:{a.status}" for a in pending],
                )
                continue
            awaiting = self._confirm_pending.get(key)
            if awaiting is not None:
                # A verified emergency close is not yet visible in this
                # snapshot: sizing from it would close (or reverse) twice.
                logger.critical(
                    "%s: %s has a verified %s fill (target %.4f) the position feed does not "
                    "show yet; no order until it confirms",
                    strategy_id,
                    symbol,
                    awaiting[2],
                    awaiting[1],
                )
                continue
            qty = net_qty.get(key, 0.0)
            # Another verified reduction may have passed the original target.
            # Never buy back exposure or flip direction to recreate it.
            if qty == target or (abs(qty) < abs(target) and qty * target >= 0):
                submitted += 1
                continue
            if previous is not None and previous.target_reached:
                # A confirmed fill plus a disagreeing current snapshot needs
                # reconciliation (stale positions or newly-added exposure).
                # Neither declare the book safe nor blindly repeat the fill.
                logger.critical(
                    "%s: %s previously filled but target no longer verified; reconcile",
                    strategy_id,
                    symbol,
                )
                continue
            if target != 0 and qty * target < 0:
                logger.critical(
                    "%s: %s reversed since risk decision; reconcile", strategy_id, symbol
                )
                continue
            if key in self._tick_submitted:
                # CL-pksi bound: one emergency order per symbol per tick (a
                # second action firing in the same tick waits for the next).
                logger.warning(
                    "%s: %s already sent an emergency order this tick; next tick",
                    strategy_id,
                    symbol,
                )
                continue
            if previous is not None:
                logger.warning(
                    "%s: %s bounded retry after verified %s; ORIGINAL target %.4f",
                    strategy_id,
                    symbol,
                    previous.status,
                    target,
                )
            intent = OrderIntent(
                strategy_id=strategy_id,
                symbol=symbol,
                target_position=target,
                urgency="urgent",
            )
            # CL-pksi: persist the attempt BEFORE the broker call, and fence
            # the symbol for every other writer (this intent is the one
            # exempt owner). No durable record -> no order: a crash after an
            # unrecorded send could never be recovered without a duplicate.
            attempt = self._begin_attempt(intent, key, symbol, qty, target)
            if attempt is None:
                continue
            self._tick_submitted.add(key)
            try:
                logger.warning(
                    "%s: submitting emergency intent %s %s target %.4f (was %.4f)",
                    strategy_id,
                    intent.intent_id,
                    symbol,
                    target,
                    qty,
                )
                # An ID-only OMS is not capable of confirming risk completion.
                outcome = self.oms.submit_intent_result(intent, bypass_halt=True)
                if not isinstance(outcome, SubmissionResult):
                    outcome = SubmissionResult(intent.intent_id, SubmissionStatus.UNKNOWN)
            except Exception:
                logger.exception(
                    "%s: submit_intent failed for %s — outcome UNKNOWN",
                    strategy_id,
                    symbol,
                )
                outcome = SubmissionResult(intent.intent_id, SubmissionStatus.UNKNOWN)
            # The per-action outcome is recorded inside _apply_evidence, under
            # the attempt lock: a streamed fill that raced ahead of this
            # response must not be overwritten by the (older) response.
            resolved = self._apply_evidence(
                attempt.intent_id,
                DeriskEvidence(
                    source=EVIDENCE_SUBMISSION,
                    client_order_id=attempt.client_order_id,
                    order_id=outcome.order_id,
                    submission=outcome,
                ),
            )
            if outcome.target_reached and resolved is not None and not resolved.unresolved:
                submitted += 1
            logger.warning(
                "%s: intent %s %s target %.4f (was %.4f) -> %s / attempt %s",
                strategy_id,
                intent.intent_id,
                symbol,
                target,
                qty,
                outcome.status,
                resolved.status if resolved is not None else "?",
            )
        return submitted, actionable

    # -- CL-pksi: emergency attempts, fences and their resolution ----------

    def _unresolved_for(self, key: str) -> list[EmergencyAttempt]:
        """Unresolved attempts fencing canonical symbol ``key`` (any action)."""
        with self._attempt_lock:
            return [a for a in self._attempts.values() if a.symbol == key and a.unresolved]

    def unresolved_derisk_symbols(self) -> list[str]:
        """Sorted canonical symbols currently fenced by an unresolved attempt."""
        with self._attempt_lock:
            return sorted({a.symbol for a in self._attempts.values() if a.unresolved})

    def derisk_fence_summary(self) -> dict[str, Any]:
        """Operator view of the fences for ``/api/system`` (CL-o9sq/CL-pksi)."""
        with self._attempt_lock:
            pending = sorted(
                (a for a in self._attempts.values() if a.unresolved),
                key=lambda a: a.created_at,
            )
            return {
                "count": len({a.symbol for a in pending}),
                "symbols": sorted({a.symbol for a in pending}),
                "durable": bool(getattr(self.attempt_store, "durable", False)),
                "awaiting_position_confirmation": sorted(self._confirm_pending),
                "attempts": [
                    {
                        "symbol": a.symbol,
                        "action": a.action,
                        "status": a.status.value,
                        "client_order_id": a.client_order_id,
                        "broker_order_id": a.broker_order_id,
                        "target": a.target,
                        "requested_qty": a.requested_qty,
                        "cumulative_fill_qty": a.cumulative_fill_qty,
                        "created_at": a.created_at.isoformat(),
                        "updated_at": a.updated_at.isoformat(),
                    }
                    for a in pending
                ],
            }

    def _note_unresolved_halt(self) -> None:
        """Hold the sticky external cause while any attempt is unresolved."""
        if EXTERNAL_HALT_PREFIX + UNRESOLVED_EMERGENCY_CAUSE in self.active_halt_causes():
            self.oms.halt_new_trades()  # idempotent; keep the brake applied
            return
        self.record_external_halt(UNRESOLVED_EMERGENCY_CAUSE)

    def _sync_fence(self, key: str, route_symbol: str) -> None:
        """Mirror this symbol's unresolved attempts onto the OMS fence."""
        pending = self._unresolved_for(key)
        fence = getattr(self.oms, "fence_symbol", None)
        if pending:
            if callable(fence):
                fence(
                    route_symbol,
                    "unresolved emergency order(s): "
                    + ", ".join(f"{a.client_order_id}={a.status.value}" for a in pending),
                )
            self._note_unresolved_halt()
            return
        if key in self._confirm_pending:
            if callable(fence):
                fence(
                    route_symbol,
                    "verified emergency fill not yet visible in the position feed",
                )
            return
        release = getattr(self.oms, "release_symbol_fence", None)
        if callable(release):
            release(route_symbol)
        logger.info("no unresolved emergency order on %s — OMS fence released", key)

    def _confirm_positions(self, positions: Any) -> None:
        """Release restart confirmation fences whose leg the position feed now
        shows at (or beyond, same side) its target. Only for legs whose fill
        is already VERIFIED by broker evidence — this never resolves an order.

        CL-pksi / CL-oqos: ``positions`` is the RAW broker snapshot. It must
        pass the shared :func:`validate_broker_snapshot` rules (a list; every
        row a non-empty canonical symbol with a finite, non-bool quantity; no
        canonical duplicates) before ANY fence is released. A malformed or
        unavailable snapshot (``{}``, a non-list, a duplicate leg, a NaN
        quantity) would otherwise read as "flat" — missing exposure taken as
        zero — and release a fence while the feed still holds the pre-fill
        position, letting the cold-start reconciler close (and reverse) it
        again. Such a snapshot keeps every fence and logs why.
        """
        if not self._confirm_pending:
            return
        try:
            validated = validate_broker_snapshot(positions)
        except SnapshotUnavailableError as exc:
            logger.warning(
                "position snapshot unavailable (%s) — every confirmation fence kept: %s",
                exc,
                sorted(self._confirm_pending),
            )
            return
        net_qty = {key: float(pos.quantity) for key, pos in validated.items()}
        with self._attempt_lock:
            for key, (route, target, _action) in list(self._confirm_pending.items()):
                qty = net_qty.get(key, 0.0)
                if abs(qty - target) < 1.0 or (abs(qty) < abs(target) and qty * target >= 0):
                    logger.warning(
                        "position feed confirms verified emergency fill on %s (%.4f, target "
                        "%.4f) — confirmation fence released",
                        key,
                        qty,
                        target,
                    )
                    del self._confirm_pending[key]
                    self._sync_fence(key, route)

    def _confirm_from_broker(self) -> None:
        if not self._confirm_pending:
            return
        try:
            logger.info("reading broker positions to confirm verified emergency fills")
            positions = self.broker.get_positions()
        except Exception:
            logger.warning("position confirmation read failed — fences kept", exc_info=True)
            return
        self._confirm_positions(positions)

    def _begin_attempt(
        self, intent: OrderIntent, key: str, symbol: str, qty: float, target: float
    ) -> EmergencyAttempt | None:
        """Persist a SUBMITTING attempt and fence the symbol, BEFORE any
        broker call. Returns None (submit nothing) when it cannot be made
        durable — an unrecorded emergency order is unrecoverable on restart."""
        now = self._clock()
        attempt = EmergencyAttempt(
            intent_id=intent.intent_id,
            client_order_id=intent.intent_id,  # the OMS sends it as the client id
            action=intent.strategy_id,
            symbol=key,
            route_symbol=symbol,
            original_qty=float(qty),
            target=float(target),
            requested_qty=abs(float(qty) - float(target)),
            status=AttemptStatus.SUBMITTING,
            created_at=now,
            updated_at=now,
            episode_id=self._episodes.get(intent.strategy_id),
        )
        try:
            logger.warning(
                "%s: recording emergency attempt %s for %s before submission",
                intent.strategy_id,
                attempt.intent_id,
                key,
            )
            self.attempt_store.insert(attempt)
        except Exception:
            logger.critical(
                "%s: emergency attempt for %s could NOT be persisted — NOT submitting "
                "(fail closed); halting entries. Restore the fx_emergency_attempts store.",
                intent.strategy_id,
                key,
                exc_info=True,
            )
            self.record_external_halt(EMERGENCY_STORE_UNAVAILABLE_CAUSE)
            return None
        with self._attempt_lock:
            self._attempts[attempt.intent_id] = attempt
        fence = getattr(self.oms, "fence_symbol", None)
        if callable(fence):
            fence(symbol, f"emergency order {attempt.intent_id} submitting", attempt.intent_id)
        return attempt

    def _apply_evidence(self, intent_id: str, ev: DeriskEvidence) -> EmergencyAttempt | None:
        """Apply one piece of evidence to a known attempt, persist it, and
        sync the fence, the halt cause and the per-action outcome."""
        with self._attempt_lock:
            current = self._attempts.get(intent_id)
            if current is None:
                return None
            updated = apply_evidence(current, ev, now=self._clock())
            self._attempts[intent_id] = updated
            try:
                self.attempt_store.update(updated)
            except Exception:
                # The durable row keeps its older, still-unresolved status, so
                # a restart re-fences and re-verifies from broker evidence —
                # the safe direction. Never undo the in-memory transition.
                logger.critical(
                    "emergency attempt %s: status %s NOT persisted — restart will re-verify",
                    intent_id,
                    updated.status,
                    exc_info=True,
                )
            if updated.status is not current.status:
                logger.warning(
                    "emergency attempt %s %s %s: %s -> %s (filled %.4f / %.4f) via %s",
                    updated.intent_id,
                    updated.action,
                    updated.symbol,
                    current.status,
                    updated.status,
                    updated.cumulative_fill_qty,
                    updated.requested_qty,
                    ev.source,
                )
            key = (updated.action, updated.symbol)
            if ev.source == EVIDENCE_SUBMISSION and ev.submission is not None:
                self._derisk_results[key] = ev.submission
            if updated.status is AttemptStatus.FILLED and (
                current.unresolved or ev.source == EVIDENCE_SUBMISSION
            ):
                # Verified complete (possibly by a fill that raced ahead of
                # the placement response): a lagging snapshot must not trigger
                # a second close for this action (see previous.target_reached).
                self._derisk_results[key] = SubmissionResult(
                    updated.intent_id,
                    SubmissionStatus.FILLED,
                    target_reached=True,
                    order_id=updated.broker_order_id,
                    requested_qty=updated.requested_qty,
                )
                if current.unresolved:
                    # Verified fill, but other writers size off the position
                    # feed: keep every writer fenced until it shows the fill.
                    self._confirm_pending[updated.symbol] = (
                        updated.route_symbol,
                        updated.target,
                        updated.action,
                    )
            elif current.unresolved and not updated.unresolved:
                if updated.status in RETRYABLE_STATUSES:
                    # Verified zero fill: one retry per tick at the ORIGINAL target.
                    self._derisk_results[key] = SubmissionResult(
                        updated.intent_id, SubmissionStatus.REJECTED
                    )
                else:
                    self._derisk_results.pop(key, None)
            self._sync_fence(updated.symbol, updated.route_symbol)
            return updated

    def resolve_derisk(self, symbol: str, evidence: DeriskEvidence) -> AttemptStatus | None:
        """Resolve an emergency fence from verified broker evidence (CL-o9sq).

        ``evidence`` must identify the attempt by its client order id (or, with
        no client id, by the broker order id). Returns the attempt's new
        status, or None when the evidence matches no attempt on ``symbol``. A
        position snapshot is not evidence and cannot be expressed here.
        """
        assert isinstance(evidence, DeriskEvidence), "evidence must be DeriskEvidence"
        key = canonical_symbol(symbol)
        with self._attempt_lock:
            match = [
                a
                for a in self._attempts.values()
                if a.symbol == key
                and (
                    (evidence.client_order_id and a.client_order_id == evidence.client_order_id)
                    or (
                        evidence.client_order_id is None
                        and evidence.order_id
                        and a.broker_order_id == evidence.order_id
                    )
                )
            ]
            if not match:
                logger.info(
                    "resolve_derisk: %s evidence for %s (client=%s order=%s) matches no attempt",
                    evidence.source,
                    key,
                    evidence.client_order_id,
                    evidence.order_id,
                )
                return None
            updated = self._apply_evidence(match[0].intent_id, evidence)
        return updated.status if updated is not None else None

    def on_broker_fill(self, fill: dict[str, Any]) -> None:
        """OMS fill listener: a streamed ORDER_FILL is evidence (CL-pksi).

        Only a fill carrying one of OUR attempts' client ids counts; anything
        else (strategy fills, fills without client extensions) is ignored.
        """
        client_id = fill.get("client_order_id")
        if not client_id:
            return
        with self._attempt_lock:
            attempt = next(
                (a for a in self._attempts.values() if a.client_order_id == str(client_id)),
                None,
            )
        if attempt is None:
            return
        try:
            units = abs(float(fill.get("units") or 0.0))
        except (TypeError, ValueError):
            logger.error("emergency fill for %s has unreadable units — not evidence", client_id)
            return
        txn = str(fill.get("transaction_id") or "")
        if not txn or units <= 0:
            logger.error(
                "emergency fill for %s lacks a transaction id or units — not evidence", client_id
            )
            return
        instrument = canonical_symbol(str(fill.get("instrument") or attempt.symbol))
        if instrument != attempt.symbol:
            logger.critical(
                "emergency fill for %s is on %s, attempt is on %s — ignored; reconcile",
                client_id,
                instrument,
                attempt.symbol,
            )
            return
        logger.warning(
            "emergency fill evidence: client=%s txn=%s %s units=%.4f",
            client_id,
            txn,
            attempt.symbol,
            units,
        )
        self.resolve_derisk(
            attempt.symbol,
            DeriskEvidence(
                source=EVIDENCE_FILL_EVENT,
                client_order_id=str(client_id),
                order_id=str(fill.get("order_id") or "") or None,
                fill_transaction_id=txn,
                fill_qty=units,
            ),
        )

    def refresh_unresolved_derisk(self) -> int:
        """Query the broker for every unresolved attempt and apply what it
        says (CL-pksi). Called each health tick and by startup recovery.

        Broker calls run with NO lock held and outside any DB transaction.
        Returns how many attempts remain unresolved.
        """
        if self._recovery_failed:
            logger.warning("emergency-attempt recovery previously failed — retrying")
            self.recover_emergency_attempts()
            if self._recovery_failed:
                return -1
        with self._attempt_lock:
            pending = [a for a in self._attempts.values() if a.unresolved]
        by_client = getattr(self.broker, "get_order_by_client_id", None)
        for attempt in pending:
            evidence = self._lookup_evidence(attempt, by_client)
            if evidence is not None:
                self.resolve_derisk(attempt.symbol, evidence)
        self._confirm_from_broker()
        with self._attempt_lock:
            return sum(1 for a in self._attempts.values() if a.unresolved)

    def _lookup_evidence(self, attempt: EmergencyAttempt, by_client: Any) -> DeriskEvidence | None:
        """One broker order lookup for ``attempt`` -> evidence (None = none)."""
        try:
            logger.info(
                "emergency attempt %s (%s): querying broker order evidence",
                attempt.client_order_id,
                attempt.status,
            )
            if callable(by_client):
                order = by_client(attempt.client_order_id)
            elif attempt.broker_order_id:
                order = self.broker.get_order(attempt.broker_order_id)
            else:
                logger.warning(
                    "emergency attempt %s: no client-id lookup and no order id — fence kept",
                    attempt.client_order_id,
                )
                return None
        except BrokerOrderNotFoundError:
            return DeriskEvidence(
                source=EVIDENCE_ORDER_LOOKUP,
                client_order_id=attempt.client_order_id,
                not_found=True,
                detail="broker: no such order",
            )
        except Exception as exc:
            logger.warning(
                "emergency attempt %s: broker lookup failed (%s) — fence kept",
                attempt.client_order_id,
                type(exc).__name__,
            )
            return None
        if not isinstance(order, Order):
            logger.warning(
                "emergency attempt %s: broker lookup returned no order — fence kept",
                attempt.client_order_id,
            )
            return None
        if order.client_order_id not in (None, attempt.client_order_id):
            logger.critical(
                "emergency attempt %s: broker returned an order for client %s — ignored",
                attempt.client_order_id,
                order.client_order_id,
            )
            return None
        return DeriskEvidence(
            source=EVIDENCE_ORDER_LOOKUP,
            client_order_id=attempt.client_order_id,
            order_id=order.order_id or None,
            order_state=order.status,
            fill_transaction_id=order.fill_transaction_id,
            fill_qty=order.filled_quantity if order.fill_transaction_id is not None else None,
            filled_quantity=order.filled_quantity,
            order_qty=order.quantity if order.quantity > 0 else None,
        )

    def recover_emergency_attempts(self) -> list[EmergencyAttempt]:
        """Startup recovery (CL-pksi): re-fence every unresolved attempt from
        the durable store, restore its ORIGINAL target, hold the sticky
        external halt, then resolve what broker evidence can resolve.

        A restart is never a resolution: an attempt leaves the unresolved set
        only through broker evidence or an operator release. An unreadable
        store BLOCKS EVERY OMS SUBMISSION (an unknown outstanding order could
        be duplicated by any writer) and halts entries
        (``emergency_attempts_unavailable``); the health tick retries the
        recovery and lifts the block once it succeeds. Raises nothing, so the
        engine stays observable. Also restores every OPEN episode's FIXED
        targets, so completed legs are not reduced again.
        """
        try:
            episodes = self.attempt_store.load_open_episodes()
            loaded = self.attempt_store.load_unresolved()
            # Every attempt of an OPEN episode, completed ones included: a
            # verified fill must keep protecting its leg after a restart when
            # the position feed still shows the pre-fill quantity.
            history = self.attempt_store.load_for_episodes([e[0] for e in episodes])
        except Exception:
            logger.critical(
                "emergency attempt store unreadable — unresolved emergency orders cannot "
                "be ruled out; blocking ALL submissions and halting entries",
                exc_info=True,
            )
            self._recovery_failed = True
            block = getattr(self.oms, "block_all_submissions", None)
            if callable(block):
                block("emergency-attempt recovery failed: outstanding orders unknown")
            self.record_external_halt(EMERGENCY_STORE_UNAVAILABLE_CAUSE)
            return []
        with self._attempt_lock:
            for episode_id, action, targets in episodes:
                logger.warning(
                    "startup: restoring episode %s %s fixed targets %s", episode_id, action, targets
                )
                self._episodes[action] = episode_id
                restored = self._derisk_targets.setdefault(action, {})
                for key, value in targets.items():
                    restored.setdefault(key, value)
        logger.warning("startup: %d unresolved emergency attempt(s) loaded", len(loaded))
        with self._attempt_lock:
            for attempt in loaded:
                self._attempts[attempt.intent_id] = attempt
                self._derisk_targets.setdefault(attempt.action, {}).setdefault(
                    attempt.symbol, (attempt.route_symbol, attempt.target)
                )
                logger.critical(
                    "startup: re-fencing %s (%s %s, filled %.4f / %.4f, client=%s)",
                    attempt.symbol,
                    attempt.action,
                    attempt.status,
                    attempt.cumulative_fill_qty,
                    attempt.requested_qty,
                    attempt.client_order_id,
                )
                self._sync_fence(attempt.symbol, attempt.route_symbol)
            for attempt in sorted(history, key=lambda a: a.created_at):
                # The latest attempt per leg decides its retry/complete outcome.
                self._attempts.setdefault(attempt.intent_id, attempt)
                leg = (attempt.action, attempt.symbol)
                if attempt.status is AttemptStatus.FILLED:
                    self._derisk_results[leg] = SubmissionResult(
                        attempt.intent_id,
                        SubmissionStatus.FILLED,
                        target_reached=True,
                        order_id=attempt.broker_order_id,
                        requested_qty=attempt.requested_qty,
                    )
                elif attempt.status in RETRYABLE_STATUSES:
                    self._derisk_results[leg] = SubmissionResult(
                        attempt.intent_id, SubmissionStatus.REJECTED
                    )
                else:
                    self._derisk_results.pop(leg, None)
                if attempt.status is AttemptStatus.FILLED:
                    # Latest verified fill per symbol; released only by a
                    # position snapshot that shows it (_confirm_positions).
                    self._confirm_pending[attempt.symbol] = (
                        attempt.route_symbol,
                        attempt.target,
                        attempt.action,
                    )
            for key, (route, _target, _action) in self._confirm_pending.items():
                self._sync_fence(key, route)
        if self._recovery_failed:
            # Only now — every recovered fence is installed — may writers run.
            logger.warning("emergency-attempt recovery succeeded after earlier failure")
            self._recovery_failed = False
            unblock = getattr(self.oms, "unblock_all_submissions", None)
            if callable(unblock):
                unblock()
        remaining = self.refresh_unresolved_derisk()
        self._confirm_from_broker()
        logger.warning(
            "startup emergency recovery: %d loaded, %d still unresolved", len(loaded), remaining
        )
        return loaded

    def release_derisk_fence(self, symbol: str, *, changed_by: str, reason: str) -> list[str]:
        """Operator release of a symbol's unresolved attempts (CL-pksi).

        For an attempt broker evidence cannot settle — a partial fill that
        went terminal, or an order the broker can no longer report — AFTER the
        operator has reconciled the position by hand. Attributed and
        persisted; returns the released intent ids. Does NOT resume trading:
        the halt cause still needs ``/api/system/resume``.
        """
        actor = (changed_by or "").strip()
        why = (reason or "").strip()
        if not actor or not why:
            msg = "release_derisk_fence requires non-empty changed_by and reason"
            raise ValueError(msg)
        key = canonical_symbol(symbol)
        released: list[EmergencyAttempt] = []
        with self._attempt_lock:
            for attempt in [a for a in self._attempts.values() if a.symbol == key and a.unresolved]:
                logger.critical(
                    "OPERATOR RELEASE of emergency attempt %s %s (%s, filled %.4f / %.4f) "
                    "by %s: %s",
                    attempt.client_order_id,
                    key,
                    attempt.status,
                    attempt.cumulative_fill_qty,
                    attempt.requested_qty,
                    actor,
                    why,
                )
                updated = replace(
                    apply_evidence(
                        attempt,
                        DeriskEvidence(source=EVIDENCE_OPERATOR, detail=f"{actor}: {why}"),
                        now=self._clock(),
                    ),
                    status=AttemptStatus.OPERATOR_RELEASED,
                )
                # Persist FIRST: a release that cannot be recorded must not
                # silently lift a fence that a restart would bring back.
                self.attempt_store.update(updated)
                self._attempts[attempt.intent_id] = updated
                self._derisk_results.pop((updated.action, updated.symbol), None)
                released.append(updated)
            if released:
                self._sync_fence(key, released[0].route_symbol)
        return [a.intent_id for a in released]

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
                    "kill switch %s: UNARMED — missing inputs %s; this switch will NEVER fire",
                    sw.name,
                    missing,
                )
            else:
                logger.info(
                    "kill switch %s: ARMED (action=%s, inputs=%s)",
                    sw.name,
                    sw.action,
                    ", ".join(sw.required_keys) or "self-feeding",
                )

    def reset_daily(self, *, clear_causes: bool = True) -> None:
        """Re-arm the once-per-day trigger dedup. Called by the live engine
        health tick at UTC-day rollover (CL-i4tx) and by /api/system/resume
        — before that, nothing called this and a fired switch stayed deduped
        until restart.

        ``clear_causes`` (CL-ssoh, P1): the AUTOMATIC UTC rollover MUST NOT
        clear ``_active_halt_causes`` while a halt is still in force — doing
        so left the OMS halted but the cause set empty, so
        ``attempt_auto_resume`` read it as a manual halt and NEVER lifted it
        (permanent deadlock, common on the weekend rollover when the price-age
        key is omitted so ``check`` can't re-populate the cause). The rollover
        path passes ``clear_causes=False``: ``_triggered_today`` is still
        re-armed (so ``stale_prices`` can fire again) while the active cause
        stays sticky and auto-resume keeps ownership. Only a manual
        ``/api/system/resume`` (operator declaring the halt reviewed) clears
        the causes — that keeps the default ``True``."""
        if self._triggered_today:
            logger.info(
                "Kill switches re-armed for the new UTC day (had fired: %s)",
                sorted(self._triggered_today),
            )
        self._triggered_today.clear()
        # Reset completed/rejected decisions only. Never discard an unresolved
        # order fence (or its ORIGINAL target) at midnight or on a resume —
        # CL-pksi: the fence set is the durable attempt record, not results.
        with self._attempt_lock:
            unresolved = {a.action for a in self._attempts.values() if a.unresolved}
            # Verified fills awaiting position confirmation are recoverable
            # state too: their episodes stay OPEN so a restart restores them.
            unresolved |= {c[2] for c in self._confirm_pending.values()}
            pending_symbols = sorted({a.symbol for a in self._attempts.values() if a.unresolved})
            dropped = [a for a in self._episodes if a not in unresolved]
            closing = [self._episodes.pop(a) for a in dropped]
            self._derisk_targets = {
                action: targets
                for action, targets in self._derisk_targets.items()
                if action in unresolved
            }
            self._derisk_results = {
                key: result for key, result in self._derisk_results.items() if key[0] in unresolved
            }
        if closing:
            try:
                self.attempt_store.close_episodes(closing, self._clock())
            except Exception:
                # A restart would restore these targets (fewer orders, never
                # more); loud so the operator can close them by hand.
                logger.critical(
                    "emergency episodes %s could not be closed in the store", closing, exc_info=True
                )
        if clear_causes:
            with self._cause_lock:
                if self._active_halt_causes:
                    logger.warning(
                        "Clearing ALL halt causes on manual resume: %s",
                        sorted(self._active_halt_causes),
                    )
                self._active_halt_causes.clear()
                if pending_symbols:
                    # CL-pksi: an operator resume cannot clear ambiguity. The
                    # cause (and the brake) stay until no attempt is unresolved.
                    logger.critical(
                        "Resume WITHHELD: unresolved emergency orders on %s — keeping %s",
                        pending_symbols,
                        EXTERNAL_HALT_PREFIX + UNRESOLVED_EMERGENCY_CAUSE,
                    )
                    self._active_halt_causes.add(EXTERNAL_HALT_PREFIX + UNRESOLVED_EMERGENCY_CAUSE)
                    self.oms.halt_new_trades()

    def active_halt_causes(self) -> frozenset[str]:
        """Snapshot of the causes currently holding the halt (CL-d7ex)."""
        with self._cause_lock:
            return frozenset(self._active_halt_causes)

    def record_external_halt(self, cause: str) -> None:
        """Record a NON-switch halt and halt the OMS (CL-d7ex).

        Called by the live engine's startup (operator entry-paused start,
        cold-start reconciliation mismatch/failure/unavailable) and by
        ``/api/system/halt``. The cause is stored as ``external:<cause>`` —
        never a data gate — so ``attempt_auto_resume`` can never lift this
        halt; only a manual ``/api/system/resume`` (``reset_daily()``) does.

        The OMS halt is (re)applied UNDER the cause lock: if a concurrent
        health-tick auto-resume drained the data-gate causes and resumed the
        OMS just before this call took the lock, the halt is re-applied here,
        so the operator's halt always wins the race. ``halt_new_trades`` is
        idempotent.
        """
        assert cause and cause.strip(), "external halt cause must be non-empty"
        name = cause if cause.startswith(EXTERNAL_HALT_PREFIX) else EXTERNAL_HALT_PREFIX + cause
        with self._cause_lock:
            logger.warning(
                "Recording external halt cause %s — halting new trades; "
                "auto-resume will NOT lift this (manual /api/system/resume only)",
                name,
            )
            self._active_halt_causes.add(name)
            self.oms.halt_new_trades()
            # Postcondition checked INSIDE the lock: a concurrent manual
            # resume (reset_daily) may legitimately clear the cause the
            # instant the lock is released.
            assert name in self._active_halt_causes

    def _oms_is_halted(self) -> bool | None:
        """The OMS local halt flag, or None when the OMS cannot report it
        (CL-d7ex). Only a real ``bool`` counts — a mock's auto-attribute or a
        legacy OMS without ``is_halted`` is "unknown", never "halted"."""
        probe = getattr(self.oms, "is_halted", None)
        if not callable(probe):
            return None
        try:
            value = probe()
        except Exception:
            logger.warning("kill switches: oms.is_halted() failed", exc_info=True)
            return None
        return value if isinstance(value, bool) else None

    def _adopt_unattributed_halt(self) -> None:
        """Before a switch adds its own cause, adopt a pre-existing halt that
        has NO recorded cause (CL-d7ex). Without this, a halt from a path that
        did not call ``record_external_halt`` would look data-gate-only once
        ``stale_prices`` fired on top of it, and auto-resume would lift a
        halt it never created."""
        with self._cause_lock:
            if self._active_halt_causes:
                return  # an owned halt is already being tracked
            if self._oms_is_halted() is not True:
                return
            logger.warning(
                "OMS already halted with no recorded cause — adopting it as "
                "sticky cause %s so a data-gate auto-resume cannot lift it",
                _UNATTRIBUTED_PRIOR_HALT,
            )
            self._active_halt_causes.add(_UNATTRIBUTED_PRIOR_HALT)

    def attempt_auto_resume(self, context: dict[str, Any]) -> bool:
        """Auto-lift a halt caused ONLY by data-availability gates whose
        condition has cleared (CL-nxjx). Called each health tick AFTER
        check(). Returns True when it resumed the OMS.

        A DATA-gate cause (stale_prices) is re-evaluated against the CURRENT
        context; if its condition is no longer true (data returned), it is
        removed from the active causes AND re-armed (dropped from
        ``_triggered_today`` so it can fire again if data goes stale later).
        RISK causes (drawdown/VIX/reconciliation/…) are NEVER auto-removed —
        they hold the halt sticky until a manual resume, so a human reviews.
        Resume fires only when the cause set drains to empty via data-gate
        clearing; an already-empty cause set means a non-switch (manual)
        halt, which is left untouched.

        CL-d7ex: operator / startup / cold-start-reconciliation halts are
        ``external:*`` causes (``record_external_halt``, or adopted by
        ``_adopt_unattributed_halt``) — never data gates — so a halt whose
        cause set includes one is never lifted here.
        """
        # CL-d7ex: the drain-and-resume decision runs under the cause lock so
        # a concurrent record_external_halt either lands before (blocks the
        # resume) or after (re-halts) — never lost in between.
        with self._cause_lock:
            return self._attempt_auto_resume_locked(context)

    def _attempt_auto_resume_locked(self, context: dict[str, Any]) -> bool:
        """Body of :meth:`attempt_auto_resume`; caller holds ``_cause_lock``."""
        if not self._active_halt_causes:
            return False  # nothing WE halted for — don't touch a manual halt
        by_name = {sw.name: sw for sw in self.switches}
        cleared: list[str] = []
        for name in list(self._active_halt_causes):
            if name not in _DATA_GATE_SWITCHES:
                continue  # risk / external cause — stays sticky
            sw = by_name.get(name)
            if sw is None:
                continue
            try:
                still_bad = bool(sw.condition(context))
            except Exception:
                # Can't confirm the gate cleared — leave the halt in place.
                logger.debug("auto-resume: %s re-eval failed", name, exc_info=True)
                continue
            if not still_bad:
                self._active_halt_causes.discard(name)
                self._triggered_today.discard(name)  # re-arm
                cleared.append(name)
                logger.info(
                    "Data-gate %s cleared — condition no longer true",
                    name,
                )
        if cleared and self._active_halt_causes:
            # CL-d7ex: make the withheld resume visible — the data gate is
            # gone but a sticky (risk / external) cause still holds the halt.
            logger.warning(
                "auto-resume WITHHELD: data gate(s) %s cleared but halt still "
                "held by %s — manual /api/system/resume required",
                cleared,
                sorted(self._active_halt_causes),
            )
        if not self._active_halt_causes:
            logger.info("auto-resume: cause set drained — resuming OMS")
            try:
                self.oms.resume_trades()
            except Exception:
                logger.exception("auto-resume: oms.resume_trades() failed")
                return False
            logger.critical(
                "AUTO-RESUMED new trades — the only halt cause(s) were "
                "data-availability gates that have cleared (e.g. price "
                "stream recovered). Risk switches were NOT involved.",
            )
            return True
        return False
