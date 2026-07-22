"""Kill-switch risk-context builder (CL-i4tx).

Before this module the live-engine health tick passed the
KillSwitchManager a context of ``{"equity": ...}`` and nothing else, so
every switch that needed ``daily_pnl_pct``, ``portfolio_dd``, a vol
index, a price age, or a position-mismatch flag sat at its safe default
forever — the kill-switch subsystem was a facade (code review
2026-07-21 §2.2 / §6.1.3, "Option B": wire the context for real).

``RiskContextBuilder.build(equity)`` assembles the REAL inputs each
health tick:

* ``equity`` — passed straight through from ``broker.get_account()``.
* ``daily_pnl_pct`` — equity vs the UTC day-start equity. Day-start
  and the running peak persist across restarts in
  ``data/risk_context_state.json`` (atomic tmp+``os.replace``, same
  idiom as ``data/equity_trailing_stop_state.json``); a redeploy must
  not grant a fresh daily-loss budget.
* ``portfolio_dd`` — equity vs the persisted running peak (mirrors the
  equity trailing stop's peak semantics, but kept in this builder's own
  state so context assembly never depends on switch-evaluation order).
* ``vix_level`` / ``vix_change_1d`` — daily VIX closes via the
  DataProvider (yfinance ``^VIX`` ingest). Daily granularity: the
  "intraday" +50% clause is approximated by close-over-close change.
* ``cvix_zscore`` — z-score of the CVIX realized-vol proxy (CL-gr8o)
  via the shared :func:`vol_regime.compute_vol_z_score` definition.
* ``price_stream_age_sec`` — seconds since the freshest tick in the
  engine's ``_last_prices`` (or since builder construction if the
  stream never delivered). Only emitted inside the FX trading window so
  the weekend close cannot false-trigger ``stale_prices``.
* ``position_mismatch`` — supplied by the engine's periodic
  broker-vs-internal alignment check (``PositionReconciler
  .check_alignment``); ``None`` (unknown) omits the key.

Every input is fail-soft: an unavailable value omits its key (the
switch conditions treat a missing key as "no opinion"), it never
fabricates a safe-looking number. The one deliberate exception is the
state file: a CORRUPT state file raises at construction — silently
resetting day-start/peak equity would hand a drawdown a fresh budget,
same posture as ``EquityTrailingStop``.

``consume_day_rollover()`` tells the health tick when the UTC day
changed so it can call ``KillSwitchManager.reset_daily()`` — before
CL-i4tx nothing ever called it and a fired switch stayed deduped until
process restart.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.strategies.vol_regime import compute_vol_z_score

logger = logging.getLogger(__name__)


_DEFAULT_STATE_PATH: Path = Path("data/risk_context_state.json")
_STATE_VERSION: int = 1

_VIX_SERIES: str = "VIX"
_CVIX_SERIES: str = "CVIX"
# Daily closes: a handful of rows is enough for level + 1d change even
# across weekends/holidays.
_VIX_CHANGE_LOOKBACK_DAYS: int = 10
# Same lookback the carry_vol/rate_diff regime filters use for the CVIX
# z-score — the kill switch should score vol against the same baseline
# the strategies do.
_DEFAULT_CVIX_Z_LOOKBACK_DAYS: int = 60


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class RiskContextBuilder:
    """Assembles the kill-switch context dict each health tick (CL-i4tx).

    All collaborators are injectable for tests: ``data_provider`` (the
    DB-backed DataProvider or a fake), ``last_prices`` (a live reference
    to the engine's tick dict), ``position_mismatch`` (callable returning
    True/False/None-unknown), ``in_trading_window`` (the engine's FX
    session predicate) and ``clock``. ``state_path=None`` keeps
    day-start/peak equity in memory only (unit tests).
    """

    data_provider: Any | None = None
    last_prices: Mapping[str, Mapping[str, Any]] | None = None
    position_mismatch: Callable[[], bool | None] | None = None
    in_trading_window: Callable[[datetime], bool] | None = None
    state_path: Path | None = _DEFAULT_STATE_PATH
    clock: Callable[[], datetime] = field(default=_utc_now)
    cvix_z_lookback_days: int = _DEFAULT_CVIX_Z_LOOKBACK_DAYS

    def __post_init__(self) -> None:
        if self.state_path is not None and not isinstance(self.state_path, Path):
            self.state_path = Path(self.state_path)
        self._day: str | None = None
        self._day_start_equity: float | None = None
        self._peak_equity: float | None = None
        self._day_rolled = False
        self._started_at = self.clock()
        self._load()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def provided_keys(self) -> set[str]:
        """Context keys this builder can genuinely supply given its wiring.

        Consumed by ``KillSwitchManager.log_arming`` at engine boot so the
        operator sees per-switch ARMED/UNARMED truth instead of trusting
        the Arch §5.5 docs.
        """
        keys = {"equity", "daily_pnl_pct", "portfolio_dd"}
        if self.data_provider is not None:
            keys |= {"vix_level", "vix_change_1d", "cvix_zscore"}
        if self.last_prices is not None:
            keys.add("price_stream_age_sec")
        if self.position_mismatch is not None:
            keys.add("position_mismatch")
        return keys

    def consume_day_rollover(self) -> bool:
        """True exactly once after a UTC-day rollover was observed.

        The health tick uses this to call ``reset_daily()`` on the manager
        BEFORE the day's first ``check()`` so yesterday's fired switches
        can re-arm.
        """
        rolled = self._day_rolled
        self._day_rolled = False
        return rolled

    def build(self, equity: float) -> dict[str, Any]:
        """Assemble the kill-switch context for one health tick.

        Never raises: each input is computed independently and a failure
        (logged) simply omits that key — a broken data path must not take
        the whole safety net down with it (the manager separately fails
        CLOSED after repeated per-switch evaluation errors).
        """
        now = self.clock()
        eq = float(equity)
        ctx: dict[str, Any] = {"equity": eq}

        try:
            self._update_equity_state(now, eq)
            if self._day_start_equity is not None and self._day_start_equity > 0:
                ctx["daily_pnl_pct"] = eq / self._day_start_equity - 1.0
            if self._peak_equity is not None and self._peak_equity > 0:
                ctx["portfolio_dd"] = eq / self._peak_equity - 1.0
        except Exception:
            logger.exception("risk context: equity state update failed")

        if self.data_provider is not None:
            self._add_vix(ctx, now)
            self._add_cvix(ctx, now)
        if self.last_prices is not None:
            self._add_price_age(ctx, now)
        if self.position_mismatch is not None:
            self._add_position_mismatch(ctx)
        return ctx

    # ------------------------------------------------------------------
    # Input assembly
    # ------------------------------------------------------------------

    def _update_equity_state(self, now: datetime, equity: float) -> None:
        if equity <= 0:
            # A zero/negative mark is a data problem, not a rollover or a
            # peak observation — leave persisted state alone.
            return
        today = now.date().isoformat()
        changed = False
        if self._day != today:
            if self._day is not None:
                self._day_rolled = True
                logger.info(
                    "risk context: UTC day rollover %s -> %s; day-start equity "
                    "%.2f -> %.2f",
                    self._day, today, self._day_start_equity or 0.0, equity,
                )
            self._day = today
            self._day_start_equity = equity
            changed = True
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity
            changed = True
        if changed:
            self._save()

    def _add_vix(self, ctx: dict[str, Any], now: datetime) -> None:
        """Daily VIX level + close-over-close change. Absent data omits."""
        provider = self.data_provider
        if provider is None:
            return
        try:
            start = now - timedelta(days=_VIX_CHANGE_LOOKBACK_DAYS)
            series = provider.get_series(_VIX_SERIES, start, now)
            values = [float(v) for v in list(series)] if series is not None else []
        except Exception as exc:
            # warning, not exception — no traceback retention in the 60s
            # health loop (CL-2yta).
            logger.warning(
                "risk context: VIX fetch failed: %s: %s",
                type(exc).__name__, exc,
            )
            return
        if not values:
            return
        ctx["vix_level"] = values[-1]
        if len(values) >= 2 and values[-2] > 0:
            ctx["vix_change_1d"] = values[-1] / values[-2] - 1.0

    def _add_cvix(self, ctx: dict[str, Any], now: datetime) -> None:
        """CVIX z-score via the shared vol_regime definition. Absent omits."""
        provider = self.data_provider
        if provider is None:
            return
        try:
            latest = provider.get_latest_value(_CVIX_SERIES, now)
            if latest is None:
                return
            ctx["cvix_zscore"] = compute_vol_z_score(
                provider, _CVIX_SERIES, self.cvix_z_lookback_days, now,
            )
        except Exception as exc:
            logger.warning(
                "risk context: CVIX z-score failed: %s: %s",
                type(exc).__name__, exc,
            )

    def _add_price_age(self, ctx: dict[str, Any], now: datetime) -> None:
        """Age of the freshest tick — a dead price stream, not a slow symbol.

        Emitted only inside the trading window: outside it (weekend FX
        close) no ticks arrive by design and a staleness trip would halt
        Monday's open until a manual resume.
        """
        prices = self.last_prices
        if prices is None:
            return
        try:
            if self.in_trading_window is not None and not self.in_trading_window(now):
                return
            freshest: datetime | None = None
            for tick in prices.values():
                ts = _parse_tick_ts(tick)
                if ts is not None and (freshest is None or ts > freshest):
                    freshest = ts
            # No parseable tick yet: measure from builder construction so a
            # stream that NEVER connects still trips stale_prices.
            reference = freshest or self._started_at
            ctx["price_stream_age_sec"] = max(
                0.0, (now - reference).total_seconds(),
            )
        except Exception as exc:
            logger.warning(
                "risk context: price-age computation failed: %s: %s",
                type(exc).__name__, exc,
            )

    def _add_position_mismatch(self, ctx: dict[str, Any]) -> None:
        supplier = self.position_mismatch
        if supplier is None:
            return
        try:
            value = supplier()
        except Exception as exc:
            logger.warning(
                "risk context: position-mismatch supplier failed: %s: %s",
                type(exc).__name__, exc,
            )
            return
        if value is not None:
            ctx["position_mismatch"] = bool(value)

    # ------------------------------------------------------------------
    # Persistence — atomic tmp+os.replace, same idiom as the equity
    # trailing stop and polymarket loss caps.
    # ------------------------------------------------------------------

    def _save(self) -> None:
        path = self.state_path
        if path is None:
            return
        payload = {
            "version": _STATE_VERSION,
            "day": self._day,
            "day_start_equity": self._day_start_equity,
            "peak_equity": self._peak_equity,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True))
            os.replace(tmp, path)
        except OSError:
            # Disk trouble must not kill the health tick; the in-memory
            # state stays correct for this process's lifetime.
            logger.exception("risk context: state save failed (%s)", path)

    def _load(self) -> None:
        """Missing file -> fresh state; corrupt file raises (fail loud —
        a silently reset day-start/peak grants a drawdown a fresh budget)."""
        path = self.state_path
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text())
            if raw.get("version") != _STATE_VERSION:
                msg = f"unsupported state version {raw.get('version')!r}"
                raise ValueError(msg)
            day = raw.get("day")
            day_start = raw.get("day_start_equity")
            peak = raw.get("peak_equity")
            self._day = str(day) if day is not None else None
            self._day_start_equity = (
                float(day_start) if day_start is not None else None
            )
            self._peak_equity = float(peak) if peak is not None else None
        except (ValueError, KeyError, TypeError) as exc:
            msg = (
                f"risk-context state at {path} is corrupt or unreadable "
                f"({exc}) — refusing to start with reset day-start/peak "
                f"equity; repair or remove the file"
            )
            raise ValueError(msg) from exc
        logger.info(
            "risk context: loaded state from %s (day=%s day_start=%s peak=%s)",
            path, self._day, self._day_start_equity, self._peak_equity,
        )


def _parse_tick_ts(tick: Mapping[str, Any]) -> datetime | None:
    """Timestamp of one price tick, or None if absent/unparseable.

    Brokers disagree on the key and the format: PaperBroker yields
    ``ts`` as ``datetime.isoformat()``; OANDA's stream yields RFC3339
    with nanosecond fractions and a ``Z`` suffix (fromisoformat on
    3.11+ truncates the extra digits). Naive stamps are assumed UTC.
    """
    raw = tick.get("ts") or tick.get("time") or tick.get("timestamp")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    else:
        text = str(raw).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            logger.debug("risk context: unparseable tick timestamp %r", raw)
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
