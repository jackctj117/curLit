"""Kill switch manager — automatic circuit breakers for risk protection."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.monitoring.metrics import kill_switch_triggered

logger = logging.getLogger(__name__)


@dataclass
class KillSwitch:
    name: str
    condition: Callable[[dict[str, Any]], bool]
    action: str
    armed: bool = True


class KillSwitchManager:
    def __init__(self, broker: Any, oms: Any, config: dict[str, Any] | None = None) -> None:
        self.broker = broker
        self.oms = oms
        self.config = config or {}
        self._triggered_today: set[str] = set()
        self.switches = self._build_switches()

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
        ]

    def check(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        triggered = []
        for sw in self.switches:
            if not sw.armed or sw.name in self._triggered_today:
                continue
            try:
                if sw.condition(context):
                    logger.critical("KILL SWITCH: %s triggered — action=%s context=%s",
                                     sw.name, sw.action,
                                     {k: v for k, v in context.items() if k in ("daily_pnl_pct", "portfolio_dd", "vix_level")})
                    self._execute_action(sw.action)
                    self._triggered_today.add(sw.name)
                    kill_switch_triggered.labels(switch_name=sw.name, action=sw.action).inc()
                    triggered.append({"switch": sw.name, "action": sw.action, "context": context})
                else:
                    logger.debug("Kill switch %s: OK (value=%s)", sw.name,
                                  {k: v for k, v in context.items() if k in ("daily_pnl_pct", "portfolio_dd", "vix_level")})
            except Exception:
                logger.exception("Kill switch %s check failed", sw.name)
        return triggered

    def _execute_action(self, action: str) -> None:
        if action == "halt_new":
            self.oms.halt_new_trades()
        elif action in ("flatten_all", "reduce_50pct"):
            logger.warning("Action '%s' requires broker integration — stub", action)

    def reset_daily(self) -> None:
        self._triggered_today.clear()
