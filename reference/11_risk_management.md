# 11 — Risk Management

Position sizing, kill switches, regime-aware exposure, correlation monitoring.

## Position Sizer

**Location:** `src/risk/position_sizer.py`
**Purpose:** Compute position sizes using fixed fractional, vol targeting, and Kelly sizing.

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class PositionSizer:
    @staticmethod
    def fixed_fractional(account_equity: float, risk_per_trade_pct: float,
                          stop_distance: float) -> float:
        """Risk a fixed % of equity per trade based on stop distance."""
        risk_amount = account_equity * risk_per_trade_pct
        return risk_amount / stop_distance
    
    @staticmethod
    def vol_target(account_equity: float, target_vol: float,
                    asset_vol: float) -> float:
        """Size to achieve target portfolio vol contribution."""
        if asset_vol == 0:
            return 0
        return account_equity * target_vol / asset_vol
    
    @staticmethod
    def kelly(edge: float, odds: float = 1.0) -> float:
        """Kelly criterion. Use fractional (e.g., 0.25 * Kelly) in practice."""
        if odds <= 0:
            return 0
        f = (edge * odds - (1 - edge)) / odds
        return max(0, min(f, 1))
    
    @staticmethod
    def conviction_scaled(base_size: float, signal_strength: float,
                            min_mult: float = 0.5, max_mult: float = 1.5) -> float:
        """Scale base size by signal conviction."""
        mult = min_mult + (max_mult - min_mult) * np.clip(signal_strength, 0, 1)
        return base_size * mult
```

## Regime-Aware Sizer

**Location:** `src/risk/regime_sizer.py`
**Purpose:** Wraps position sizer with regime-based scaling. Reduces size in volatile/drawdown regimes.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SizingAdjustment:
    base_size: float
    final_size: float
    multipliers: dict[str, float]
    reasons: list[str]


@dataclass  
class RegimeAwareSizer:
    base_sizer: PositionSizer
    
    vix_thresholds: dict = field(default_factory=lambda: {
        15: 1.10,   # Low vol → can size up
        20: 1.00,   # Normal
        25: 0.75,   # Elevated → reduce
        30: 0.50,   # High vol
        40: 0.25,   # Crisis
    })
    
    dd_thresholds: dict = field(default_factory=lambda: {
        0.00: 1.00,
        0.05: 0.85,
        0.10: 0.65,
        0.15: 0.40,
        0.20: 0.20,  # Heavily reduced near max acceptable DD
    })
    
    correlation_regime: dict = field(default_factory=lambda: {
        'normal': 1.00,
        'stressed': 0.75,
        'crisis': 0.50,
    })
    
    def size(self, account_equity: float, asset_vol: float, 
              context: dict) -> SizingAdjustment:
        base = self.base_sizer.vol_target(
            account_equity, 
            target_vol=context.get('target_vol', 0.10),
            asset_vol=asset_vol,
        )
        
        multipliers = {}
        reasons = []
        
        # VIX/CVIX regime
        vix = context.get('vix', 20)
        vix_mult = self._lookup_threshold(vix, self.vix_thresholds)
        multipliers['vix'] = vix_mult
        if vix_mult < 1.0:
            reasons.append(f'VIX={vix:.1f} reducing by {(1-vix_mult)*100:.0f}%')
        
        # Drawdown
        dd = context.get('current_drawdown', 0)
        dd_mult = self._lookup_threshold(abs(dd), self.dd_thresholds)
        multipliers['drawdown'] = dd_mult
        if dd_mult < 1.0:
            reasons.append(f'DD={dd:.1%} reducing by {(1-dd_mult)*100:.0f}%')
        
        # Correlation regime
        corr_regime = context.get('correlation_regime', 'normal')
        corr_mult = self.correlation_regime[corr_regime]
        multipliers['correlation'] = corr_mult
        if corr_mult < 1.0:
            reasons.append(f'{corr_regime} correlation regime '
                            f'reducing by {(1-corr_mult)*100:.0f}%')
        
        # Recent overnight gap (geopolitical / weekend risk)
        if context.get('approaching_weekend', False):
            multipliers['weekend'] = 0.80
            reasons.append('Weekend risk reducing by 20%')
        
        # FOMC/major event coming up
        if context.get('major_event_within_24h', False):
            multipliers['event_risk'] = 0.50
            reasons.append('Major event in 24h reducing by 50%')
        
        final = base * np.prod(list(multipliers.values()))
        
        return SizingAdjustment(
            base_size=base,
            final_size=final,
            multipliers=multipliers,
            reasons=reasons,
        )
    
    def _lookup_threshold(self, value: float, thresholds: dict) -> float:
        sorted_t = sorted(thresholds.items())
        result = sorted_t[0][1]
        for threshold, mult in sorted_t:
            if value >= threshold:
                result = mult
        return result
```

## Kill Switch Manager

**Location:** `src/risk/kill_switch.py`
**Purpose:** Automated emergency stops for various failure conditions.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class KillSwitch:
    name: str
    condition: Callable[[dict], bool]
    action: str  # 'halt_strategy', 'flatten_all', 'reduce_50pct'
    cooldown: timedelta = timedelta(hours=1)
    last_triggered: datetime = None
    triggered_count: int = 0


class KillSwitchManager:
    def __init__(self, oms, broker, alert_fn: Callable[[str, str], None]):
        self.oms = oms
        self.broker = broker
        self.alert = alert_fn
        self.kill_switches: list[KillSwitch] = []
        self.halted_strategies: set[str] = set()
        self.global_halt: bool = False
    
    def register_default_switches(self):
        self.kill_switches.extend([
            # Drawdown-based
            KillSwitch(
                name='daily_loss_limit',
                condition=lambda ctx: ctx.get('daily_pnl_pct', 0) < -0.02,
                action='halt_strategy',
            ),
            KillSwitch(
                name='weekly_loss_limit', 
                condition=lambda ctx: ctx.get('weekly_pnl_pct', 0) < -0.05,
                action='reduce_50pct',
            ),
            KillSwitch(
                name='monthly_loss_limit',
                condition=lambda ctx: ctx.get('monthly_pnl_pct', 0) < -0.08,
                action='flatten_all',
            ),
            # Reconciliation issues
            KillSwitch(
                name='broker_state_mismatch',
                condition=lambda ctx: ctx.get('reconciliation_failures', 0) > 3,
                action='halt_strategy',
                cooldown=timedelta(minutes=15),
            ),
            # Data quality
            KillSwitch(
                name='stale_prices',
                condition=lambda ctx: ctx.get('max_price_staleness_sec', 0) > 300,
                action='halt_strategy',
                cooldown=timedelta(minutes=5),
            ),
            # Sudden volatility
            KillSwitch(
                name='vix_spike',
                condition=lambda ctx: ctx.get('vix_change_pct', 0) > 0.30,
                action='reduce_50pct',
            ),
            # Order rejection cascade
            KillSwitch(
                name='order_rejection_cascade',
                condition=lambda ctx: ctx.get('rejection_rate_5min', 0) > 0.20,
                action='halt_strategy',
            ),
            # Portfolio-level correlation
            KillSwitch(
                name='portfolio_correlation_crisis',
                condition=lambda ctx: ctx.get('correlation_regime') == 'crisis',
                action='reduce_50pct',
            ),
            KillSwitch(
                name='strategy_correlation_spike',
                condition=lambda ctx: ctx.get('max_pair_corr', 0) > 0.9,
                action='reduce_50pct',
            ),
            KillSwitch(
                name='single_strategy_drawdown',
                condition=lambda ctx: any(
                    dd < -0.25 for dd in ctx.get('per_strategy_dd', {}).values()
                ),
                action='halt_strategy',
            ),
        ])
    
    async def check_all(self, context: dict, strategy_id: str = None):
        for ks in self.kill_switches:
            if (ks.last_triggered and 
                datetime.utcnow() - ks.last_triggered < ks.cooldown):
                continue
            
            try:
                if ks.condition(context):
                    await self._trigger(ks, strategy_id)
            except Exception as e:
                logger.exception(f"Kill switch {ks.name} eval error: {e}")
    
    async def _trigger(self, ks: KillSwitch, strategy_id: str = None):
        ks.last_triggered = datetime.utcnow()
        ks.triggered_count += 1
        
        msg = (f"KILL SWITCH TRIGGERED: {ks.name} → {ks.action}"
               + (f" for {strategy_id}" if strategy_id else " globally"))
        logger.critical(msg)
        self.alert('CRITICAL', msg)
        
        if ks.action == 'halt_strategy' and strategy_id:
            self.halted_strategies.add(strategy_id)
        elif ks.action == 'flatten_all':
            await self._flatten_all_positions()
            self.global_halt = True
        elif ks.action == 'reduce_50pct':
            await self._reduce_positions(0.5)
    
    async def _flatten_all_positions(self):
        positions = self.broker.get_positions()
        for pos in positions:
            from src.execution.broker import Order, OrderSide, OrderType
            order = Order(
                order_id='',
                symbol=pos.symbol,
                side=OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY,
                quantity=abs(pos.quantity),
                order_type=OrderType.MARKET,
            )
            await self.broker.submit_order(order)
    
    async def _reduce_positions(self, fraction: float):
        positions = self.broker.get_positions()
        for pos in positions:
            target_size = pos.quantity * (1 - fraction)
            diff = target_size - pos.quantity
            if abs(diff) < 1:
                continue
            from src.execution.broker import Order, OrderSide, OrderType
            order = Order(
                order_id='',
                symbol=pos.symbol,
                side=OrderSide.BUY if diff > 0 else OrderSide.SELL,
                quantity=abs(diff),
                order_type=OrderType.MARKET,
            )
            await self.broker.submit_order(order)
    
    def can_trade(self, strategy_id: str) -> bool:
        return not self.global_halt and strategy_id not in self.halted_strategies
    
    def manual_resume(self, strategy_id: str = None):
        if strategy_id:
            self.halted_strategies.discard(strategy_id)
        else:
            self.global_halt = False
            self.halted_strategies.clear()
        logger.info(f"Manual resume: {strategy_id or 'global'}")
```

## Stress Test Definitions

**Location:** `src/risk/stress_scenarios.py`
**Purpose:** Historical crisis periods used for portfolio stress testing.

```python
from datetime import date


SCENARIOS = {
    'gfc_2008': (date(2008, 9, 1), date(2008, 12, 31)),
    'eurocrisis_2011': (date(2011, 7, 1), date(2011, 12, 31)),
    'taper_tantrum_2013': (date(2013, 5, 1), date(2013, 9, 30)),
    'chf_shock_2015': (date(2015, 1, 1), date(2015, 2, 28)),
    'brexit_2016': (date(2016, 6, 1), date(2016, 7, 31)),
    'covid_2020': (date(2020, 2, 15), date(2020, 4, 30)),
    'inflation_shock_2022': (date(2022, 1, 1), date(2022, 6, 30)),
    'svb_2023': (date(2023, 3, 1), date(2023, 3, 31)),
    'yen_intervention_2024': (date(2024, 7, 1), date(2024, 8, 15)),
}


def run_scenario_stress_test(strategies, data_provider, scenarios=SCENARIOS):
    """Replay strategies through historical crisis periods."""
    import pandas as pd
    
    results = {}
    for name, (start, end) in scenarios.items():
        scenario_data = data_provider.get_range(start, end)
        if len(scenario_data) == 0:
            continue
        
        portfolio_returns = pd.Series(0, index=scenario_data.index)
        for strategy in strategies:
            strat_returns = strategy.backtest(scenario_data)
            portfolio_returns = portfolio_returns.add(strat_returns, fill_value=0)
        
        cum_return = (1 + portfolio_returns).prod() - 1
        max_dd = _max_drawdown(portfolio_returns)
        worst_day = portfolio_returns.min()
        
        results[name] = {
            'period': f"{start} to {end}",
            'total_return': cum_return,
            'max_drawdown': max_dd,
            'worst_day': worst_day,
            'days': len(portfolio_returns),
        }
    
    return pd.DataFrame(results).T


def _max_drawdown(returns):
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return dd.min()
```

## Risk Context Builder

**Location:** `src/risk/context.py`
**Purpose:** Aggregate all risk-relevant info into a single context dict for sizers and kill switches.

```python
from datetime import datetime, timedelta


class RiskContextBuilder:
    def __init__(self, broker, data_provider, state_store):
        self.broker = broker
        self.data = data_provider
        self.state = state_store
    
    def build(self) -> dict:
        account = self.broker.get_account()
        
        return {
            # Account state
            'equity': account.equity,
            'leverage': self._compute_leverage(),
            
            # P&L
            'daily_pnl_pct': self._period_pnl_pct('day'),
            'weekly_pnl_pct': self._period_pnl_pct('week'),
            'monthly_pnl_pct': self._period_pnl_pct('month'),
            
            # Drawdown
            'current_drawdown': self._current_drawdown(),
            'per_strategy_dd': self._per_strategy_dd(),
            
            # Volatility
            'vix': self.data.get_latest_value('VIX'),
            'cvix': self.data.get_latest_value('CVIX'),
            'vix_change_pct': self._vix_change(),
            
            # Correlations
            'correlation_regime': self._correlation_regime(),
            'max_pair_corr': self._max_pairwise_correlation(),
            
            # Data quality
            'max_price_staleness_sec': self._max_price_staleness(),
            'reconciliation_failures': self._recent_reconciliation_failures(),
            'rejection_rate_5min': self._rejection_rate_5min(),
            
            # Calendar
            'approaching_weekend': self._approaching_weekend(),
            'major_event_within_24h': self._major_event_within_24h(),
        }
    
    def _compute_leverage(self) -> float:
        positions = self.broker.get_positions()
        gross = sum(abs(p.quantity * p.avg_price) for p in positions)
        equity = self.broker.get_account().equity
        return gross / equity if equity > 0 else 0
    
    def _period_pnl_pct(self, period: str) -> float:
        # Query state store for P&L since period start
        return 0.0  # Implementation specific
    
    def _current_drawdown(self) -> float:
        return 0.0  # From state store equity history
    
    def _per_strategy_dd(self) -> dict:
        return {}
    
    def _vix_change(self) -> float:
        return 0.0
    
    def _correlation_regime(self) -> str:
        return 'normal'
    
    def _max_pairwise_correlation(self) -> float:
        return 0.0
    
    def _max_price_staleness(self) -> float:
        return 0.0
    
    def _recent_reconciliation_failures(self) -> int:
        return 0
    
    def _rejection_rate_5min(self) -> float:
        return 0.0
    
    def _approaching_weekend(self) -> bool:
        now = datetime.utcnow()
        return now.weekday() == 4 and now.hour >= 18
    
    def _major_event_within_24h(self) -> bool:
        return False
```
