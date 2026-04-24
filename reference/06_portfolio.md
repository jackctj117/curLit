# 06 — Portfolio Coordination

Portfolio coordinator, risk parity, correlation monitoring, and P&L attribution.

## Risk Parity Computation

**Location:** `src/portfolio/risk_parity.py`
**Purpose:** Solve for weights where each strategy contributes equal risk to the portfolio.

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def risk_parity_weights(returns: pd.DataFrame, 
                         cov_matrix: np.ndarray = None,
                         target_total_vol: float = 0.10,
                         bounds: tuple = (0.0, 0.40)) -> pd.Series:
    """
    Compute risk parity weights for a set of strategies.
    
    Each strategy contributes equal absolute risk to the total portfolio.
    
    returns: DataFrame with strategies as columns, time-series of returns
    target_total_vol: Target annualized portfolio volatility
    bounds: (min, max) weight per strategy
    """
    if cov_matrix is None:
        cov_matrix = returns.cov().values * 252
    
    n = len(cov_matrix)
    strategies = returns.columns.tolist()
    
    def risk_contribution(w, cov):
        portfolio_vol = np.sqrt(w @ cov @ w)
        if portfolio_vol == 0:
            return np.zeros(n)
        marginal = cov @ w
        return w * marginal / portfolio_vol
    
    def objective(w, cov):
        rc = risk_contribution(w, cov)
        target = np.mean(rc)
        return np.sum((rc - target) ** 2)
    
    x0 = np.ones(n) / n
    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
    ]
    bounds_list = [bounds] * n
    
    result = minimize(
        objective, x0, args=(cov_matrix,),
        method='SLSQP', bounds=bounds_list, constraints=constraints
    )
    
    weights = pd.Series(result.x, index=strategies)
    
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
    scale = target_total_vol / portfolio_vol
    weights = weights * scale
    
    return weights


def diagnose_allocation(weights: pd.Series, cov_matrix: np.ndarray, 
                         returns: pd.DataFrame):
    """Print diagnostics on allocation."""
    w = weights.values
    portfolio_vol = np.sqrt(w @ cov_matrix @ w)
    
    marginal = cov_matrix @ w
    contributions = w * marginal / portfolio_vol
    
    print(f"\nTotal portfolio vol: {portfolio_vol:.2%}")
    print(f"\n{'Strategy':<30} {'Weight':>8} {'Risk Contrib':>15}")
    print("-" * 55)
    for strat, weight, rc in zip(returns.columns, w, contributions):
        rc_pct = rc / portfolio_vol * 100
        print(f"{strat:<30} {weight:>7.2%} {rc_pct:>14.1%}")
    
    print(f"\nStrategy correlation matrix:")
    print(returns.corr().round(2))


def rolling_risk_parity_weights(returns: pd.DataFrame, 
                                  window_days: int = 252,
                                  halflife_days: int = 60,
                                  target_vol: float = 0.10,
                                  refit_freq_days: int = 21) -> pd.DataFrame:
    """
    Compute time-varying risk parity weights.
    
    Uses EWMA covariance with given halflife, refits monthly.
    """
    decay = np.log(2) / halflife_days
    
    weights_history = []
    refit_dates = returns.index[window_days::refit_freq_days]
    
    for refit_date in refit_dates:
        end_idx = returns.index.get_loc(refit_date)
        start_idx = max(0, end_idx - window_days)
        window_returns = returns.iloc[start_idx:end_idx]
        
        weights_ewma = np.exp(-decay * np.arange(len(window_returns))[::-1])
        weights_ewma /= weights_ewma.sum()
        
        mean = (window_returns.values * weights_ewma[:, None]).sum(axis=0)
        centered = window_returns.values - mean
        cov = (centered * weights_ewma[:, None]).T @ centered * 252
        
        w = risk_parity_weights(window_returns, cov_matrix=cov, 
                                  target_total_vol=target_vol)
        
        weights_history.append({
            'date': refit_date,
            **w.to_dict(),
        })
    
    return pd.DataFrame(weights_history).set_index('date')
```

## Correlation Regime Monitor

**Location:** `src/portfolio/correlation_monitor.py`
**Purpose:** Detect when normally decorrelated strategies start moving together (crisis indicator).

```python
import numpy as np
import pandas as pd


class CorrelationMonitor:
    def __init__(self, baseline_window: int = 252, 
                  recent_window: int = 40,
                  alert_threshold_z: float = 2.5):
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.alert_threshold_z = alert_threshold_z
    
    def check_regime(self, strategy_returns: pd.DataFrame) -> dict:
        """
        Detect if strategies are unusually correlated right now.
        
        Returns regime assessment and recommended exposure scaling.
        """
        recent = strategy_returns.iloc[-self.recent_window:]
        baseline = strategy_returns.iloc[-(self.baseline_window + self.recent_window):-self.recent_window]
        
        recent_corr = recent.corr()
        baseline_corr = baseline.corr()
        
        mask = np.triu(np.ones_like(recent_corr, dtype=bool), k=1)
        recent_avg = recent_corr.where(mask).stack().mean()
        baseline_avg = baseline_corr.where(mask).stack().mean()
        
        rolling_corrs = []
        for i in range(self.recent_window, len(baseline)):
            window = baseline.iloc[i-self.recent_window:i]
            c = window.corr().where(mask).stack().mean()
            if not np.isnan(c):
                rolling_corrs.append(c)
        
        std = np.std(rolling_corrs)
        z = (recent_avg - baseline_avg) / std if std > 0 else 0
        
        if z > self.alert_threshold_z:
            regime = 'crisis'
            exposure_mult = 0.5
        elif z > 1.5:
            regime = 'stressed'
            exposure_mult = 0.75
        else:
            regime = 'normal'
            exposure_mult = 1.0
        
        return {
            'regime': regime,
            'recent_avg_corr': recent_avg,
            'baseline_avg_corr': baseline_avg,
            'z_score': z,
            'exposure_multiplier': exposure_mult,
        }
```

## Portfolio Coordinator

**Location:** `src/portfolio/coordinator.py`
**Purpose:** Sits above strategies, below OMS. Combines intents, applies constraints, manages allocations.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import logging
import asyncio
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.execution.oms import OrderIntent

logger = logging.getLogger(__name__)


@dataclass
class StrategyAllocation:
    strategy_id: str
    target_weight: float
    current_exposure_mult: float
    paper_mode: bool = False
    performance_override: float = 1.0


@dataclass
class PortfolioConstraints:
    max_gross_leverage: float = 3.0
    max_net_leverage: float = 2.0
    max_total_positions: int = 10
    max_notional_per_pair_pct: float = 0.35
    max_directional_exposure_per_currency: float = 0.40
    max_concurrent_same_direction: int = 5
    max_portfolio_vol: float = 0.12
    portfolio_vol_target: float = 0.10


class PortfolioCoordinator:
    def __init__(self, strategies: list, oms, broker, state_store, 
                 constraints: PortfolioConstraints = None):
        self.strategies = {s.id: s for s in strategies}
        self.oms = oms
        self.broker = broker
        self.state = state_store
        self.constraints = constraints or PortfolioConstraints()
        
        self.allocations: dict[str, StrategyAllocation] = {}
        self._last_rebalance: Optional[datetime] = None
        self._intent_lock = asyncio.Lock()
        self._conflicts_log: list = []
    
    def initialize_allocations(self, initial_weights: dict[str, float]):
        for sid, weight in initial_weights.items():
            if sid not in self.strategies:
                logger.warning(f"Unknown strategy {sid} in initial weights")
                continue
            self.allocations[sid] = StrategyAllocation(
                strategy_id=sid,
                target_weight=weight,
                current_exposure_mult=1.0,
            )
    
    async def rebalance_allocations(self, force: bool = False):
        """Recompute risk parity weights based on recent strategy returns."""
        now = datetime.utcnow()
        if (not force and self._last_rebalance is not None 
            and (now - self._last_rebalance).days < 21):
            return
        
        returns_df = self._load_strategy_returns_history(lookback_days=252 * 2)
        
        if len(returns_df) < 252:
            logger.warning("Insufficient history for risk parity; using equal weight")
            n = len(self.allocations)
            new_weights = {sid: 1.0/n for sid in self.allocations}
        else:
            new_weights = self._compute_risk_parity(returns_df)
        
        corr_regime = self._check_correlation_regime(returns_df)
        
        for sid, weight in new_weights.items():
            if sid not in self.allocations:
                continue
            alloc = self.allocations[sid]
            old_weight = alloc.target_weight
            alloc.target_weight = weight
            alloc.current_exposure_mult = corr_regime['exposure_multiplier']
            
            logger.info(f"Reallocated {sid}: {old_weight:.3f} → {weight:.3f} "
                       f"(exposure {alloc.current_exposure_mult:.2f})")
        
        self._last_rebalance = now
        self.state.record_reallocation(now, new_weights, corr_regime)
    
    def _load_strategy_returns_history(self, lookback_days: int) -> pd.DataFrame:
        query = text("""
            SELECT date, strategy_id, daily_pnl_pct 
            FROM strategy_daily_returns
            WHERE date >= :start AND strategy_id = ANY(:sids)
            ORDER BY date
        """)
        
        start = datetime.utcnow().date() - timedelta(days=lookback_days)
        with self.state.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                'start': start, 'sids': list(self.allocations.keys())
            })
        
        if df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        pivot = df.pivot(index='date', columns='strategy_id', 
                          values='daily_pnl_pct')
        return pivot.fillna(0)
    
    def _compute_risk_parity(self, returns: pd.DataFrame) -> dict[str, float]:
        from scipy.optimize import minimize
        
        cov = returns.cov().values * 252
        n = len(cov)
        strategies = returns.columns.tolist()
        
        def objective(w):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol == 0:
                return 0
            marginal = cov @ w
            rc = w * marginal / port_vol
            target = np.mean(rc)
            return np.sum((rc - target) ** 2)
        
        x0 = np.ones(n) / n
        bounds = [(0.05, 0.40)] * n
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
        
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return dict(zip(strategies, result.x))
    
    def _check_correlation_regime(self, returns: pd.DataFrame) -> dict:
        if len(returns) < 100:
            return {'regime': 'normal', 'exposure_multiplier': 1.0, 'avg_corr': 0}
        
        recent = returns.iloc[-40:]
        baseline = returns.iloc[-252:-40] if len(returns) >= 252 else returns.iloc[:-40]
        
        mask = np.triu(np.ones((len(returns.columns), len(returns.columns)), 
                                dtype=bool), k=1)
        recent_avg = recent.corr().where(mask).stack().mean()
        baseline_avg = baseline.corr().where(mask).stack().mean()
        
        rolling_corrs = []
        for i in range(40, len(baseline)):
            window = baseline.iloc[i-40:i]
            c = window.corr().where(mask).stack().mean()
            if not pd.isna(c):
                rolling_corrs.append(c)
        
        std = np.std(rolling_corrs) if rolling_corrs else 0.1
        z = (recent_avg - baseline_avg) / (std + 1e-9)
        
        if z > 2.5:
            return {'regime': 'crisis', 'exposure_multiplier': 0.5, 
                    'avg_corr': recent_avg, 'z': z}
        elif z > 1.5:
            return {'regime': 'stressed', 'exposure_multiplier': 0.75,
                    'avg_corr': recent_avg, 'z': z}
        else:
            return {'regime': 'normal', 'exposure_multiplier': 1.0,
                    'avg_corr': recent_avg, 'z': z}
    
    async def process_intents(self, raw_intents: dict[str, list[OrderIntent]]):
        """
        Process intents from all strategies, resolve conflicts, apply constraints,
        submit final orders to OMS.
        """
        async with self._intent_lock:
            scaled_intents = []
            for sid, intents in raw_intents.items():
                if sid not in self.allocations:
                    logger.warning(f"Intent from unknown strategy {sid}, ignoring")
                    continue
                alloc = self.allocations[sid]
                
                if alloc.paper_mode:
                    for intent in intents:
                        logger.info(f"[PAPER] {sid}: {intent.symbol} "
                                   f"→ {intent.target_position:.0f}")
                    continue
                
                scale = (alloc.target_weight * 
                         alloc.current_exposure_mult * 
                         alloc.performance_override)
                
                for intent in intents:
                    scaled = OrderIntent(
                        strategy_id=intent.strategy_id,
                        symbol=intent.symbol,
                        target_position=intent.target_position * scale,
                        urgency=intent.urgency,
                        max_slippage_bps=intent.max_slippage_bps,
                    )
                    scaled_intents.append(scaled)
            
            aggregated = self._aggregate_by_symbol(scaled_intents)
            feasible = self._apply_portfolio_constraints(aggregated)
            
            for symbol, intent_info in feasible.items():
                final_intent = OrderIntent(
                    strategy_id='portfolio',
                    symbol=symbol,
                    target_position=intent_info['target_position'],
                    urgency=intent_info['urgency'],
                )
                
                self.oms.submit_intent(final_intent)
                
                self.state.record_portfolio_order(
                    datetime.utcnow(),
                    symbol,
                    intent_info['target_position'],
                    intent_info['strategy_contributions'],
                )
    
    def _aggregate_by_symbol(self, intents: list[OrderIntent]) -> dict:
        by_symbol = defaultdict(lambda: {
            'target_position': 0.0,
            'urgency': 'normal',
            'strategy_contributions': {},
        })
        
        urgency_rank = {'passive': 0, 'normal': 1, 'urgent': 2}
        
        for intent in intents:
            agg = by_symbol[intent.symbol]
            agg['target_position'] += intent.target_position
            agg['strategy_contributions'][intent.strategy_id] = intent.target_position
            
            if urgency_rank[intent.urgency] > urgency_rank[agg['urgency']]:
                agg['urgency'] = intent.urgency
        
        for symbol, agg in by_symbol.items():
            contribs = agg['strategy_contributions']
            signs = [np.sign(v) for v in contribs.values() if v != 0]
            if len(signs) > 1 and len(set(signs)) > 1:
                self._conflicts_log.append({
                    'ts': datetime.utcnow(),
                    'symbol': symbol,
                    'contributions': contribs,
                    'net': agg['target_position'],
                })
                logger.info(f"Conflict on {symbol}: net {agg['target_position']:.0f}, "
                           f"contribs {contribs}")
        
        return dict(by_symbol)
    
    def _apply_portfolio_constraints(self, aggregated: dict) -> dict:
        account = self.broker.get_account()
        equity = account.equity
        
        gross_notional = sum(abs(a['target_position']) * self._get_price(sym)
                              for sym, a in aggregated.items())
        gross_leverage = gross_notional / equity
        
        if gross_leverage > self.constraints.max_gross_leverage:
            scale = self.constraints.max_gross_leverage / gross_leverage
            logger.warning(f"Gross leverage {gross_leverage:.2f}x exceeds max, "
                          f"scaling by {scale:.3f}")
            for agg in aggregated.values():
                agg['target_position'] *= scale
                for sid in agg['strategy_contributions']:
                    agg['strategy_contributions'][sid] *= scale
        
        max_pair_notional = equity * self.constraints.max_notional_per_pair_pct
        for symbol, agg in aggregated.items():
            price = self._get_price(symbol)
            notional = abs(agg['target_position']) * price
            if notional > max_pair_notional:
                scale = max_pair_notional / notional
                sign = np.sign(agg['target_position'])
                agg['target_position'] = sign * max_pair_notional / price
                for sid in agg['strategy_contributions']:
                    agg['strategy_contributions'][sid] *= scale
                logger.warning(f"{symbol} exceeded pair cap, scaled to {max_pair_notional}")
        
        currency_exposure = self._compute_currency_exposure(aggregated)
        for ccy, exposure in currency_exposure.items():
            max_exposure = equity * self.constraints.max_directional_exposure_per_currency
            if abs(exposure) > max_exposure:
                logger.warning(f"Currency {ccy} net exposure {exposure:.0f} "
                              f"exceeds max {max_exposure:.0f}")
        
        return aggregated
    
    def _compute_currency_exposure(self, aggregated: dict) -> dict[str, float]:
        exposures = defaultdict(float)
        for symbol, agg in aggregated.items():
            base, quote = symbol[:3], symbol[3:]
            notional = agg['target_position'] * self._get_price(symbol)
            exposures[base] += notional
            exposures[quote] -= notional
        return dict(exposures)
    
    def _get_price(self, symbol: str) -> float:
        try:
            bid, ask = self.broker.get_price(symbol)
            return (bid + ask) / 2
        except Exception:
            return 1.0
    
    def add_strategy(self, strategy, initial_paper_days: int = 30):
        sid = strategy.id
        if sid in self.strategies:
            raise ValueError(f"Strategy {sid} already exists")
        
        self.strategies[sid] = strategy
        self.allocations[sid] = StrategyAllocation(
            strategy_id=sid,
            target_weight=0.0,
            current_exposure_mult=0.0,
            paper_mode=True,
        )
        
        logger.info(f"Added {sid} in paper mode for {initial_paper_days} days")
    
    def remove_strategy(self, strategy_id: str):
        if strategy_id not in self.strategies:
            return
        
        attributed_positions = self.state.get_positions_by_strategy(strategy_id)
        for pos in attributed_positions:
            self.oms.submit_intent(OrderIntent(
                strategy_id=f'removal-{strategy_id}',
                symbol=pos.symbol,
                target_position=0,
                urgency='normal',
            ))
        
        del self.strategies[strategy_id]
        del self.allocations[strategy_id]
        
        logger.info(f"Removed strategy {strategy_id}")
        asyncio.create_task(self.rebalance_allocations(force=True))
```

## P&L Attribution

**Location:** `src/portfolio/attribution.py`
**Purpose:** Attribute fills back to contributing strategies so each strategy's P&L is tracked separately.

```python
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd


class PnLAttributor:
    def __init__(self, state_store):
        self.state = state_store
    
    def attribute_fill(self, fill, order_id: str):
        """
        When an order fills, attribute the fill proportionally to 
        contributing strategies based on their recorded contribution.
        """
        contributions = self.state.get_order_contributions(order_id)
        
        for strategy_id, quantity in contributions.items():
            proportion = quantity / sum(abs(q) for q in contributions.values())
            attributed_qty = fill.quantity * proportion
            attributed_cost = fill.commission * proportion
            
            self.state.record_strategy_fill(
                strategy_id=strategy_id,
                symbol=fill.symbol,
                quantity=attributed_qty,
                price=fill.price,
                cost=attributed_cost,
                ts=fill.timestamp,
            )
    
    def compute_strategy_pnl(self, strategy_id: str, as_of: datetime) -> dict:
        """Compute attributed P&L for one strategy up to as_of."""
        fills = self.state.get_strategy_fills(strategy_id, end=as_of)
        
        realized = 0.0
        position_by_symbol = defaultdict(lambda: {'qty': 0, 'cost': 0})
        
        for fill in fills.sort_values('ts'):
            pos = position_by_symbol[fill['symbol']]
            
            if np.sign(fill['quantity']) == np.sign(pos['qty']) or pos['qty'] == 0:
                pos['qty'] += fill['quantity']
                pos['cost'] += fill['quantity'] * fill['price']
            else:
                closing_qty = min(abs(fill['quantity']), abs(pos['qty']))
                avg_entry = pos['cost'] / pos['qty'] if pos['qty'] else fill['price']
                pnl = closing_qty * (fill['price'] - avg_entry) * np.sign(pos['qty'])
                realized += pnl
                
                pos['qty'] += fill['quantity']
                pos['cost'] = pos['qty'] * fill['price'] if pos['qty'] else 0
        
        unrealized = 0.0
        for symbol, pos in position_by_symbol.items():
            if pos['qty'] == 0:
                continue
            current_price = self._get_current_price(symbol)
            avg_entry = pos['cost'] / pos['qty']
            unrealized += pos['qty'] * (current_price - avg_entry)
        
        return {
            'realized': realized,
            'unrealized': unrealized,
            'total': realized + unrealized,
            'open_positions': dict(position_by_symbol),
        }
    
    def _get_current_price(self, symbol: str) -> float:
        # Fetch from broker or cache
        return 1.0
```

## Combined Portfolio Simulation

**Location:** `src/portfolio/simulator.py`
**Purpose:** Backtest combined portfolio with dynamic risk parity allocation.

```python
import pandas as pd
import numpy as np

from src.portfolio.risk_parity import rolling_risk_parity_weights
from src.portfolio.correlation_monitor import CorrelationMonitor


def simulate_combined_portfolio(
    strategy_returns: dict[str, pd.Series],
    target_vol: float = 0.10,
    halflife: int = 60,
    initial_capital: float = 100000,
) -> dict:
    """Simulate portfolio with dynamic risk parity allocation."""
    returns_df = pd.DataFrame(strategy_returns).dropna()
    warmup = 252
    
    weights_df = rolling_risk_parity_weights(
        returns_df, 
        window_days=warmup,
        halflife_days=halflife,
        target_vol=target_vol,
        refit_freq_days=21,
    )
    
    weights_daily = weights_df.reindex(returns_df.index, method='ffill')
    weights_daily = weights_daily.dropna()
    
    corr_monitor = CorrelationMonitor()
    regime_mults = []
    for date in weights_daily.index:
        end = returns_df.index.get_loc(date)
        start = max(0, end - 300)
        window = returns_df.iloc[start:end]
        if len(window) >= 292:
            regime = corr_monitor.check_regime(window)
            regime_mults.append((date, regime['exposure_multiplier']))
        else:
            regime_mults.append((date, 1.0))
    
    regime_series = pd.Series(
        [m for _, m in regime_mults],
        index=[d for d, _ in regime_mults]
    )
    
    aligned_returns = returns_df.loc[weights_daily.index]
    daily_portfolio_returns = (
        aligned_returns * weights_daily * regime_series.values[:, None]
    ).sum(axis=1)
    
    equity = (1 + daily_portfolio_returns).cumprod() * initial_capital
    
    return {
        'returns': daily_portfolio_returns,
        'equity': equity,
        'weights': weights_daily,
        'regime': regime_series,
    }
```

## Add Strategy Workflow

**Location:** `scripts/add_strategy.py`
**Purpose:** Playbook for adding a new strategy to live system without disrupting existing positions.

```python
"""
Workflow for adding a new strategy:
1. Implement strategy class
2. Backtest in isolation (full walk-forward)
3. Check correlation with existing strategies (backtest)
4. Add to system in paper mode
5. Run paper for 30+ days
6. Review paper performance vs. expectations
7. Flip to live with small initial weight (5%)
8. Wait 60 days
9. If performing, include in risk parity rebalance
"""
import logging
logger = logging.getLogger(__name__)


async def add_strategy_workflow(new_strategy, coordinator, 
                                  initial_paper_days: int = 30,
                                  initial_live_weight: float = 0.05):
    coordinator.add_strategy(new_strategy, initial_paper_days=initial_paper_days)
    logger.info(f"Added {new_strategy.id} in paper mode")


async def promote_strategy_to_live(strategy_id: str, coordinator, 
                                     initial_weight: float = 0.05):
    """Flip a paper strategy to live trading."""
    alloc = coordinator.allocations[strategy_id]
    if not alloc.paper_mode:
        logger.warning(f"{strategy_id} is already live")
        return
    
    paper_perf = get_paper_performance(strategy_id)
    if paper_perf['days'] < 30:
        raise ValueError(f"Only {paper_perf['days']} days of paper, need 30+")
    if paper_perf['sharpe'] < 0:
        raise ValueError(f"Paper Sharpe negative ({paper_perf['sharpe']:.2f}), "
                        f"not promoting")
    
    remaining = 1.0 - initial_weight
    for sid, a in coordinator.allocations.items():
        if sid == strategy_id or a.paper_mode:
            continue
        a.target_weight *= remaining
    
    alloc.paper_mode = False
    alloc.target_weight = initial_weight
    alloc.current_exposure_mult = 1.0
    
    logger.info(f"Promoted {strategy_id} to live with weight {initial_weight}")


def get_paper_performance(strategy_id: str) -> dict:
    """Query paper trading performance for evaluation."""
    return {'days': 0, 'sharpe': 0.0}
```
