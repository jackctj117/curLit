# 04 — Backtesting Framework

Walk-forward runner, performance analytics, and event-driven backtest.

## Walk-Forward Runner

**Location:** `src/backtest/walkforward.py`
**Purpose:** Walk-forward backtest framework that prevents look-ahead bias.

```python
from dataclasses import dataclass
from typing import Protocol
import pandas as pd
import numpy as np


class Strategy(Protocol):
    def fit(self, train_data: pd.DataFrame) -> None: ...
    def generate_signals(self, test_data: pd.DataFrame) -> pd.Series: ...


@dataclass
class WalkForwardConfig:
    is_window_days: int = 756       # 3 years
    oos_window_days: int = 63       # ~3 months
    step_days: int = 63             # Non-overlapping
    min_history: int = 756


@dataclass 
class WalkForwardResult:
    oos_signals: pd.Series
    oos_returns: pd.Series
    trades: pd.DataFrame
    fold_metrics: pd.DataFrame
    params_by_fold: list[dict]


class WalkForwardRunner:
    def __init__(self, config: WalkForwardConfig):
        self.config = config
    
    def run(self, data: pd.DataFrame, strategy_factory: callable, 
            cost_model: 'CostModel') -> WalkForwardResult:
        cfg = self.config
        folds = []
        oos_signals_all = []
        trades_all = []
        
        start = cfg.min_history
        while start + cfg.oos_window_days <= len(data):
            train_end = start
            train_start = max(0, train_end - cfg.is_window_days)
            test_end = min(len(data), train_end + cfg.oos_window_days)
            
            train = data.iloc[train_start:train_end]
            test = data.iloc[train_end:test_end]
            
            strategy = strategy_factory()
            strategy.fit(train)
            
            signals = strategy.generate_signals(test)
            oos_signals_all.append(signals)
            
            trades = self._signals_to_trades(signals, test, cost_model)
            trades_all.append(trades)
            
            folds.append({
                'fold_id': len(folds),
                'train_start': train.index[0],
                'train_end': train.index[-1],
                'test_start': test.index[0],
                'test_end': test.index[-1],
                'params': getattr(strategy, 'params', {}),
                'train_sharpe': self._sharpe(train.get('returns', pd.Series())),
                'test_sharpe': self._sharpe(trades['net_return']),
            })
            
            start += cfg.step_days
        
        return WalkForwardResult(
            oos_signals=pd.concat(oos_signals_all),
            oos_returns=pd.concat([t['net_return'] for t in trades_all]),
            trades=pd.concat(trades_all),
            fold_metrics=pd.DataFrame(folds),
            params_by_fold=[f['params'] for f in folds],
        )
    
    def _signals_to_trades(self, signals: pd.Series, data: pd.DataFrame, 
                            cost_model: 'CostModel') -> pd.DataFrame:
        df = pd.DataFrame(index=signals.index)
        df['signal'] = signals
        df['position'] = signals.shift(1).fillna(0)  # Trade at next bar
        df['price'] = data['close']
        df['return'] = df['price'].pct_change()
        df['strategy_return'] = df['position'] * df['return']
        df['position_change'] = df['position'].diff().abs().fillna(0)
        df['cost'] = df['position_change'] * cost_model.cost_per_turn
        df['net_return'] = df['strategy_return'] - df['cost']
        return df
    
    @staticmethod
    def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(periods_per_year)


@dataclass
class CostModel:
    spread_bps: float = 0.5
    slippage_bps: float = 0.3
    
    @property
    def cost_per_turn(self) -> float:
        return (self.spread_bps + self.slippage_bps) / 10000
```

## Performance Analytics

**Location:** `src/backtest/analytics.py`
**Purpose:** Compute return, risk, and trade metrics. Regime-aware analysis.

```python
import pandas as pd
import numpy as np


class PerformanceAnalytics:
    @staticmethod
    def metrics(returns: pd.Series) -> dict:
        if len(returns) == 0:
            return {}
        
        ann_factor = 252
        total_return = (1 + returns).prod() - 1
        years = len(returns) / ann_factor
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        vol = returns.std() * np.sqrt(ann_factor)
        sharpe = returns.mean() / returns.std() * np.sqrt(ann_factor) \
                 if returns.std() > 0 else 0
        
        downside = returns[returns < 0]
        sortino = returns.mean() / downside.std() * np.sqrt(ann_factor) \
                  if len(downside) > 0 and downside.std() > 0 else 0
        
        equity = (1 + returns).cumprod()
        drawdown = (equity - equity.cummax()) / equity.cummax()
        max_dd = drawdown.min()
        
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0
        
        trades = returns[returns != 0]
        hit_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0
        avg_win = trades[trades > 0].mean() if (trades > 0).any() else 0
        avg_loss = trades[trades < 0].mean() if (trades < 0).any() else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'calmar': calmar,
            'hit_rate': hit_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': (trades[trades > 0].sum() / 
                            abs(trades[trades < 0].sum()))
                           if (trades < 0).any() else float('inf'),
        }
    
    @staticmethod
    def regime_metrics(returns: pd.Series, regime: pd.Series) -> pd.DataFrame:
        """Performance by regime."""
        df = pd.DataFrame({'returns': returns, 'regime': regime})
        grouped = df.groupby('regime')['returns']
        return pd.DataFrame({
            'mean_ret': grouped.mean() * 252,
            'sharpe': grouped.apply(lambda x: x.mean() / x.std() * np.sqrt(252)
                                    if x.std() > 0 else 0),
            'count': grouped.count(),
        })
    
    @staticmethod
    def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
        return (returns.rolling(window).mean() / 
                returns.rolling(window).std() * np.sqrt(252))
```

## Bootstrap Confidence Intervals

**Location:** `src/backtest/bootstrap.py`
**Purpose:** Honest confidence intervals for Sharpe ratios, accounting for serial correlation.

```python
import numpy as np
import pandas as pd


def bootstrap_sharpe_ci(returns: pd.Series, n_bootstrap: int = 10000, 
                        confidence: float = 0.95) -> tuple[float, float]:
    """Bootstrap CI for Sharpe ratio."""
    sharpes = []
    n = len(returns)
    for _ in range(n_bootstrap):
        sample = returns.sample(n, replace=True)
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std() * np.sqrt(252))
    alpha = (1 - confidence) / 2
    return (np.quantile(sharpes, alpha), np.quantile(sharpes, 1 - alpha))


def stationary_bootstrap(returns: pd.Series, block_mean_len: int = 20, 
                         n_bootstrap: int = 10000) -> np.ndarray:
    """Politis-Romano stationary bootstrap — preserves serial correlation."""
    n = len(returns)
    p = 1.0 / block_mean_len
    results = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        indices = []
        i = np.random.randint(n)
        while len(indices) < n:
            indices.append(i)
            if np.random.random() < p:
                i = np.random.randint(n)
            else:
                i = (i + 1) % n
        sample = returns.iloc[indices[:n]]
        if sample.std() > 0:
            results[b] = sample.mean() / sample.std() * np.sqrt(252)
    
    return results
```

## Event-Driven Backtest

**Location:** `src/backtest/event_backtest.py`
**Purpose:** Backtest for event-driven strategies (CB sentiment). Each event is a discrete trade.

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class EventBacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict


class EventBacktester:
    def __init__(self, cost_model):
        self.cost_model = cost_model
    
    def run(self, events: pd.DataFrame, prices: pd.DataFrame,
            config, starting_equity: float = 100000) -> EventBacktestResult:
        """
        events: DataFrame with columns ['ts', 'cb', 'net_shift', 'doc_id']
        prices: DataFrame with FX pair columns, indexed by timestamp
        """
        trades = []
        equity = starting_equity
        equity_over_time = []
        
        for idx, event in events.iterrows():
            # Threshold from events BEFORE this one (walk-forward)
            past_events = events[events['ts'] < event['ts']]
            cb_past = past_events[past_events['cb'] == event['cb']]
            
            if len(cb_past) < 20:
                continue
            
            dovish_thr = np.percentile(cb_past['net_shift'], 
                                         config.strong_shift_percentile * 100)
            hawkish_thr = np.percentile(cb_past['net_shift'],
                                          (1 - config.strong_shift_percentile) * 100)
            
            shift = event['net_shift']
            if shift < dovish_thr:
                signal_type = 'dovish'
            elif shift > hawkish_thr:
                signal_type = 'hawkish'
            else:
                continue
            
            cb = event['cb']
            if cb not in config.cb_to_pair:
                continue
            pair, hawkish_side = config.cb_to_pair[cb]
            
            if signal_type == 'hawkish':
                direction = 1 if hawkish_side == 'long' else -1
            else:
                direction = -1 if hawkish_side == 'long' else 1
            
            entry_ts = event['ts']
            if entry_ts not in prices.index:
                entry_ts = prices.index[prices.index > entry_ts][0]
            entry_price = prices.loc[entry_ts, pair]
            
            exit_window = prices.loc[entry_ts:].head(config.holding_days + 1)
            if len(exit_window) < 2:
                continue
            
            stop_loss = entry_price * (1 - direction * config.hard_stop_pct)
            trailing_stop = None
            peak = entry_price
            exit_price = None
            exit_reason = 'time_exit'
            
            for ts in exit_window.index[1:]:
                current = prices.loc[ts, pair]
                pnl_pct = (current - entry_price) / entry_price * direction
                
                if pnl_pct >= config.trailing_trigger_pct:
                    if direction > 0:
                        peak = max(peak, current)
                        trailing_stop = peak * (1 - config.trailing_distance_pct)
                    else:
                        peak = min(peak, current)
                        trailing_stop = peak * (1 + config.trailing_distance_pct)
                
                if direction > 0:
                    if current <= stop_loss:
                        exit_price, exit_reason = stop_loss, 'hard_stop'
                        break
                    if trailing_stop and current <= trailing_stop:
                        exit_price, exit_reason = trailing_stop, 'trailing_stop'
                        break
                else:
                    if current >= stop_loss:
                        exit_price, exit_reason = stop_loss, 'hard_stop'
                        break
                    if trailing_stop and current >= trailing_stop:
                        exit_price, exit_reason = trailing_stop, 'trailing_stop'
                        break
            
            if exit_price is None:
                exit_price = exit_window[pair].iloc[-1]
            
            gross_pnl_pct = (exit_price - entry_price) / entry_price * direction
            costs_pct = 2 * self.cost_model.cost_per_turn
            net_pnl_pct = gross_pnl_pct - costs_pct
            
            risk_amount = equity * config.risk_per_trade_pct
            stop_distance = abs(entry_price - stop_loss)
            size_notional = risk_amount * entry_price / stop_distance
            pnl_dollars = size_notional * net_pnl_pct
            equity += pnl_dollars
            
            trades.append({
                'entry_ts': entry_ts,
                'cb': cb,
                'pair': pair,
                'direction': direction,
                'shift_score': shift,
                'signal_type': signal_type,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'gross_pnl_pct': gross_pnl_pct,
                'net_pnl_pct': net_pnl_pct,
                'pnl_dollars': pnl_dollars,
                'equity_after': equity,
            })
            equity_over_time.append((entry_ts, equity))
        
        trades_df = pd.DataFrame(trades)
        equity_curve = pd.Series(
            [e[1] for e in equity_over_time],
            index=[e[0] for e in equity_over_time]
        )
        
        metrics = self._compute_metrics(trades_df, starting_equity, equity)
        return EventBacktestResult(trades_df, equity_curve, metrics)
    
    def _compute_metrics(self, trades, start_eq, end_eq):
        if len(trades) == 0:
            return {}
        
        years = (trades['entry_ts'].max() - trades['entry_ts'].min()).days / 365
        total_return = (end_eq / start_eq) - 1
        cagr = (end_eq / start_eq) ** (1/years) - 1 if years > 0 else 0
        
        hit_rate = (trades['net_pnl_pct'] > 0).mean()
        avg_win = trades[trades['net_pnl_pct'] > 0]['net_pnl_pct'].mean()
        avg_loss = trades[trades['net_pnl_pct'] < 0]['net_pnl_pct'].mean()
        
        return {
            'n_trades': len(trades),
            'total_return': total_return,
            'cagr': cagr,
            'hit_rate': hit_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': (trades[trades['net_pnl_pct'] > 0]['net_pnl_pct'].sum() /
                            abs(trades[trades['net_pnl_pct'] < 0]['net_pnl_pct'].sum()))
                           if (trades['net_pnl_pct'] < 0).any() else float('inf'),
            'trades_per_year': len(trades) / years if years > 0 else 0,
        }
```

## Swap/Funding Cost Model

**Location:** `src/backtest/swap_model.py`
**Purpose:** Realistic modeling of overnight funding costs in backtests.

```python
from dataclasses import dataclass
from datetime import date, timedelta
import pandas as pd
import numpy as np


@dataclass
class SwapModelConfig:
    broker_markup_pct: float = 0.50     # % markup broker charges on swap
    weekend_roll_weekday: int = 2       # Wednesday (Mon=0)


class SwapModel:
    def __init__(self, config: SwapModelConfig):
        self.config = config
    
    def compute_daily_swap(self, pair: str, position: float,
                            target_rate: float, base_rate: float,
                            day_of_week: int) -> float:
        """
        pair: e.g., 'EURUSD'
        position: signed position size (positive = long base currency)
        target_rate: rate on the currency being held
        base_rate: rate on the currency being shorted
        day_of_week: 0=Mon ... 6=Sun
        """
        if day_of_week == self.config.weekend_roll_weekday:
            multiplier = 3  # Triple-swap Wednesday
        elif day_of_week in (5, 6):
            return 0  # No swap on weekend (captured in triple)
        else:
            multiplier = 1
        
        rate_diff = target_rate - base_rate
        daily_rate_diff = rate_diff / 360
        
        sign = np.sign(position * rate_diff)
        markup_penalty = self.config.broker_markup_pct / 100 / 360
        
        if position > 0:
            effective = daily_rate_diff - markup_penalty
        else:
            effective = -daily_rate_diff - markup_penalty
        
        swap_amount = abs(position) * effective * multiplier
        return -swap_amount if sign < 0 else swap_amount


def simulate_position_with_swap(
    entry_date: date, exit_date: date, 
    entry_price: float, exit_price: float,
    position_size: float,
    pair: str,
    rate_data: pd.DataFrame,
    swap_model: SwapModel,
    cost_model,
) -> dict:
    price_pnl = (exit_price - entry_price) * position_size
    
    total_notional = abs(position_size) * entry_price
    transaction_cost = total_notional * cost_model.cost_per_turn * 2
    
    total_swap = 0
    current = entry_date
    while current < exit_date:
        weekday = current.weekday()
        if weekday < 5 or weekday == swap_model.config.weekend_roll_weekday:
            day_rates = rate_data.loc[:pd.Timestamp(current)].iloc[-1]
            target_rate = day_rates['base_rate'] if position_size > 0 else day_rates['quote_rate']
            base_rate = day_rates['quote_rate'] if position_size > 0 else day_rates['base_rate']
            
            swap = swap_model.compute_daily_swap(
                pair=pair,
                position=position_size,
                target_rate=target_rate,
                base_rate=base_rate,
                day_of_week=weekday,
            )
            total_swap += swap
        current += timedelta(days=1)
    
    net_pnl = price_pnl - transaction_cost + total_swap
    
    return {
        'price_pnl': price_pnl,
        'transaction_cost': -transaction_cost,
        'swap_pnl': total_swap,
        'net_pnl': net_pnl,
        'days_held': (exit_date - entry_date).days,
    }
```

## Historical Replay Harness

**Location:** `src/testing/replay_harness.py`
**Purpose:** Replay historical data through the live system at accelerated speed. Validates live stack matches backtest.

```python
from datetime import date


class HistoricalReplay:
    """
    Replay historical data through the live system at accelerated speed.
    Tests the full stack: ingestion → features → signals → portfolio → OMS → paper broker.
    """
    
    def run(self, start_date: date, end_date: date, speed_multiplier: int = 100):
        """
        - Set up clock mock
        - Feed historical prices tick by tick
        - Feed macro data at historical release times
        - Feed CB statements at historical publication times
        - Generate signals, submit to paper broker
        - Record full P&L attribution
        - At end: compare replay P&L to offline backtest P&L
        """
        pass  # Implementation wires together clock mock, data feed, live engine
```

## Stress Test Scenarios

**Location:** `src/backtest/stress_test.py`
**Purpose:** Replay strategies through historical crisis periods.

```python
from datetime import date
import pandas as pd


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


def stress_test_portfolio(portfolio_returns: pd.Series):
    """Print stress test results for all historical crisis periods."""
    for name, (start, end) in SCENARIOS.items():
        period = portfolio_returns.loc[start:end]
        if len(period) == 0:
            continue
        total_return = (1 + period).prod() - 1
        max_dd = min((1 + period).cumprod() / (1 + period).cumprod().cummax() - 1)
        worst_day = period.min()
        print(f"{name:30} Return: {total_return:+7.2%}  "
              f"MaxDD: {max_dd:+7.2%}  "
              f"Worst Day: {worst_day:+6.2%}")
```

## Walk-Forward Tests

**Location:** `tests/unit/test_walkforward.py`
**Purpose:** Verify the backtest framework doesn't leak look-ahead information.

```python
import pytest
import numpy as np
import pandas as pd

from src.backtest.walkforward import WalkForwardRunner, WalkForwardConfig, CostModel


def test_walkforward_no_lookahead():
    """Verify the walk-forward runner never uses future data."""
    data = make_fake_data(n=2000)
    
    class SpyStrategy:
        def __init__(self):
            self.train_max_dates = []
            self.test_min_dates = []
        def fit(self, train):
            self.train_max_dates.append(train.index.max())
        def generate_signals(self, test):
            self.test_min_dates.append(test.index.min())
            return pd.Series(0, index=test.index)
    
    strat = SpyStrategy()
    runner = WalkForwardRunner(WalkForwardConfig())
    runner.run(data, lambda: strat, CostModel())
    
    for train_end, test_start in zip(strat.train_max_dates, 
                                      strat.test_min_dates):
        assert train_end < test_start, "Look-ahead leakage!"


def test_rate_diff_fits_known_relationship():
    """If we construct data with a known β, model should recover it."""
    from src.models.rate_diff import RateDiffModel
    
    np.random.seed(42)
    n = 1000
    spread = np.random.randn(n).cumsum() * 0.01 + 1.5
    price = 1.10 - 0.05 * spread + np.random.randn(n) * 0.005
    
    df = pd.DataFrame({'target': price, 'spread': spread})
    model = RateDiffModel('EURUSD', ['spread'])
    result = model.fit(df)
    
    assert abs(result.coefficients['spread'] - (-0.05)) < 0.01
    assert result.r_squared > 0.8


def make_fake_data(n: int) -> pd.DataFrame:
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    return pd.DataFrame({
        'close': 100 + np.random.randn(n).cumsum(),
        'returns': np.random.randn(n) * 0.01,
    }, index=dates)
```

## Property-Based Tests

**Location:** `tests/unit/test_properties.py`
**Purpose:** Hypothesis-driven tests for numerical code correctness.

```python
from hypothesis import given, strategies as st
from src.portfolio.sizing import PositionSizer


@given(st.floats(min_value=-5, max_value=5), 
       st.floats(min_value=0.1, max_value=10))
def test_kelly_never_negative(edge, odds):
    result = PositionSizer.kelly(edge, odds)
    assert result >= 0
    assert result <= 1
```
