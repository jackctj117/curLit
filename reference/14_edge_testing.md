# 14 — Edge Testing

Framework for rigorously testing whether your strategies actually have edge, or whether the backtest is lying to you.

## Null Hypothesis Framework

**Location:** `src/edge_testing/null_hypothesis.py`
**Purpose:** Compare strategy performance against random-signal baselines. If your Sharpe isn't distinguishable from randomness, you don't have edge.

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable


@dataclass
class NullTestResult:
    null_name: str
    strategy_sharpe: float
    null_mean_sharpe: float
    null_std_sharpe: float
    null_95th_percentile: float
    p_value: float
    edge_exists: bool
    n_simulations: int


class NullHypothesisFramework:
    """
    For any strategy, compare its performance against multiple null hypotheses.
    """
    
    def __init__(self, periods_per_year: int = 252):
        self.periods_per_year = periods_per_year
    
    def test_strategy(self, strategy_returns: pd.Series, 
                       price_data: pd.DataFrame,
                       n_simulations: int = 10000) -> dict[str, NullTestResult]:
        results = {}
        strategy_sharpe = self._sharpe(strategy_returns)
        
        # Null 1: Random signal with matched turnover
        results['random_same_turnover'] = self._test_random_turnover(
            strategy_returns, price_data, strategy_sharpe, n_simulations
        )
        
        # Null 2: Random signal with matched autocorrelation
        results['random_autocorr_preserved'] = self._test_random_autocorr(
            strategy_returns, price_data, strategy_sharpe, n_simulations
        )
        
        # Null 3: Buy and hold
        results['buy_and_hold'] = self._test_buy_and_hold(
            strategy_returns, price_data, strategy_sharpe
        )
        
        # Null 4: Simple momentum (12-1 month)
        results['simple_momentum'] = self._test_simple_momentum(
            strategy_returns, price_data, strategy_sharpe
        )
        
        # Null 5: Simple carry (highest yielder long, lowest short)
        if self._has_rate_data(price_data):
            results['simple_carry'] = self._test_simple_carry(
                strategy_returns, price_data, strategy_sharpe
            )
        
        return results
    
    def _test_random_turnover(self, strategy_returns, price_data, 
                               strategy_sharpe, n_sim):
        turnover = self._compute_turnover(strategy_returns)
        null_sharpes = []
        
        for _ in range(n_sim):
            random_signal = self._generate_random_signal(
                length=len(strategy_returns), turnover=turnover
            )
            random_returns = self._apply_signal(random_signal, price_data)
            null_sharpes.append(self._sharpe(random_returns))
        
        null_sharpes = np.array(null_sharpes)
        p_value = (null_sharpes >= strategy_sharpe).mean()
        
        return NullTestResult(
            null_name='random_same_turnover',
            strategy_sharpe=strategy_sharpe,
            null_mean_sharpe=null_sharpes.mean(),
            null_std_sharpe=null_sharpes.std(),
            null_95th_percentile=np.percentile(null_sharpes, 95),
            p_value=p_value,
            edge_exists=p_value < 0.05,
            n_simulations=n_sim,
        )
    
    def _test_random_autocorr(self, strategy_returns, price_data,
                                strategy_sharpe, n_sim):
        """Generate random signals with similar serial correlation structure."""
        signal = self._infer_signal(strategy_returns, price_data)
        ac1 = signal.autocorr(lag=1) if len(signal) > 1 else 0
        
        null_sharpes = []
        for _ in range(n_sim):
            random_signal = self._generate_ar1_signal(
                length=len(strategy_returns), autocorr=ac1
            )
            random_returns = self._apply_signal(random_signal, price_data)
            null_sharpes.append(self._sharpe(random_returns))
        
        null_sharpes = np.array(null_sharpes)
        p_value = (null_sharpes >= strategy_sharpe).mean()
        
        return NullTestResult(
            null_name='random_autocorr_preserved',
            strategy_sharpe=strategy_sharpe,
            null_mean_sharpe=null_sharpes.mean(),
            null_std_sharpe=null_sharpes.std(),
            null_95th_percentile=np.percentile(null_sharpes, 95),
            p_value=p_value,
            edge_exists=p_value < 0.05,
            n_simulations=n_sim,
        )
    
    def _test_buy_and_hold(self, strategy_returns, price_data, strategy_sharpe):
        bh_returns = price_data.iloc[:, 0].pct_change().dropna()
        bh_sharpe = self._sharpe(bh_returns)
        
        return NullTestResult(
            null_name='buy_and_hold',
            strategy_sharpe=strategy_sharpe,
            null_mean_sharpe=bh_sharpe,
            null_std_sharpe=0,
            null_95th_percentile=bh_sharpe,
            p_value=0.5 if strategy_sharpe > bh_sharpe else 1.0,
            edge_exists=strategy_sharpe > bh_sharpe * 1.2,
            n_simulations=1,
        )
    
    def _test_simple_momentum(self, strategy_returns, price_data, strategy_sharpe):
        momentum_signal = price_data.iloc[:, 0].pct_change(252 - 21).shift(21)
        momentum_signal = np.sign(momentum_signal)
        momentum_returns = momentum_signal * price_data.iloc[:, 0].pct_change()
        momentum_returns = momentum_returns.dropna()
        momentum_sharpe = self._sharpe(momentum_returns)
        
        return NullTestResult(
            null_name='simple_momentum',
            strategy_sharpe=strategy_sharpe,
            null_mean_sharpe=momentum_sharpe,
            null_std_sharpe=0,
            null_95th_percentile=momentum_sharpe,
            p_value=0.5 if strategy_sharpe > momentum_sharpe else 1.0,
            edge_exists=strategy_sharpe > momentum_sharpe * 1.2,
            n_simulations=1,
        )
    
    def _test_simple_carry(self, strategy_returns, price_data, strategy_sharpe):
        # Simplified: assumes rate columns present
        return NullTestResult(
            null_name='simple_carry',
            strategy_sharpe=strategy_sharpe,
            null_mean_sharpe=0.5,
            null_std_sharpe=0,
            null_95th_percentile=0.5,
            p_value=0.5 if strategy_sharpe > 0.5 else 1.0,
            edge_exists=strategy_sharpe > 0.6,
            n_simulations=1,
        )
    
    def _sharpe(self, returns: pd.Series) -> float:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(self.periods_per_year)
    
    def _compute_turnover(self, returns: pd.Series) -> float:
        return (returns != 0).mean()
    
    def _generate_random_signal(self, length: int, turnover: float) -> np.ndarray:
        raw = np.random.choice([-1, 0, 1], size=length, 
                               p=[turnover/2, 1-turnover, turnover/2])
        return raw
    
    def _generate_ar1_signal(self, length: int, autocorr: float) -> np.ndarray:
        signal = np.zeros(length)
        signal[0] = np.random.choice([-1, 0, 1])
        for i in range(1, length):
            if np.random.random() < abs(autocorr):
                signal[i] = signal[i-1] * np.sign(autocorr)
            else:
                signal[i] = np.random.choice([-1, 0, 1])
        return signal
    
    def _apply_signal(self, signal: np.ndarray, 
                       price_data: pd.DataFrame) -> pd.Series:
        prices = price_data.iloc[:, 0]
        returns = prices.pct_change().fillna(0).values
        lagged_signal = np.roll(signal, 1)
        lagged_signal[0] = 0
        strategy_returns = lagged_signal * returns
        return pd.Series(strategy_returns, index=price_data.index)
    
    def _infer_signal(self, strategy_returns: pd.Series, 
                        price_data: pd.DataFrame) -> pd.Series:
        prices = price_data.iloc[:, 0]
        market_returns = prices.pct_change()
        signal = np.sign(strategy_returns / market_returns.replace(0, np.nan))
        return signal.fillna(0)
    
    def _has_rate_data(self, price_data: pd.DataFrame) -> bool:
        return any('rate' in col.lower() for col in price_data.columns)
```

## Multiple Testing Correction

**Location:** `src/edge_testing/multiple_testing.py`
**Purpose:** White's Reality Check and Bonferroni/BH corrections. Prevents data snooping bias when you've tested many strategies.

```python
import numpy as np
import pandas as pd


class MultipleTestingCorrection:
    """
    Apply corrections to prevent finding false "edge" from testing 
    many strategies or parameter combinations.
    """
    
    def white_reality_check(self, strategy_returns_dict: dict[str, pd.Series],
                             benchmark_returns: pd.Series,
                             n_bootstrap: int = 10000,
                             block_mean_length: int = 20) -> dict:
        """
        White's Reality Check: tests whether the BEST strategy from a set
        has statistically significant outperformance given that you tested many.
        
        Use when you've tried multiple variations and are picking the best.
        """
        strategy_names = list(strategy_returns_dict.keys())
        n_strategies = len(strategy_names)
        
        # Excess returns over benchmark
        excess_returns = {}
        for name, returns in strategy_returns_dict.items():
            aligned = returns.align(benchmark_returns, join='inner')
            excess_returns[name] = aligned[0] - aligned[1]
        
        # Observed best strategy
        mean_excess = {n: r.mean() for n, r in excess_returns.items()}
        best_name = max(mean_excess.keys(), key=lambda k: mean_excess[k])
        best_excess = mean_excess[best_name]
        
        # Stationary bootstrap under null (no edge)
        null_max_stats = []
        for _ in range(n_bootstrap):
            resampled = self._stationary_bootstrap_multi(
                excess_returns, block_mean_length
            )
            # Center at zero (null: no edge)
            centered_means = [
                (ret - ret.mean()).mean() for ret in resampled.values()
            ]
            null_max_stats.append(max(centered_means))
        
        null_max_stats = np.array(null_max_stats)
        p_value = (null_max_stats >= best_excess).mean()
        
        return {
            'best_strategy': best_name,
            'best_excess_return': best_excess,
            'best_excess_annualized': best_excess * 252,
            'p_value_corrected': p_value,
            'n_strategies_tested': n_strategies,
            'has_edge_after_correction': p_value < 0.05,
            'null_95th_percentile': np.percentile(null_max_stats, 95),
        }
    
    def bonferroni_correction(self, p_values: dict[str, float], 
                                alpha: float = 0.05) -> dict:
        """Conservative correction: multiply each p-value by n_tests."""
        n = len(p_values)
        corrected_alpha = alpha / n
        
        return {
            name: {
                'raw_p_value': p,
                'corrected_alpha': corrected_alpha,
                'significant': p < corrected_alpha,
            }
            for name, p in p_values.items()
        }
    
    def benjamini_hochberg(self, p_values: dict[str, float], 
                             fdr: float = 0.10) -> dict:
        """Controls false discovery rate. Less conservative than Bonferroni."""
        n = len(p_values)
        sorted_items = sorted(p_values.items(), key=lambda x: x[1])
        
        results = {}
        max_significant_rank = -1
        
        for i, (name, p) in enumerate(sorted_items, start=1):
            threshold = (i / n) * fdr
            if p <= threshold:
                max_significant_rank = i
        
        for i, (name, p) in enumerate(sorted_items, start=1):
            results[name] = {
                'raw_p_value': p,
                'rank': i,
                'bh_threshold': (i / n) * fdr,
                'significant': i <= max_significant_rank,
            }
        
        return results
    
    def deflated_sharpe_ratio(self, observed_sharpe: float, 
                                n_trials: int,
                                n_observations: int,
                                skew: float = 0,
                                kurtosis: float = 3) -> float:
        """
        Bailey & Lopez de Prado's Deflated Sharpe Ratio.
        Adjusts observed Sharpe downward for the fact that you tried many variations.
        """
        from scipy.stats import norm
        
        # Expected max Sharpe under null when testing n_trials strategies
        e_max = np.sqrt(2 * np.log(n_trials))
        euler_mascheroni = 0.5772156649
        expected_max_sharpe = (
            (1 - euler_mascheroni) * norm.ppf(1 - 1/n_trials) +
            euler_mascheroni * norm.ppf(1 - 1/(n_trials * np.e))
        )
        
        # Adjustment for non-normality
        denominator = np.sqrt(
            (1 - skew * observed_sharpe + 
             (kurtosis - 1) / 4 * observed_sharpe**2) / (n_observations - 1)
        )
        
        z = (observed_sharpe - expected_max_sharpe) / denominator
        deflated_prob = norm.cdf(z)
        
        return {
            'observed_sharpe': observed_sharpe,
            'expected_max_sharpe_under_null': expected_max_sharpe,
            'deflation_adjustment': observed_sharpe - expected_max_sharpe,
            'probability_real_edge': deflated_prob,
            'has_edge': deflated_prob > 0.95,
        }
    
    def _stationary_bootstrap_multi(self, returns_dict: dict, 
                                       block_mean_length: int) -> dict:
        """Bootstrap all strategies together preserving alignment."""
        n = len(next(iter(returns_dict.values())))
        p = 1.0 / block_mean_length
        
        indices = []
        i = np.random.randint(n)
        while len(indices) < n:
            indices.append(i)
            if np.random.random() < p:
                i = np.random.randint(n)
            else:
                i = (i + 1) % n
        
        return {
            name: returns.iloc[indices[:n]].reset_index(drop=True)
            for name, returns in returns_dict.items()
        }
```

## Live Performance Tracker

**Location:** `src/edge_testing/live_tracker.py`
**Purpose:** Continuously test whether live performance matches backtest expectations. Alerts when they diverge.

```python
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats


class LiveEdgeTracker:
    """
    Monitors live returns vs. backtest expectations.
    Tests divergence with proper statistical power.
    """
    
    def __init__(self, strategy_id: str, backtest_metrics: dict):
        self.strategy_id = strategy_id
        self.expected_sharpe = backtest_metrics['sharpe']
        self.expected_hit_rate = backtest_metrics['hit_rate']
        self.expected_mean_return = backtest_metrics['mean_return']
        self.expected_vol = backtest_metrics['vol']
        self.expected_max_dd = backtest_metrics['max_drawdown']
    
    def check_divergence(self, live_returns: pd.Series, 
                           min_sample_days: int = 60) -> dict:
        if len(live_returns) < min_sample_days:
            return {
                'status': 'insufficient_data',
                'days_live': len(live_returns),
                'days_required': min_sample_days,
            }
        
        live_sharpe = self._sharpe(live_returns)
        live_hit_rate = (live_returns > 0).mean()
        live_mean = live_returns.mean()
        live_vol = live_returns.std() * np.sqrt(252)
        live_dd = self._max_drawdown(live_returns)
        
        # Test: is live Sharpe statistically different from expected?
        n = len(live_returns)
        sharpe_se = np.sqrt((1 + 0.5 * self.expected_sharpe**2) / n)
        sharpe_z = (live_sharpe - self.expected_sharpe) / sharpe_se
        sharpe_p_value = 2 * (1 - stats.norm.cdf(abs(sharpe_z)))
        
        # Hit rate test (binomial)
        n_trades = (live_returns != 0).sum()
        observed_wins = (live_returns > 0).sum()
        hit_rate_p = stats.binomtest(observed_wins, n_trades, 
                                       self.expected_hit_rate).pvalue \
                     if n_trades > 0 else 1.0
        
        # Mean return test (t-test)
        if live_returns.std() > 0:
            t_stat = (live_mean - self.expected_mean_return) / (
                live_returns.std() / np.sqrt(n)
            )
            mean_p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        else:
            mean_p_value = 1.0
        
        severity = self._assess_severity(sharpe_z, live_sharpe, live_dd)
        
        return {
            'days_live': n,
            'live_sharpe': live_sharpe,
            'expected_sharpe': self.expected_sharpe,
            'sharpe_z_score': sharpe_z,
            'sharpe_p_value': sharpe_p_value,
            'live_hit_rate': live_hit_rate,
            'expected_hit_rate': self.expected_hit_rate,
            'hit_rate_p_value': hit_rate_p,
            'live_mean_return': live_mean,
            'mean_return_p_value': mean_p_value,
            'live_max_dd': live_dd,
            'expected_max_dd': self.expected_max_dd,
            'dd_worse_than_expected': live_dd < self.expected_max_dd * 1.3,
            'severity': severity,
            'recommendation': self._recommendation(severity, n),
        }
    
    def _assess_severity(self, z_score: float, live_sharpe: float, 
                          live_dd: float) -> str:
        if live_dd < self.expected_max_dd * 1.5:
            return 'drawdown_exceeded'
        if z_score > -1.0:
            return 'on_track'
        elif z_score > -2.0:
            return 'underperforming'
        elif z_score > -3.0:
            return 'significantly_underperforming'
        else:
            return 'severely_underperforming'
    
    def _recommendation(self, severity: str, days_live: int) -> str:
        if severity == 'on_track':
            return 'continue'
        if severity == 'drawdown_exceeded':
            return 'halt_investigate_drawdown'
        if days_live < 90:
            return 'continue_monitoring_insufficient_data'
        if severity == 'underperforming':
            return 'review_at_next_checkpoint'
        if severity == 'significantly_underperforming':
            return 'reduce_size_50pct'
        if severity == 'severely_underperforming':
            return 'halt_and_investigate'
        return 'continue'
    
    def _sharpe(self, returns: pd.Series) -> float:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        equity = (1 + returns).cumprod()
        return ((equity - equity.cummax()) / equity.cummax()).min()
```

## Paper vs Live Divergence

**Location:** `src/edge_testing/paper_live_divergence.py`
**Purpose:** Compare paper and live execution on the same strategy. Divergence indicates execution problems.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class Fill:
    ts: datetime
    strategy_id: str
    symbol: str
    side: str
    quantity: float
    price: float


class PaperLiveDivergence:
    """
    Runs paper shadow of every live strategy. 
    Divergence between them indicates broker execution issues.
    """
    
    def __init__(self, state_store):
        self.state = state_store
    
    def compare(self, strategy_id: str, 
                 lookback_days: int = 30) -> dict:
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        
        paper = self.state.get_fills(
            strategy_id=strategy_id, mode='paper', since=cutoff
        )
        live = self.state.get_fills(
            strategy_id=strategy_id, mode='live', since=cutoff
        )
        
        matched = self._match_fills(paper, live)
        unmatched_paper = [p for p in paper if not any(
            self._fills_match(p, l) for _, l in matched
        )]
        unmatched_live = [l for l in live if not any(
            self._fills_match(p, l) for p, _ in matched
        )]
        
        price_diffs_bps = []
        latency_diffs_ms = []
        
        for paper_fill, live_fill in matched:
            price_diff_bps = ((live_fill.price - paper_fill.price) 
                              / paper_fill.price * 10000)
            if paper_fill.side != live_fill.side:
                continue
            if live_fill.side == 'buy':
                price_diffs_bps.append(price_diff_bps)
            else:
                price_diffs_bps.append(-price_diff_bps)
            
            latency = (live_fill.ts - paper_fill.ts).total_seconds() * 1000
            latency_diffs_ms.append(latency)
        
        # Compute strategy P&L divergence
        paper_pnl = self._compute_pnl(paper)
        live_pnl = self._compute_pnl(live)
        pnl_ratio = live_pnl / paper_pnl if paper_pnl != 0 else 0
        
        results = {
            'strategy_id': strategy_id,
            'n_paper_fills': len(paper),
            'n_live_fills': len(live),
            'n_matched': len(matched),
            'n_unmatched_paper': len(unmatched_paper),
            'n_unmatched_live': len(unmatched_live),
            'avg_price_diff_bps': np.mean(price_diffs_bps) if price_diffs_bps else 0,
            'p95_price_diff_bps': np.percentile(price_diffs_bps, 95) if price_diffs_bps else 0,
            'avg_latency_ms': np.mean(latency_diffs_ms) if latency_diffs_ms else 0,
            'paper_pnl': paper_pnl,
            'live_pnl': live_pnl,
            'pnl_ratio': pnl_ratio,
            'concerns': [],
        }
        
        # Flag concerning divergences
        if abs(results['avg_price_diff_bps']) > 2:
            results['concerns'].append(
                f"Average {results['avg_price_diff_bps']:.1f} bps slippage "
                f"vs paper — broker execution worse than expected"
            )
        if results['p95_price_diff_bps'] > 5:
            results['concerns'].append(
                f"95th percentile slippage {results['p95_price_diff_bps']:.1f} bps "
                f"indicates tail execution risk"
            )
        if results['n_unmatched_live'] > len(matched) * 0.1:
            results['concerns'].append(
                f"{results['n_unmatched_live']} live fills don't match paper — "
                f"possible strategy logic drift or timing issues"
            )
        if pnl_ratio < 0.7 and paper_pnl > 0:
            results['concerns'].append(
                f"Live P&L only {pnl_ratio:.0%} of paper P&L — "
                f"edge eroded by execution costs"
            )
        
        return results
    
    def _match_fills(self, paper: list[Fill], live: list[Fill], 
                       max_time_diff_sec: int = 60) -> list[tuple]:
        matched = []
        used_live = set()
        
        for p_fill in paper:
            best_match = None
            best_time_diff = float('inf')
            
            for i, l_fill in enumerate(live):
                if i in used_live:
                    continue
                if p_fill.symbol != l_fill.symbol:
                    continue
                if p_fill.side != l_fill.side:
                    continue
                
                time_diff = abs((l_fill.ts - p_fill.ts).total_seconds())
                if time_diff < max_time_diff_sec and time_diff < best_time_diff:
                    best_match = (i, l_fill)
                    best_time_diff = time_diff
            
            if best_match:
                used_live.add(best_match[0])
                matched.append((p_fill, best_match[1]))
        
        return matched
    
    def _fills_match(self, p: Fill, l: Fill) -> bool:
        if p.symbol != l.symbol or p.side != l.side:
            return False
        return abs((l.ts - p.ts).total_seconds()) < 60
    
    def _compute_pnl(self, fills: list[Fill]) -> float:
        # Simplified — full implementation in P&L attributor
        pnl = 0
        positions = {}
        for f in sorted(fills, key=lambda x: x.ts):
            if f.symbol not in positions:
                positions[f.symbol] = {'qty': 0, 'avg': 0}
            pos = positions[f.symbol]
            signed = f.quantity if f.side == 'buy' else -f.quantity
            
            if np.sign(signed) == np.sign(pos['qty']) or pos['qty'] == 0:
                new_qty = pos['qty'] + signed
                if new_qty != 0:
                    pos['avg'] = (pos['avg'] * pos['qty'] + f.price * signed) / new_qty
                pos['qty'] = new_qty
            else:
                closing = min(abs(signed), abs(pos['qty']))
                pnl += closing * (f.price - pos['avg']) * np.sign(pos['qty'])
                pos['qty'] += signed
        
        return pnl
```

## Feature Edge Attribution

**Location:** `src/edge_testing/feature_attribution.py`
**Purpose:** Ablation tests. Which features actually contribute to edge vs. which are overfitting artifacts?

```python
import numpy as np
import pandas as pd
from copy import deepcopy


class FeatureEdgeAttributor:
    """
    For each feature in a strategy, measure its marginal contribution to Sharpe.
    Compare against random-feature noise floor to identify overfit features.
    """
    
    def __init__(self, backtest_runner):
        self.runner = backtest_runner
    
    def attribute(self, strategy_factory, historical_data: pd.DataFrame,
                   features: list[str], 
                   n_random_tests: int = 10) -> dict:
        
        baseline_result = self.runner.run(
            historical_data, strategy_factory, None
        )
        baseline_sharpe = self._sharpe(baseline_result.oos_returns)
        
        contributions = {}
        
        # Test each feature by ablation
        for feature in features:
            def modified_factory():
                strat = strategy_factory()
                self._disable_feature(strat, feature)
                return strat
            
            ablated_result = self.runner.run(
                historical_data, modified_factory, None
            )
            ablated_sharpe = self._sharpe(ablated_result.oos_returns)
            
            contribution = baseline_sharpe - ablated_sharpe
            contributions[feature] = {
                'baseline_sharpe': baseline_sharpe,
                'without_feature_sharpe': ablated_sharpe,
                'contribution': contribution,
                'pct_contribution': (contribution / baseline_sharpe 
                                     if baseline_sharpe > 0 else 0),
            }
        
        # Estimate noise floor: what's the Sharpe contribution of random features?
        random_contributions = []
        for _ in range(n_random_tests):
            random_series = pd.Series(
                np.random.randn(len(historical_data)),
                index=historical_data.index
            )
            
            def random_factory():
                strat = strategy_factory()
                self._add_random_feature(strat, random_series)
                return strat
            
            random_result = self.runner.run(
                historical_data, random_factory, None
            )
            random_sharpe = self._sharpe(random_result.oos_returns)
            random_contributions.append(random_sharpe - baseline_sharpe)
        
        noise_floor = np.std(random_contributions) * 2
        
        # Flag features below noise floor
        for feature, data in contributions.items():
            data['above_noise_floor'] = abs(data['contribution']) > noise_floor
            data['noise_floor_2sigma'] = noise_floor
            if not data['above_noise_floor']:
                data['warning'] = (f"Feature contribution {data['contribution']:.3f} "
                                   f"within 2-sigma of noise ({noise_floor:.3f}) — "
                                   f"likely overfitting")
        
        return contributions
    
    def _sharpe(self, returns: pd.Series) -> float:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def _disable_feature(self, strategy, feature: str):
        """Strategy must implement disable_feature method."""
        if hasattr(strategy, 'disable_feature'):
            strategy.disable_feature(feature)
    
    def _add_random_feature(self, strategy, random_series: pd.Series):
        """Strategy must implement add_feature method."""
        if hasattr(strategy, 'add_feature'):
            strategy.add_feature('_random_test', random_series)
```

## Regime Edge Analysis

**Location:** `src/edge_testing/regime_edge.py`
**Purpose:** Decompose strategy returns by market regime. Reveals where edge actually comes from and whether it's regime-dependent.

```python
import numpy as np
import pandas as pd
from enum import Enum


class MarketRegime(Enum):
    LOW_VOL_TRENDING = 'low_vol_trending'
    HIGH_VOL_TRENDING = 'high_vol_trending'
    LOW_VOL_CHOPPY = 'low_vol_choppy'
    HIGH_VOL_CHOPPY = 'high_vol_choppy'
    CRISIS = 'crisis'


class RegimeEdgeAnalyzer:
    """
    Decompose strategy returns by market regime.
    Reveals where edge actually comes from.
    """
    
    def __init__(self, vol_window: int = 20, trend_window: int = 63):
        self.vol_window = vol_window
        self.trend_window = trend_window
    
    def classify_regimes(self, market_data: pd.DataFrame,
                          benchmark_col: str = None) -> pd.Series:
        """Classify each period into one of 5 regimes."""
        if benchmark_col is None:
            benchmark_col = market_data.columns[0]
        
        prices = market_data[benchmark_col]
        returns = prices.pct_change()
        
        realized_vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        vol_median = realized_vol.median()
        high_vol = realized_vol > vol_median
        
        trend = prices.pct_change(self.trend_window).abs()
        trend_median = trend.median()
        trending = trend > trend_median
        
        # Crisis: top 5% vol periods
        crisis_threshold = realized_vol.quantile(0.95)
        crisis = realized_vol > crisis_threshold
        
        regimes = pd.Series(index=market_data.index, dtype=str)
        regimes[crisis] = MarketRegime.CRISIS.value
        regimes[~crisis & high_vol & trending] = MarketRegime.HIGH_VOL_TRENDING.value
        regimes[~crisis & high_vol & ~trending] = MarketRegime.HIGH_VOL_CHOPPY.value
        regimes[~crisis & ~high_vol & trending] = MarketRegime.LOW_VOL_TRENDING.value
        regimes[~crisis & ~high_vol & ~trending] = MarketRegime.LOW_VOL_CHOPPY.value
        
        return regimes.fillna(MarketRegime.LOW_VOL_CHOPPY.value)
    
    def analyze(self, strategy_returns: pd.Series,
                 market_data: pd.DataFrame) -> dict:
        regimes = self.classify_regimes(market_data)
        regimes = regimes.reindex(strategy_returns.index, method='ffill')
        
        results = {}
        total_return_sum = strategy_returns.sum()
        
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
            mask = regimes == regime
            regime_returns = strategy_returns[mask]
            
            if len(regime_returns) < 20:
                continue
            
            results[regime] = {
                'n_periods': len(regime_returns),
                'pct_of_time': mask.mean(),
                'sharpe': self._sharpe(regime_returns),
                'mean_return_annualized': regime_returns.mean() * 252,
                'vol_annualized': regime_returns.std() * np.sqrt(252),
                'win_rate': (regime_returns > 0).mean(),
                'contribution_to_total': (regime_returns.sum() / total_return_sum
                                           if total_return_sum != 0 else 0),
                'max_drawdown': self._max_drawdown(regime_returns),
            }
        
        # Concentration analysis
        contributions = [r['contribution_to_total'] for r in results.values()]
        if contributions:
            max_contribution = max(contributions)
            sorted_contributions = sorted(contributions, reverse=True)
            top_two_pct = sum(sorted_contributions[:2])
        else:
            max_contribution = 0
            top_two_pct = 0
        
        warnings = []
        if max_contribution > 0.7:
            warnings.append(
                f"Edge concentrated: {max_contribution:.0%} of returns from one regime — "
                f"vulnerable to regime change"
            )
        if top_two_pct > 0.9 and len(contributions) > 2:
            warnings.append(
                f"Top 2 regimes contribute {top_two_pct:.0%} — "
                f"strategy may be regime-specific"
            )
        
        # Identify worst regime
        worst_sharpe_regime = min(
            results.keys(), 
            key=lambda k: results[k]['sharpe']
        ) if results else None
        
        if worst_sharpe_regime:
            worst = results[worst_sharpe_regime]
            if worst['sharpe'] < -0.5:
                warnings.append(
                    f"Strategy has strongly negative Sharpe ({worst['sharpe']:.2f}) "
                    f"in {worst_sharpe_regime} regime"
                )
        
        results['_meta'] = {
            'edge_concentration': max_contribution,
            'top_2_regime_pct': top_two_pct,
            'warnings': warnings,
            'worst_regime': worst_sharpe_regime,
        }
        
        return results
    
    def _sharpe(self, returns: pd.Series) -> float:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        equity = (1 + returns).cumprod()
        return ((equity - equity.cummax()) / equity.cummax()).min()
```

## Edge Decay Detection

**Location:** `src/edge_testing/decay_detection.py`
**Purpose:** Detect when edge is decaying over time. Edges die; you need to know when yours has.

```python
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu


class EdgeDecayMonitor:
    """
    Tracks whether strategy edge is decaying over time.
    Uses multiple tests: trend analysis, recent-vs-historical, rolling Sharpe.
    """
    
    def check_decay(self, returns: pd.Series, 
                     min_history_days: int = 252) -> dict:
        n = len(returns)
        
        if n < min_history_days:
            return {
                'status': 'insufficient_history',
                'days_available': n,
                'days_required': min_history_days,
            }
        
        # Test 1: Quarterly Sharpe trend (Mann-Kendall)
        quarter_size = n // 4
        quarters = [
            returns.iloc[i*quarter_size:(i+1)*quarter_size] 
            for i in range(4)
        ]
        quarter_sharpes = [self._sharpe(q) for q in quarters]
        
        tau, trend_p_value = kendalltau(range(4), quarter_sharpes)
        
        # Test 2: Recent period vs historical (Mann-Whitney U)
        recent_window = min(60, n // 4)
        recent = returns.iloc[-recent_window:]
        historical = returns.iloc[:-recent_window]
        
        try:
            mw_stat, mw_p = mannwhitneyu(recent, historical, alternative='less')
        except ValueError:
            mw_stat, mw_p = 0, 1.0
        
        # Test 3: Rolling 6-month Sharpe trajectory
        rolling_sharpe = returns.rolling(126).apply(
            lambda x: self._sharpe(x)
        ).dropna()
        
        if len(rolling_sharpe) > 20:
            # Linear regression slope
            x = np.arange(len(rolling_sharpe))
            slope, intercept = np.polyfit(x, rolling_sharpe.values, 1)
            slope_p = self._slope_significance(rolling_sharpe.values, slope)
        else:
            slope = 0
            slope_p = 1.0
        
        # Test 4: Recent drawdown vs historical max
        recent_dd = self._max_drawdown(recent)
        historical_dd = self._max_drawdown(historical)
        dd_worse = recent_dd < historical_dd * 1.3 if historical_dd < 0 else False
        
        # Combine signals
        decaying = (
            (tau < -0.5 and trend_p_value < 0.10) or
            (mw_p < 0.05) or
            (slope < -0.005 and slope_p < 0.05)
        )
        
        return {
            'quarter_sharpes': quarter_sharpes,
            'trend_tau': tau,
            'trend_p_value': trend_p_value,
            'recent_sharpe': self._sharpe(recent),
            'historical_sharpe': self._sharpe(historical),
            'recent_vs_historical_p': mw_p,
            'recent_significantly_worse': mw_p < 0.05,
            'rolling_sharpe_slope': slope,
            'rolling_sharpe_slope_p': slope_p,
            'recent_max_dd': recent_dd,
            'historical_max_dd': historical_dd,
            'dd_worse_than_historical': dd_worse,
            'decaying': decaying,
            'severity': self._severity(tau, trend_p_value, mw_p),
            'recommendation': self._recommend(tau, trend_p_value, mw_p, slope),
        }
    
    def _severity(self, tau, trend_p, mw_p) -> str:
        if tau < -0.7 and trend_p < 0.01:
            return 'strong_decay'
        if tau < -0.5 and mw_p < 0.05:
            return 'moderate_decay'
        if tau < -0.3:
            return 'possible_decay'
        return 'no_decay'
    
    def _recommend(self, tau, trend_p, mw_p, slope) -> str:
        if tau < -0.7 and trend_p < 0.01:
            return 'retire_strategy'
        if tau < -0.5 and mw_p < 0.05:
            return 'reduce_size_50pct_and_investigate'
        if tau < -0.3 or mw_p < 0.10:
            return 'monitor_closely_weekly_review'
        if slope < -0.005:
            return 'check_regime_hypothesis'
        return 'no_action_edge_stable'
    
    def _sharpe(self, returns: pd.Series) -> float:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def _max_drawdown(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0.0
        equity = (1 + returns).cumprod()
        return ((equity - equity.cummax()) / equity.cummax()).min()
    
    def _slope_significance(self, y: np.ndarray, slope: float) -> float:
        """Test if slope is significantly different from zero."""
        n = len(y)
        if n < 3:
            return 1.0
        x = np.arange(n)
        y_pred = slope * x + (y.mean() - slope * x.mean())
        residuals = y - y_pred
        se = np.sqrt(np.sum(residuals ** 2) / (n - 2)) / \
             np.sqrt(np.sum((x - x.mean()) ** 2))
        t_stat = slope / se if se > 0 else 0
        from scipy.stats import t
        return 2 * (1 - t.cdf(abs(t_stat), n - 2))
```

## Edge Dashboard

**Location:** `src/edge_testing/dashboard.py`
**Purpose:** Single unified view of edge status across all strategies. Daily-reviewed operational tool.

```python
import pandas as pd
from datetime import datetime, timedelta


class EdgeDashboard:
    """
    Unified view of edge status for every strategy.
    Used for daily operational review.
    """
    
    def __init__(self, strategies: list, state_store, 
                  null_framework, live_tracker_dict, 
                  decay_monitor, regime_analyzer,
                  divergence_monitor):
        self.strategies = strategies
        self.state = state_store
        self.null_framework = null_framework
        self.live_trackers = live_tracker_dict
        self.decay_monitor = decay_monitor
        self.regime_analyzer = regime_analyzer
        self.divergence_monitor = divergence_monitor
    
    def generate(self) -> pd.DataFrame:
        rows = []
        
        for strategy in self.strategies:
            sid = strategy.id
            
            live_returns = self.state.get_daily_returns(sid)
            days_live = len(live_returns)
            
            row = {
                'strategy': sid,
                'days_live': days_live,
            }
            
            # Backtest reference
            backtest = self.state.get_backtest_metrics(sid)
            row['backtest_sharpe'] = backtest.get('sharpe', 0)
            row['backtest_max_dd'] = backtest.get('max_drawdown', 0)
            
            # Live performance
            if days_live >= 30:
                row['live_sharpe'] = self._sharpe(live_returns)
                row['live_max_dd'] = self._max_dd(live_returns)
                
                # Divergence test
                tracker = self.live_trackers.get(sid)
                if tracker:
                    div = tracker.check_divergence(live_returns)
                    row['sharpe_z'] = div.get('sharpe_z_score', None)
                    row['sharpe_divergence_p'] = div.get('sharpe_p_value', None)
                    row['live_severity'] = div.get('severity', 'unknown')
                
                # Decay test
                if days_live >= 252:
                    decay = self.decay_monitor.check_decay(live_returns)
                    row['decay_tau'] = decay.get('trend_tau', None)
                    row['decay_severity'] = decay.get('severity', 'no_decay')
            
            # Paper-live divergence
            try:
                div = self.divergence_monitor.compare(sid, lookback_days=30)
                row['execution_slippage_bps'] = div.get('avg_price_diff_bps', 0)
                row['pnl_ratio_vs_paper'] = div.get('pnl_ratio', 1.0)
            except Exception:
                row['execution_slippage_bps'] = None
                row['pnl_ratio_vs_paper'] = None
            
            # Overall verdict
            row['verdict'] = self._compute_verdict(row)
            row['action'] = self._recommended_action(row)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _compute_verdict(self, row: dict) -> str:
        days_live = row.get('days_live', 0)
        
        if days_live < 30:
            return 'too_new_to_assess'
        
        if row.get('live_severity') == 'severely_underperforming':
            return 'no_edge_detected_live'
        
        if row.get('decay_severity') == 'strong_decay':
            return 'edge_decayed'
        
        if row.get('pnl_ratio_vs_paper', 1.0) < 0.5:
            return 'execution_destroying_edge'
        
        if row.get('live_severity') == 'on_track':
            return 'edge_confirmed'
        
        if row.get('live_severity') == 'underperforming':
            return 'edge_weak_monitoring'
        
        return 'indeterminate'
    
    def _recommended_action(self, row: dict) -> str:
        verdict = row.get('verdict')
        
        if verdict == 'no_edge_detected_live':
            return 'halt_strategy'
        if verdict == 'edge_decayed':
            return 'retire_strategy'
        if verdict == 'execution_destroying_edge':
            return 'investigate_broker_or_reduce'
        if verdict == 'edge_weak_monitoring':
            return 'reduce_size_25pct'
        if verdict == 'edge_confirmed':
            return 'continue'
        return 'continue_monitoring'
    
    def _sharpe(self, returns):
        import numpy as np
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def _max_dd(self, returns):
        if len(returns) == 0:
            return 0.0
        equity = (1 + returns).cumprod()
        return ((equity - equity.cummax()) / equity.cummax()).min()
```

## Edge Policy Configuration

**Location:** `configs/edge_policy.yaml`
**Purpose:** Pre-committed rules for strategy lifecycle actions based on edge test results. Removes in-the-moment judgment calls.

```yaml
strategy_lifecycle:
  paper_trading:
    min_days_before_live: 90
    required_tests:
      - name: "null_hypothesis_random_turnover"
        threshold: "p_value < 0.10"
      - name: "null_hypothesis_simple_momentum"
        threshold: "strategy_sharpe > momentum_sharpe * 1.2"
      - name: "white_reality_check"
        threshold: "p_value < 0.15"
      - name: "edge_concentration"
        threshold: "max_regime_contribution < 0.60"
      - name: "feature_attribution"
        threshold: "at_least_one_feature_above_noise_floor"
  
  live_promotion:
    initial_size_pct: 0.25
    ramp_to_full_days: 180
    ramp_schedule:
      - days: 30
        size_pct: 0.25
      - days: 60
        size_pct: 0.40
      - days: 90
        size_pct: 0.60
      - days: 180
        size_pct: 1.00
  
  live_trading:
    review_frequency_days: 7
    monthly_deep_review: true
    
    degradation_triggers:
      - condition: "sharpe_z < -2 AND days_live > 90"
        severity: medium
        action: "reduce_size_50pct"
        notify: ["operator"]
      
      - condition: "sharpe_z < -3 AND days_live > 90"
        severity: high
        action: "halt_strategy"
        notify: ["operator"]
      
      - condition: "live_max_dd < backtest_max_dd * 1.5"
        severity: high
        action: "halt_investigate"
        notify: ["operator"]
      
      - condition: "decay_tau < -0.7 AND decay_trend_p < 0.01"
        severity: critical
        action: "retire_strategy"
        notify: ["operator"]
      
      - condition: "pnl_ratio_vs_paper < 0.5 AND days_live > 30"
        severity: high
        action: "investigate_execution"
        notify: ["operator"]
      
      - condition: "execution_slippage_bps > 5"
        severity: medium
        action: "review_broker_routing"
  
  retirement:
    reactivation_cooldown_days: 180
    requires_new_research: true
    requires_new_backtest: true
    requires_paper_trading: true
```

## Edge Test Runner

**Location:** `scripts/run_edge_tests.py`
**Purpose:** Run all edge tests on all strategies. Designed for weekly scheduled execution.

```python
#!/usr/bin/env python
"""
Run complete edge test suite on all strategies.
Runs weekly via cron. Outputs report and sends alerts on degradation.
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine

from src.security.vault_client import VaultClient
from src.edge_testing.null_hypothesis import NullHypothesisFramework
from src.edge_testing.multiple_testing import MultipleTestingCorrection
from src.edge_testing.live_tracker import LiveEdgeTracker
from src.edge_testing.decay_detection import EdgeDecayMonitor
from src.edge_testing.regime_edge import RegimeEdgeAnalyzer
from src.edge_testing.paper_live_divergence import PaperLiveDivergence
from src.edge_testing.dashboard import EdgeDashboard
from src.strategies.state import StrategyStateStore
from src.monitoring.logging_config import setup_logging


logger = logging.getLogger(__name__)


def run_all_tests(strategies, state_store, engine) -> dict:
    results = {
        'generated_at': datetime.utcnow().isoformat(),
        'strategies': {},
    }
    
    null_framework = NullHypothesisFramework()
    decay_monitor = EdgeDecayMonitor()
    regime_analyzer = RegimeEdgeAnalyzer()
    divergence = PaperLiveDivergence(state_store)
    
    for strategy in strategies:
        sid = strategy.id
        logger.info(f"Running edge tests for {sid}")
        
        strategy_results = {}
        
        backtest_returns = state_store.get_backtest_returns(sid)
        live_returns = state_store.get_daily_returns(sid)
        market_data = state_store.get_market_data_for_strategy(sid)
        
        # Null hypothesis tests
        if backtest_returns is not None and len(backtest_returns) > 100:
            strategy_results['null_tests'] = null_framework.test_strategy(
                backtest_returns, market_data, n_simulations=5000
            )
        
        # Live performance test
        if live_returns is not None and len(live_returns) >= 30:
            backtest_metrics = state_store.get_backtest_metrics(sid)
            tracker = LiveEdgeTracker(sid, backtest_metrics)
            strategy_results['live_divergence'] = tracker.check_divergence(
                live_returns
            )
        
        # Decay detection
        if live_returns is not None and len(live_returns) >= 252:
            strategy_results['decay'] = decay_monitor.check_decay(live_returns)
        
        # Regime analysis
        if backtest_returns is not None and market_data is not None:
            strategy_results['regime_analysis'] = regime_analyzer.analyze(
                backtest_returns, market_data
            )
        
        # Paper-live divergence
        try:
            strategy_results['execution_divergence'] = divergence.compare(sid)
        except Exception as e:
            logger.warning(f"Divergence test failed for {sid}: {e}")
        
        results['strategies'][sid] = strategy_results
    
    return results


def apply_policy(results: dict, policy: dict) -> list[dict]:
    """Apply edge policy rules and return list of required actions."""
    actions = []
    
    for sid, strategy_results in results['strategies'].items():
        live = strategy_results.get('live_divergence', {})
        decay = strategy_results.get('decay', {})
        execution = strategy_results.get('execution_divergence', {})
        
        days_live = live.get('days_live', 0)
        sharpe_z = live.get('sharpe_z_score', 0)
        decay_tau = decay.get('trend_tau', 0)
        decay_p = decay.get('trend_p_value', 1)
        pnl_ratio = execution.get('pnl_ratio', 1)
        
        if sharpe_z < -3 and days_live > 90:
            actions.append({
                'strategy': sid, 
                'action': 'halt_strategy',
                'reason': f'sharpe_z={sharpe_z:.2f} after {days_live}d',
                'severity': 'high',
            })
        elif sharpe_z < -2 and days_live > 90:
            actions.append({
                'strategy': sid,
                'action': 'reduce_size_50pct',
                'reason': f'sharpe_z={sharpe_z:.2f} after {days_live}d',
                'severity': 'medium',
            })
        
        if decay_tau < -0.7 and decay_p < 0.01:
            actions.append({
                'strategy': sid,
                'action': 'retire_strategy',
                'reason': f'strong_decay tau={decay_tau:.2f}',
                'severity': 'critical',
            })
        
        if pnl_ratio < 0.5 and days_live > 30:
            actions.append({
                'strategy': sid,
                'action': 'investigate_execution',
                'reason': f'live P&L only {pnl_ratio:.0%} of paper',
                'severity': 'high',
            })
    
    return actions


def send_alerts(actions: list[dict], vault):
    if not actions:
        return
    
    message = "Edge test alerts:\n\n"
    for action in actions:
        message += (f"[{action['severity'].upper()}] {action['strategy']}: "
                    f"{action['action']} — {action['reason']}\n")
    
    import httpx
    httpx.post(
        'https://api.pushover.net/1/messages.json',
        data={
            'token': vault.get('PUSHOVER_API_TOKEN'),
            'user': vault.get('PUSHOVER_USER_KEY'),
            'title': 'FX Edge Test Results',
            'message': message,
            'priority': 1 if any(a['severity'] == 'critical' for a in actions) else 0,
        },
        timeout=10,
    )


def main():
    setup_logging()
    vault = VaultClient()
    
    db_url = (f"postgresql://fx:{vault.get('POSTGRES_FX_PASSWORD')}"
              f"@localhost:5432/fx")
    engine = create_engine(db_url)
    state_store = StrategyStateStore(engine)
    
    # Load strategies
    from src.runtime.run_engine import load_strategies
    strategies = load_strategies()
    
    # Run all tests
    results = run_all_tests(strategies, state_store, engine)
    
    # Save results
    output_path = Path('/opt/fx-system/edge_reports') / \
                   f'edge_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Results saved to {output_path}")
    
    # Apply policy and send alerts
    import yaml
    policy = yaml.safe_load(Path('configs/edge_policy.yaml').read_text())
    actions = apply_policy(results, policy)
    
    if actions:
        logger.warning(f"Edge tests generated {len(actions)} action items")
        for action in actions:
            logger.warning(f"  {action}")
        send_alerts(actions, vault)
    else:
        logger.info("All strategies passed edge tests")


if __name__ == '__main__':
    sys.exit(main() or 0)
```

## Cron Schedule Entry

**Location:** `/etc/cron.d/fx-system`
**Purpose:** Weekly edge test automation

```cron
# Weekly edge tests — Sundays at 02:00 UTC (after weekend data refresh)
0 2 * * 0 fx /opt/fx-system/venv/bin/python /opt/fx-system/scripts/run_edge_tests.py >> /var/log/fx-edge-tests.log 2>&1
```

## Streamlit Edge Report Viewer

**Location:** `scripts/edge_dashboard_app.py`
**Purpose:** Interactive Streamlit app for reviewing weekly edge test results.

```python
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import plotly.express as px


def main():
    st.set_page_config(page_title='FX Edge Dashboard', layout='wide')
    st.title('FX Trading System — Edge Status')
    
    reports_dir = Path('/opt/fx-system/edge_reports')
    reports = sorted(reports_dir.glob('edge_*.json'), reverse=True)
    
    if not reports:
        st.warning('No edge reports found. Run scripts/run_edge_tests.py')
        return
    
    selected = st.selectbox('Report', [r.name for r in reports])
    report_path = reports_dir / selected
    report = json.loads(report_path.read_text())
    
    st.caption(f"Generated: {report['generated_at']}")
    
    # Summary row
    strategies_data = report['strategies']
    n_strategies = len(strategies_data)
    n_with_edge = sum(
        1 for s in strategies_data.values()
        if s.get('live_divergence', {}).get('severity') == 'on_track'
    )
    n_concerning = sum(
        1 for s in strategies_data.values()
        if s.get('live_divergence', {}).get('severity') in 
           ('significantly_underperforming', 'severely_underperforming')
    )
    
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Strategies', n_strategies)
    col2.metric('Edge Confirmed', n_with_edge)
    col3.metric('Concerning', n_concerning)
    
    st.divider()
    
    for sid, results in strategies_data.items():
        with st.expander(f"**{sid}**"):
            # Live status
            live = results.get('live_divergence', {})
            if live:
                cols = st.columns(4)
                cols[0].metric('Days Live', live.get('days_live', 0))
                cols[1].metric('Live Sharpe', 
                              f"{live.get('live_sharpe', 0):.2f}")
                cols[2].metric('Expected Sharpe',
                              f"{live.get('expected_sharpe', 0):.2f}")
                cols[3].metric('Z-Score',
                              f"{live.get('sharpe_z_score', 0):.2f}")
                
                severity = live.get('severity', 'unknown')
                color = {
                    'on_track': 'green',
                    'underperforming': 'orange',
                    'significantly_underperforming': 'red',
                    'severely_underperforming': 'red',
                    'drawdown_exceeded': 'red',
                }.get(severity, 'gray')
                st.markdown(f"Severity: :{color}[{severity}]")
                st.caption(f"Recommendation: {live.get('recommendation')}")
            
            # Null tests
            null_tests = results.get('null_tests', {})
            if null_tests:
                st.subheader('Null Hypothesis Tests')
                null_df = pd.DataFrame([
                    {
                        'Null': name,
                        'Strategy Sharpe': r['strategy_sharpe'],
                        'Null Mean Sharpe': r['null_mean_sharpe'],
                        'P-value': r['p_value'],
                        'Edge?': '✓' if r['edge_exists'] else '✗',
                    }
                    for name, r in null_tests.items()
                ])
                st.dataframe(null_df)
            
            # Regime analysis
            regime = results.get('regime_analysis', {})
            if regime:
                st.subheader('Regime Decomposition')
                regime_rows = [
                    {'regime': k, **v}
                    for k, v in regime.items()
                    if k != '_meta'
                ]
                if regime_rows:
                    regime_df = pd.DataFrame(regime_rows)
                    st.dataframe(regime_df)
                
                warnings = regime.get('_meta', {}).get('warnings', [])
                for w in warnings:
                    st.warning(w)
            
            # Decay
            decay = results.get('decay', {})
            if decay and decay.get('status') != 'insufficient_history':
                st.subheader('Decay Analysis')
                cols = st.columns(3)
                cols[0].metric('Trend Tau', f"{decay.get('trend_tau', 0):.3f}")
                cols[1].metric('Trend P', f"{decay.get('trend_p_value', 1):.3f}")
                cols[2].metric('Recent Sharpe', 
                              f"{decay.get('recent_sharpe', 0):.2f}")
                if decay.get('decaying'):
                    st.error(f"Decay detected: {decay.get('severity')}")


if __name__ == '__main__':
    main()
```
