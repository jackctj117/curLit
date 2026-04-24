# 05 — Strategies

All six trading strategies: rate differential mean reversion, CB sentiment shift, carry+vol filter, FX momentum, FX value, and COT positioning reversal.

## Strategy 1: Rate Differential Mean Reversion

**Location:** `src/strategies/rate_diff_mean_reversion.py`
**Purpose:** Trade mean reversion of FX pair vs. its rate-differential fair value.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.execution.oms import OrderIntent

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    pair: str = 'EURUSD'
    rate_spread_series: str = 'US2Y_MINUS_DE2Y'
    lookback_days: int = 252          # 1 year for regression
    entry_z_threshold: float = 1.5
    exit_z_threshold: float = 0.3
    stop_loss_z: float = 3.5
    max_holding_days: int = 30
    volatility_target: float = 0.10
    max_position_pct: float = 0.20
    min_r_squared: float = 0.25
    signal_interval_seconds: int = 3600
    id: str = 'eurusd_rate_diff_mr'


class RateDiffMeanReversionStrategy:
    def __init__(self, config: StrategyConfig, data_provider, state_store):
        self.config = config
        self.data = data_provider
        self.state = state_store
        self._model = None
        self._last_fit_date = None
        self._current_position = 0
        self._entry_z = None
        self._entry_ts = None
    
    @property
    def id(self) -> str:
        return self.config.id
    
    @property
    def symbols(self) -> list[str]:
        return [self.config.pair]
    
    @property
    def signal_interval_seconds(self) -> int:
        return self.config.signal_interval_seconds
    
    def _fit_model(self, as_of: datetime) -> dict:
        """Refit the rate differential regression."""
        end = as_of
        start = end - timedelta(days=self.config.lookback_days * 2)
        
        df = self.data.get_aligned_series(
            symbols=[self.config.pair, 'US_2Y', 'DE_2Y'],
            start=start, end=end,
        )
        df = df.dropna().tail(self.config.lookback_days)
        
        if len(df) < self.config.lookback_days * 0.8:
            logger.warning(f"Insufficient data: {len(df)} rows")
            return None
        
        df['spread'] = df['US_2Y'] - df['DE_2Y']
        X = sm.add_constant(df[['spread']])
        y = df[self.config.pair]
        model = sm.OLS(y, X).fit()
        
        result = {
            'alpha': model.params['const'],
            'beta': model.params['spread'],
            'r_squared': model.rsquared,
            'residual_std': model.resid.std(),
            'fit_date': as_of,
            'n_obs': len(df),
        }
        
        self._model = result
        self._last_fit_date = as_of
        return result
    
    def _compute_signal(self, current_price: float, 
                         current_spread: float) -> dict:
        """Compute current deviation Z-score."""
        fair_value = self._model['alpha'] + self._model['beta'] * current_spread
        deviation = current_price - fair_value
        z_score = deviation / self._model['residual_std']
        
        return {
            'current_price': current_price,
            'current_spread': current_spread,
            'fair_value': fair_value,
            'deviation': deviation,
            'z_score': z_score,
            'model_quality': self._model['r_squared'],
        }
    
    def _compute_position_size(self, signal: dict, account_equity: float,
                                recent_vol: float) -> float:
        excess_z = abs(signal['z_score']) - self.config.entry_z_threshold
        conviction_multiplier = min(excess_z / 1.0, 1.5)
        
        vol_target_notional = (
            account_equity * self.config.volatility_target / recent_vol
        )
        notional = vol_target_notional * (0.5 + 0.5 * conviction_multiplier)
        
        max_notional = account_equity * self.config.max_position_pct
        notional = min(notional, max_notional)
        
        direction = -1 if signal['z_score'] > 0 else 1
        units = direction * notional / signal['current_price']
        return units
    
    def _should_exit(self, signal: dict, now: datetime) -> tuple[bool, str]:
        """Check exit conditions on open position."""
        if self._current_position == 0:
            return False, ''
        
        z = signal['z_score']
        
        if self._current_position > 0 and z >= -self.config.exit_z_threshold:
            return True, 'mean_reversion_hit'
        if self._current_position < 0 and z <= self.config.exit_z_threshold:
            return True, 'mean_reversion_hit'
        
        if self._current_position > 0 and z < -self.config.stop_loss_z:
            return True, 'stop_loss'
        if self._current_position < 0 and z > self.config.stop_loss_z:
            return True, 'stop_loss'
        
        if self._entry_ts and (now - self._entry_ts).days > self.config.max_holding_days:
            return True, 'time_stop'
        
        return False, ''
    
    async def generate_intents(self, prices: dict, broker) -> list[OrderIntent]:
        now = datetime.utcnow()
        
        if (self._last_fit_date is None or 
            (now - self._last_fit_date).days >= 7):
            result = self._fit_model(now)
            if result is None:
                return []
        
        if self._model['r_squared'] < self.config.min_r_squared:
            if self._current_position != 0:
                return [self._flatten_intent('low_r_squared')]
            return []
        
        tick = prices.get(self.config.pair)
        if tick is None:
            return []
        
        current_price = (tick['bid'] + tick['ask']) / 2
        current_spread = await self.data.get_latest_rate_spread()
        if current_spread is None:
            return []
        
        signal = self._compute_signal(current_price, current_spread)
        self.state.record_signal(self.id, now, signal)
        
        should_exit, exit_reason = self._should_exit(signal, now)
        if should_exit:
            return [self._flatten_intent(exit_reason)]
        
        if self._current_position == 0:
            if abs(signal['z_score']) >= self.config.entry_z_threshold:
                account = broker.get_account()
                recent_vol = await self.data.get_recent_vol(
                    self.config.pair, window=20
                )
                if recent_vol is None or recent_vol == 0:
                    return []
                
                size = self._compute_position_size(
                    signal, account.equity, recent_vol
                )
                self._current_position = size
                self._entry_z = signal['z_score']
                self._entry_ts = now
                self.state.record_entry(self.id, now, signal, size)
                
                return [OrderIntent(
                    strategy_id=self.id,
                    symbol=self.config.pair,
                    target_position=size,
                    urgency='normal',
                )]
        
        return []
    
    def _flatten_intent(self, reason: str) -> OrderIntent:
        intent = OrderIntent(
            strategy_id=self.id,
            symbol=self.config.pair,
            target_position=0,
            urgency='normal',
        )
        self.state.record_exit(self.id, datetime.utcnow(), reason)
        self._current_position = 0
        self._entry_z = None
        self._entry_ts = None
        return intent
```

## Strategy 2: CB Sentiment Shift

**Location:** `src/strategies/cb_sentiment_shift.py`
**Purpose:** Trade hawkish/dovish shifts in central bank statements, holding 10 business days.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd

from src.execution.oms import OrderIntent

logger = logging.getLogger(__name__)


@dataclass
class CBSentimentConfig:
    cb_to_pair: dict = field(default_factory=lambda: {
        'fed': ('EURUSD', 'short'),   # Hawkish Fed → short EUR/USD
        'ecb': ('EURUSD', 'long'),    # Hawkish ECB → long EUR/USD
        'boe': ('GBPUSD', 'long'),
        'boj': ('USDJPY', 'short'),
        'boc': ('USDCAD', 'short'),
    })
    
    strong_shift_percentile: float = 0.15
    min_diff_score_abs: float = 0.3
    holding_days: int = 10
    hard_stop_pct: float = 0.015
    trailing_trigger_pct: float = 0.02
    trailing_distance_pct: float = 0.015
    risk_per_trade_pct: float = 0.01
    max_concurrent_positions: int = 3
    signal_interval_seconds: int = 300
    id: str = 'cb_sentiment_shift'


@dataclass
class OpenPosition:
    symbol: str
    entry_ts: datetime
    entry_price: float
    quantity: float
    direction: int
    stop_loss: float
    trailing_stop: float | None = None
    peak_price: float | None = None
    source_cb: str = ''
    source_doc_id: str = ''


class CBSentimentShiftStrategy:
    def __init__(self, config: CBSentimentConfig, data_provider, 
                 nlp_provider, state_store):
        self.config = config
        self.data = data_provider
        self.nlp = nlp_provider
        self.state = state_store
        self.open_positions: dict[str, OpenPosition] = {}
        self._historical_diffs = None
        self._last_refresh = None
    
    @property
    def id(self) -> str:
        return self.config.id
    
    @property
    def symbols(self) -> list[str]:
        return list(set(p for p, _ in self.config.cb_to_pair.values()))
    
    @property
    def signal_interval_seconds(self) -> int:
        return self.config.signal_interval_seconds
    
    def _refresh_historical_distribution(self):
        self._historical_diffs = self.nlp.get_historical_diff_scores(
            cbs=list(self.config.cb_to_pair.keys()),
            lookback_years=5,
        )
        self._last_refresh = datetime.utcnow()
    
    def _get_threshold(self, cb: str) -> tuple[float, float]:
        cb_diffs = self._historical_diffs[
            self._historical_diffs['cb'] == cb
        ]['net_shift']
        
        if len(cb_diffs) < 10:
            return (-self.config.min_diff_score_abs, 
                    self.config.min_diff_score_abs)
        
        dovish_thr = np.percentile(cb_diffs, 
                                     self.config.strong_shift_percentile * 100)
        hawkish_thr = np.percentile(cb_diffs,
                                      (1 - self.config.strong_shift_percentile) * 100)
        
        dovish_thr = min(dovish_thr, -self.config.min_diff_score_abs)
        hawkish_thr = max(hawkish_thr, self.config.min_diff_score_abs)
        
        return dovish_thr, hawkish_thr
    
    async def _check_new_events(self, now: datetime) -> list[dict]:
        recent = self.nlp.get_recent_diff_events(
            since=now - timedelta(hours=2),
            cbs=list(self.config.cb_to_pair.keys()),
        )
        
        signals = []
        for event in recent:
            cb = event['cb']
            if cb not in self.config.cb_to_pair:
                continue
            
            dovish_thr, hawkish_thr = self._get_threshold(cb)
            shift = event['net_shift']
            pair, cb_hawkish_side = self.config.cb_to_pair[cb]
            
            if shift >= hawkish_thr:
                direction = 1 if cb_hawkish_side == 'long' else -1
                signals.append({
                    'cb': cb, 'doc_id': event['doc_id'],
                    'shift': shift, 'pair': pair,
                    'direction': direction, 'signal_type': 'hawkish',
                })
            elif shift <= dovish_thr:
                direction = -1 if cb_hawkish_side == 'long' else 1
                signals.append({
                    'cb': cb, 'doc_id': event['doc_id'],
                    'shift': shift, 'pair': pair,
                    'direction': direction, 'signal_type': 'dovish',
                })
        
        return signals
    
    def _compute_size(self, account_equity: float, entry_price: float,
                       stop_distance: float) -> float:
        risk_amount = account_equity * self.config.risk_per_trade_pct
        size_notional = risk_amount / stop_distance
        return size_notional / entry_price
    
    def _update_trailing_stops(self, prices: dict) -> list[OrderIntent]:
        exits = []
        for symbol, pos in list(self.open_positions.items()):
            tick = prices.get(symbol)
            if tick is None:
                continue
            current = (tick['bid'] + tick['ask']) / 2
            
            pnl_pct = (current - pos.entry_price) / pos.entry_price * pos.direction
            
            if pnl_pct >= self.config.trailing_trigger_pct:
                if pos.direction > 0:
                    new_peak = max(pos.peak_price or pos.entry_price, current)
                    new_trail = new_peak * (1 - self.config.trailing_distance_pct)
                    if pos.trailing_stop is None or new_trail > pos.trailing_stop:
                        pos.peak_price = new_peak
                        pos.trailing_stop = new_trail
                else:
                    new_peak = min(pos.peak_price or pos.entry_price, current)
                    new_trail = new_peak * (1 + self.config.trailing_distance_pct)
                    if pos.trailing_stop is None or new_trail < pos.trailing_stop:
                        pos.peak_price = new_peak
                        pos.trailing_stop = new_trail
            
            exit_reason = None
            if pos.direction > 0:
                if current <= pos.stop_loss:
                    exit_reason = 'hard_stop'
                elif pos.trailing_stop and current <= pos.trailing_stop:
                    exit_reason = 'trailing_stop'
            else:
                if current >= pos.stop_loss:
                    exit_reason = 'hard_stop'
                elif pos.trailing_stop and current >= pos.trailing_stop:
                    exit_reason = 'trailing_stop'
            
            days_held = (datetime.utcnow() - pos.entry_ts).days
            if days_held >= self.config.holding_days:
                exit_reason = 'time_exit'
            
            if exit_reason:
                self.state.record_exit(
                    self.id, datetime.utcnow(), 
                    {'symbol': symbol, 'reason': exit_reason, 
                     'pnl_pct': pnl_pct}
                )
                exits.append(OrderIntent(
                    strategy_id=self.id,
                    symbol=symbol,
                    target_position=0,
                    urgency='normal',
                ))
                del self.open_positions[symbol]
        
        return exits
    
    async def generate_intents(self, prices: dict, broker) -> list[OrderIntent]:
        now = datetime.utcnow()
        
        if (self._last_refresh is None or 
            (now - self._last_refresh).days >= 7):
            self._refresh_historical_distribution()
        
        intents = []
        intents.extend(self._update_trailing_stops(prices))
        
        if len(self.open_positions) >= self.config.max_concurrent_positions:
            return intents
        
        signals = await self._check_new_events(now)
        
        account = broker.get_account()
        for signal in signals:
            pair = signal['pair']
            direction = signal['direction']
            
            if pair in self.open_positions:
                continue
            
            tick = prices.get(pair)
            if tick is None:
                continue
            
            entry_price = tick['ask'] if direction > 0 else tick['bid']
            stop_price = entry_price * (1 - direction * self.config.hard_stop_pct)
            stop_distance = abs(entry_price - stop_price)
            
            size = self._compute_size(account.equity, entry_price, stop_distance)
            quantity = direction * size
            
            self.open_positions[pair] = OpenPosition(
                symbol=pair, entry_ts=now, entry_price=entry_price,
                quantity=quantity, direction=direction,
                stop_loss=stop_price, source_cb=signal['cb'],
                source_doc_id=signal['doc_id'],
            )
            
            self.state.record_entry(self.id, now, {
                **signal, 'entry_price': entry_price,
                'quantity': quantity, 'stop_loss': stop_price,
            }, quantity)
            
            intents.append(OrderIntent(
                strategy_id=self.id,
                symbol=pair,
                target_position=quantity,
                urgency='urgent',
            ))
        
        return intents
```

## Strategy 3: Carry + Volatility Filter

**Location:** `src/strategies/carry_vol_filter.py`
**Purpose:** G10 carry trade with vol-regime-based exposure scaling. Based on Menkhoff et al. 2012.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.execution.oms import OrderIntent

logger = logging.getLogger(__name__)


@dataclass
class CarryVolFilterConfig:
    currencies: list[str] = field(default_factory=lambda: [
        'USD', 'EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD', 'NOK', 'SEK'
    ])
    top_k: int = 3
    bottom_k: int = 3
    rebalance_day: int = 1
    vol_index_series: str = 'CVIX'
    vol_lookback_days: int = 120
    vol_z_thresholds: dict = field(default_factory=lambda: {
        0.0: 1.00,
        1.0: 0.75,
        2.0: 0.50,
        3.0: 0.00,
    })
    target_portfolio_vol: float = 0.08
    max_position_pct: float = 0.15
    min_rate_spread: float = 0.005
    signal_interval_seconds: int = 3600
    id: str = 'carry_vol_filter'


@dataclass
class CarryPosition:
    currency: str
    side: int
    weight: float
    entry_ts: datetime
    reference_rate: float


class CarryVolFilterStrategy:
    RATE_SERIES_MAP = {
        'USD': 'USD_3M_OIS',
        'EUR': 'EUR_3M_ESTR_OIS',
        'JPY': 'JPY_3M_TONA_OIS',
        'GBP': 'GBP_3M_SONIA_OIS',
        'CHF': 'CHF_3M_SARON_OIS',
        'CAD': 'CAD_3M_CORRA_OIS',
        'AUD': 'AUD_3M_BBSW',
        'NZD': 'NZD_3M_BKBM',
        'NOK': 'NOK_3M_NIBOR',
        'SEK': 'SEK_3M_STIBOR',
    }
    
    def __init__(self, config: CarryVolFilterConfig, data_provider, state_store):
        self.config = config
        self.data = data_provider
        self.state = state_store
        self.current_positions: dict[str, CarryPosition] = {}
        self._last_rebalance: Optional[date] = None
        self._current_exposure: float = 1.0
    
    @property
    def id(self) -> str:
        return self.config.id
    
    @property
    def symbols(self) -> list[str]:
        return [f'{c}USD' if c != 'USD' else 'EURUSD' 
                for c in self.config.currencies]
    
    @property
    def signal_interval_seconds(self) -> int:
        return self.config.signal_interval_seconds
    
    def _get_current_rates(self, as_of: datetime) -> dict[str, float]:
        rates = {}
        for ccy in self.config.currencies:
            series_id = self.RATE_SERIES_MAP[ccy]
            rate = self.data.get_latest_value(series_id, as_of)
            if rate is not None:
                rates[ccy] = rate
        return rates
    
    def _construct_baskets(self, rates: dict[str, float]) -> dict:
        sorted_ccys = sorted(rates.items(), key=lambda x: x[1], reverse=True)
        long_basket = sorted_ccys[:self.config.top_k]
        short_basket = sorted_ccys[-self.config.bottom_k:]
        
        rate_spread = long_basket[-1][1] - short_basket[0][1]
        if rate_spread < self.config.min_rate_spread:
            return {'long': [], 'short': [], 'spread': rate_spread}
        
        return {
            'long': [(c, r, 1.0/self.config.top_k) for c, r in long_basket],
            'short': [(c, r, 1.0/self.config.bottom_k) for c, r in short_basket],
            'spread': rate_spread,
        }
    
    def _compute_vol_z_score(self, as_of: datetime) -> float:
        end = as_of
        start = end - timedelta(days=self.config.vol_lookback_days * 2)
        
        vol_series = self.data.get_series(
            self.config.vol_index_series, start, end
        )
        if len(vol_series) < self.config.vol_lookback_days * 0.8:
            return 0.0
        
        recent = vol_series.tail(self.config.vol_lookback_days)
        current = vol_series.iloc[-1]
        mean = recent.mean()
        std = recent.std()
        if std == 0:
            return 0.0
        return (current - mean) / std
    
    def _exposure_multiplier(self, vol_z: float) -> float:
        sorted_thresholds = sorted(self.config.vol_z_thresholds.items())
        multiplier = 1.0
        for threshold, mult in sorted_thresholds:
            if vol_z >= threshold:
                multiplier = mult
        return multiplier
    
    def _should_rebalance(self, now: datetime) -> bool:
        if self._last_rebalance is None:
            return True
        today = now.date()
        if today == self._last_rebalance:
            return False
        if today.month != self._last_rebalance.month:
            return today.weekday() < 5
        return False
    
    def _compute_position_size(self, currency: str, basket_weight: float,
                                account_equity: float, exposure_mult: float,
                                pair_vol: float) -> float:
        base_notional = (
            account_equity * 
            self.config.target_portfolio_vol / pair_vol * 
            basket_weight * 
            exposure_mult
        )
        max_notional = account_equity * self.config.max_position_pct
        return min(base_notional, max_notional)
    
    def _currency_to_pair(self, ccy: str, side: int) -> tuple[str, int]:
        if ccy == 'USD':
            return None, 0
        pair = f'{ccy}USD'
        return pair, side
    
    async def generate_intents(self, prices: dict, broker) -> list[OrderIntent]:
        now = datetime.utcnow()
        vol_z = self._compute_vol_z_score(now)
        new_exposure = self._exposure_multiplier(vol_z)
        
        intents = []
        
        if new_exposure < self._current_exposure * 0.8 and self.current_positions:
            intents.extend(self._scale_positions(new_exposure / self._current_exposure))
            self._current_exposure = new_exposure
        
        if not self._should_rebalance(now):
            return intents
        
        rates = self._get_current_rates(now)
        if len(rates) < self.config.top_k + self.config.bottom_k:
            return intents
        
        baskets = self._construct_baskets(rates)
        
        if not baskets['long']:
            intents.extend(self._flatten_all())
            self._last_rebalance = now.date()
            return intents
        
        account = broker.get_account()
        new_positions: dict[str, CarryPosition] = {}
        
        for ccy, rate, weight in baskets['long']:
            new_positions[ccy] = CarryPosition(
                currency=ccy, side=1, weight=weight,
                entry_ts=now, reference_rate=rate
            )
        for ccy, rate, weight in baskets['short']:
            new_positions[ccy] = CarryPosition(
                currency=ccy, side=-1, weight=weight,
                entry_ts=now, reference_rate=rate
            )
        
        all_ccys = set(self.current_positions.keys()) | set(new_positions.keys())
        
        for ccy in all_ccys:
            pair, _ = self._currency_to_pair(ccy, 1)
            if pair is None:
                continue
            
            if ccy in new_positions:
                pos = new_positions[ccy]
                pair_vol = self.data.get_realized_vol(pair, window=20)
                if pair_vol is None or pair_vol == 0:
                    continue
                
                notional = self._compute_position_size(
                    ccy, pos.weight, account.equity, 
                    new_exposure, pair_vol
                )
                target_qty = pos.side * notional / prices[pair]['bid']
            else:
                target_qty = 0
            
            intents.append(OrderIntent(
                strategy_id=self.id,
                symbol=pair,
                target_position=target_qty,
                urgency='normal',
            ))
        
        self.current_positions = new_positions
        self._current_exposure = new_exposure
        self._last_rebalance = now.date()
        
        self.state.record_rebalance(self.id, now, {
            'rates': rates,
            'long_basket': [c for c, _, _ in baskets['long']],
            'short_basket': [c for c, _, _ in baskets['short']],
            'vol_z': vol_z,
            'exposure': new_exposure,
        })
        
        return intents
    
    def _scale_positions(self, scale_factor: float) -> list[OrderIntent]:
        # Scale all current positions by scale_factor
        return []  # Implementation details
    
    def _flatten_all(self) -> list[OrderIntent]:
        intents = []
        for ccy in self.current_positions:
            pair, _ = self._currency_to_pair(ccy, 1)
            if pair is not None:
                intents.append(OrderIntent(
                    strategy_id=self.id,
                    symbol=pair,
                    target_position=0,
                    urgency='normal',
                ))
        self.current_positions.clear()
        return intents
```

## Strategy 4: FX Momentum

**Location:** `src/strategies/fx_momentum.py`
**Purpose:** Cross-sectional momentum on G10 currencies. Based on Menkhoff et al. 2012 JFE.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class FXMomentumConfig:
    currencies: list[str] = field(default_factory=lambda: [
        'EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD', 'NOK', 'SEK'
    ])
    lookback_months: int = 3
    holding_months: int = 1
    top_k: int = 3
    bottom_k: int = 3
    include_carry_in_return: bool = True
    target_vol: float = 0.08
    id: str = 'fx_momentum'


class FXMomentumStrategy:
    def compute_momentum_signal(self, as_of: datetime) -> dict:
        """Rank currencies by past total return."""
        lookback_days = self.config.lookback_months * 21
        
        returns = {}
        for ccy in self.config.currencies:
            total_return = self._compute_total_return(
                ccy, as_of - timedelta(days=lookback_days), as_of
            )
            if total_return is not None:
                returns[ccy] = total_return
        
        if len(returns) < self.config.top_k + self.config.bottom_k:
            return {'long': [], 'short': []}
        
        sorted_ccys = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        return {
            'long': [c for c, _ in sorted_ccys[:self.config.top_k]],
            'short': [c for c, _ in sorted_ccys[-self.config.bottom_k:]],
            'returns': returns,
        }
    
    def _compute_total_return(self, ccy: str, start: datetime, 
                                end: datetime) -> float:
        """Spot return plus accumulated carry."""
        spot_return = self._spot_return(ccy, start, end)
        
        if not self.config.include_carry_in_return:
            return spot_return
        
        rate_diff_daily = self._get_rate_differential(ccy, start, end)
        carry_return = rate_diff_daily.sum() / 100 / 252
        
        return spot_return + carry_return
```

## Strategy 5: FX Value (PPP Deviation)

**Location:** `src/strategies/fx_value.py`
**Purpose:** Long cheap currencies (per PPP), short expensive ones. Slow-moving, based on Asness/Moskowitz/Pedersen 2013.

```python
from datetime import date, timedelta
import pandas as pd


class FXValueStrategy:
    def compute_ppp_deviation(self, ccy: str, as_of: date) -> float:
        """
        Returns: % deviation from PPP fair value.
        Positive = overvalued (expensive) → expected to weaken
        Negative = undervalued (cheap) → expected to strengthen
        
        PPP fair value via relative CPI:
        FV_t = FV_0 × (CPI_domestic_t / CPI_domestic_0) / (CPI_US_t / CPI_US_0)
        """
        domestic_cpi = self.data.get_series(f'{ccy}_CPI', 
                                              start=as_of - timedelta(days=365*20))
        us_cpi = self.data.get_series('US_CPI',
                                         start=as_of - timedelta(days=365*20))
        
        domestic_idx = domestic_cpi / domestic_cpi.iloc[0] * 100
        us_idx = us_cpi / us_cpi.iloc[0] * 100
        
        rel_price = domestic_idx / us_idx
        
        start_date = domestic_cpi.index[0]
        start_spot = self.data.get_fx_rate(f'{ccy}USD', start_date)
        if start_spot is None:
            return None
        
        current_rel = rel_price.iloc[-1] / rel_price.iloc[0]
        ppp_fair_value = start_spot / current_rel
        
        current_spot = self.data.get_fx_rate(f'{ccy}USD', as_of)
        if current_spot is None:
            return None
        
        deviation_pct = (current_spot / ppp_fair_value - 1) * 100
        return deviation_pct
```

## Strategy 6: COT Positioning Reversal

**Location:** `src/strategies/cot_reversal.py`
**Purpose:** Contrarian trade on extreme speculative positioning in CFTC currency futures.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import text

from src.execution.oms import OrderIntent


@dataclass
class COTReversalConfig:
    cot_to_pair: dict = field(default_factory=lambda: {
        'EC': 'EURUSD',   # Euro FX
        'JY': 'USDJPY',   # JPY (inverted)
        'BP': 'GBPUSD',
        'SF': 'USDCHF',   # CHF (inverted)
        'CD': 'USDCAD',   # CAD (inverted)
        'AD': 'AUDUSD',
        'NE': 'NZDUSD',
    })
    inverted_pairs: set = field(default_factory=lambda: {'JY', 'SF', 'CD'})
    lookback_years: int = 3
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    max_holding_weeks: int = 8
    risk_per_trade_pct: float = 0.01
    id: str = 'cot_reversal'


class COTReversalStrategy:
    async def generate_intents(self, prices: dict, broker) -> list[OrderIntent]:
        now = datetime.utcnow()
        
        if now.weekday() not in (5, 6, 0):  # Sat, Sun, Mon
            return self._manage_existing_positions(prices)
        
        new_signals = self._compute_signals(now)
        intents = []
        
        for cot_code, signal in new_signals.items():
            pair = self.config.cot_to_pair[cot_code]
            
            direction = signal['direction']
            if cot_code in self.config.inverted_pairs:
                direction = -direction
            
            # Contrarian: specs long extreme → we go short
            trade_direction = -direction
            
            if pair not in self.open_positions and abs(signal['z_score']) > self.config.zscore_entry:
                intents.append(self._build_entry(pair, trade_direction, signal, 
                                                    prices, broker))
        
        intents.extend(self._manage_existing_positions(prices))
        return intents
    
    def _compute_signals(self, now: datetime) -> dict:
        signals = {}
        
        for cot_code in self.config.cot_to_pair:
            query = text("""
                SELECT report_date, longs, shorts, open_interest
                FROM cot_positioning 
                WHERE currency_code = :code AND category = 'leveraged_funds'
                AND report_date >= :start
                ORDER BY report_date
            """)
            history = self.data.execute(query, {
                'code': cot_code,
                'start': now - timedelta(days=self.config.lookback_years * 365),
            })
            
            if len(history) < 50:
                continue
            
            history['net_pct_oi'] = (
                (history['longs'] - history['shorts']) / history['open_interest']
            )
            
            current = history['net_pct_oi'].iloc[-1]
            mean = history['net_pct_oi'].iloc[:-1].mean()
            std = history['net_pct_oi'].iloc[:-1].std()
            
            if std == 0:
                continue
            
            z = (current - mean) / std
            
            signals[cot_code] = {
                'z_score': z,
                'direction': 1 if current > mean else -1,
                'current_net_pct': current,
                'historical_mean': mean,
            }
        
        return signals
```

## Strategy State Store

**Location:** `src/strategies/state.py`
**Purpose:** Persist strategy state for crash recovery and monitoring.

```python
from datetime import datetime
import json
from sqlalchemy import text


class StrategyStateStore:
    def __init__(self, engine):
        self.engine = engine
        self._create_tables()
    
    def _create_tables(self):
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS strategy_signals (
                    ts TIMESTAMPTZ NOT NULL,
                    strategy_id TEXT NOT NULL,
                    signal_data JSONB,
                    PRIMARY KEY (ts, strategy_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS strategy_trades (
                    ts TIMESTAMPTZ NOT NULL,
                    strategy_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details JSONB,
                    PRIMARY KEY (ts, strategy_id, action)
                )
            """))
    
    def record_signal(self, strategy_id: str, ts: datetime, signal: dict):
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO strategy_signals (ts, strategy_id, signal_data)
                VALUES (:ts, :sid, :data)
                ON CONFLICT (ts, strategy_id) DO UPDATE 
                SET signal_data = EXCLUDED.signal_data
            """), {'ts': ts, 'sid': strategy_id, 'data': json.dumps(signal)})
    
    def record_entry(self, strategy_id: str, ts: datetime, 
                      signal: dict, size: float):
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO strategy_trades (ts, strategy_id, action, details)
                VALUES (:ts, :sid, 'entry', :details)
            """), {
                'ts': ts, 'sid': strategy_id,
                'details': json.dumps({**signal, 'size': size})
            })
    
    def record_exit(self, strategy_id: str, ts: datetime, reason: str):
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO strategy_trades (ts, strategy_id, action, details)
                VALUES (:ts, :sid, 'exit', :details)
            """), {
                'ts': ts, 'sid': strategy_id,
                'details': json.dumps({'reason': reason})
            })
    
    def record_rebalance(self, strategy_id: str, ts: datetime, details: dict):
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO strategy_trades (ts, strategy_id, action, details)
                VALUES (:ts, :sid, 'rebalance', :details)
            """), {
                'ts': ts, 'sid': strategy_id,
                'details': json.dumps(details)
            })
    
    def load_current_position(self, strategy_id: str) -> dict | None:
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT action, details, ts FROM strategy_trades
                WHERE strategy_id = :sid
                ORDER BY ts DESC LIMIT 1
            """), {'sid': strategy_id}).fetchone()
        
        if result is None or result[0] == 'exit':
            return None
        
        return {**json.loads(result[1]), 'entry_ts': result[2]}
```

## Backtest Strategy Adapter

**Location:** `src/backtest/adapters.py`
**Purpose:** Wraps live strategies for use in walk-forward backtesting.

```python
import pandas as pd
import statsmodels.api as sm


class BacktestStrategy:
    """Adapter to run the live strategy in backtest mode."""
    def __init__(self, config):
        self.config = config
        self._model = None
    
    def fit(self, train_data: pd.DataFrame):
        df = train_data.dropna()
        df['spread'] = df['US_2Y'] - df['DE_2Y']
        X = sm.add_constant(df[['spread']])
        y = df[self.config.pair]
        model = sm.OLS(y, X).fit()
        self._model = {
            'alpha': model.params['const'],
            'beta': model.params['spread'],
            'r_squared': model.rsquared,
            'residual_std': model.resid.std(),
        }
        self.params = self._model.copy()
    
    def generate_signals(self, test_data: pd.DataFrame) -> pd.Series:
        df = test_data.copy()
        df['spread'] = df['US_2Y'] - df['DE_2Y']
        df['fair_value'] = (self._model['alpha'] + 
                           self._model['beta'] * df['spread'])
        df['deviation'] = df[self.config.pair] - df['fair_value']
        df['z'] = df['deviation'] / self._model['residual_std']
        
        positions = []
        pos = 0
        entry_z = None
        entry_i = None
        for i, z in enumerate(df['z']):
            if pos == 0:
                if z > self.config.entry_z_threshold:
                    pos, entry_z, entry_i = -1, z, i
                elif z < -self.config.entry_z_threshold:
                    pos, entry_z, entry_i = 1, z, i
            else:
                days_held = i - entry_i
                stop_hit = (pos > 0 and z < -self.config.stop_loss_z) or \
                           (pos < 0 and z > self.config.stop_loss_z)
                exit_hit = (pos > 0 and z >= -self.config.exit_z_threshold) or \
                           (pos < 0 and z <= self.config.exit_z_threshold)
                time_stop = days_held > self.config.max_holding_days
                
                if stop_hit or exit_hit or time_stop:
                    pos = 0
                    entry_z = None
                    entry_i = None
            positions.append(pos)
        
        return pd.Series(positions, index=df.index, dtype=float)
```
