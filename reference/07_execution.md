# 07 — Execution Layer

Broker abstraction, OANDA implementation, IBKR skeleton, OMS, and paper broker.

## Broker Abstract Base

**Location:** `src/execution/broker.py`
**Purpose:** Common interface so strategies are broker-agnostic.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'


class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    status: str = 'pending'
    submitted_at: datetime = None
    filled_at: datetime = None
    fill_price: float = None
    strategy_id: str = None


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float
    margin_used: float


@dataclass
class Account:
    account_id: str
    balance: float
    equity: float
    margin_used: float
    margin_available: float


@dataclass
class Fill:
    order_id: str
    symbol: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime


class Broker(ABC):
    @abstractmethod
    async def submit_order(self, order: Order) -> str: ...
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool: ...
    
    @abstractmethod
    def get_account(self) -> Account: ...
    
    @abstractmethod
    def get_positions(self) -> list[Position]: ...
    
    @abstractmethod
    def get_price(self, symbol: str) -> tuple[float, float]:
        """Returns (bid, ask)."""
        ...
    
    @abstractmethod
    async def stream_prices(self, symbols: list[str]):
        """Async iterator yielding price updates."""
        ...
```

## OANDA Broker Implementation

**Location:** `src/execution/oanda.py`
**Purpose:** OANDA REST API integration. Production-grade with retry logic.

```python
import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.execution.broker import (Broker, Order, OrderSide, OrderType,
                                    Position, Account, Fill)

logger = logging.getLogger(__name__)


class OandaBroker(Broker):
    PRACTICE_URL = 'https://api-fxpractice.oanda.com'
    LIVE_URL = 'https://api-fxtrade.oanda.com'
    PRACTICE_STREAM = 'https://stream-fxpractice.oanda.com'
    LIVE_STREAM = 'https://stream-fxtrade.oanda.com'
    
    def __init__(self, api_key: str, account_id: str, practice: bool = True):
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = self.PRACTICE_URL if practice else self.LIVE_URL
        self.stream_url = self.PRACTICE_STREAM if practice else self.LIVE_STREAM
        
        self.client = httpx.AsyncClient(
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
                'Accept-Datetime-Format': 'RFC3339',
            },
            timeout=30,
        )
        
        self._price_cache = {}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def submit_order(self, order: Order) -> str:
        oanda_symbol = self._convert_symbol(order.symbol)
        units = int(order.quantity if order.side == OrderSide.BUY 
                    else -order.quantity)
        
        order_data = {
            'order': {
                'units': str(units),
                'instrument': oanda_symbol,
                'timeInForce': 'FOK',
                'type': order.order_type.value.upper(),
                'positionFill': 'DEFAULT',
            }
        }
        
        if order.order_type == OrderType.LIMIT:
            order_data['order']['price'] = str(order.limit_price)
        
        url = f'{self.base_url}/v3/accounts/{self.account_id}/orders'
        resp = await self.client.post(url, json=order_data)
        
        if resp.status_code >= 400:
            logger.error(f"OANDA order rejected: {resp.text}")
            raise httpx.HTTPStatusError(
                f"OANDA error: {resp.text}",
                request=resp.request, response=resp
            )
        
        result = resp.json()
        if 'orderFillTransaction' in result:
            return result['orderFillTransaction']['id']
        elif 'orderCreateTransaction' in result:
            return result['orderCreateTransaction']['id']
        else:
            raise RuntimeError(f"Unexpected OANDA response: {result}")
    
    async def cancel_order(self, order_id: str) -> bool:
        url = f'{self.base_url}/v3/accounts/{self.account_id}/orders/{order_id}/cancel'
        resp = await self.client.put(url)
        return resp.status_code == 200
    
    def get_account(self) -> Account:
        url = f'{self.base_url}/v3/accounts/{self.account_id}/summary'
        resp = httpx.get(url, headers=self.client.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()['account']
        
        return Account(
            account_id=self.account_id,
            balance=float(data['balance']),
            equity=float(data['NAV']),
            margin_used=float(data['marginUsed']),
            margin_available=float(data['marginAvailable']),
        )
    
    def get_positions(self) -> list[Position]:
        url = f'{self.base_url}/v3/accounts/{self.account_id}/positions'
        resp = httpx.get(url, headers=self.client.headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()['positions']
        
        positions = []
        for p in data:
            long_units = int(p['long']['units'])
            short_units = int(p['short']['units'])
            net = long_units + short_units
            if net == 0:
                continue
            
            symbol = p['instrument'].replace('_', '')
            
            if long_units != 0:
                avg = float(p['long']['averagePrice'])
                upl = float(p['long']['unrealizedPL'])
            else:
                avg = float(p['short']['averagePrice'])
                upl = float(p['short']['unrealizedPL'])
            
            positions.append(Position(
                symbol=symbol,
                quantity=net,
                avg_price=avg,
                unrealized_pnl=upl,
                margin_used=float(p.get('marginUsed', 0)),
            ))
        
        return positions
    
    def get_price(self, symbol: str) -> tuple[float, float]:
        cached = self._price_cache.get(symbol)
        if cached and (datetime.utcnow() - cached['ts']).total_seconds() < 1:
            return cached['bid'], cached['ask']
        
        oanda_symbol = self._convert_symbol(symbol)
        url = f'{self.base_url}/v3/accounts/{self.account_id}/pricing'
        resp = httpx.get(url, params={'instruments': oanda_symbol},
                         headers=self.client.headers, timeout=10)
        resp.raise_for_status()
        prices = resp.json()['prices']
        if not prices:
            raise ValueError(f"No price for {symbol}")
        
        p = prices[0]
        bid = float(p['bids'][0]['price'])
        ask = float(p['asks'][0]['price'])
        self._price_cache[symbol] = {
            'bid': bid, 'ask': ask, 'ts': datetime.utcnow()
        }
        return bid, ask
    
    async def stream_prices(self, symbols: list[str]) -> AsyncIterator[dict]:
        instruments = ','.join(self._convert_symbol(s) for s in symbols)
        url = f'{self.stream_url}/v3/accounts/{self.account_id}/pricing/stream'
        
        async with self.client.stream(
            'GET', url, params={'instruments': instruments}
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    import json
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if data.get('type') == 'PRICE':
                    symbol = data['instrument'].replace('_', '')
                    yield {
                        'symbol': symbol,
                        'bid': float(data['bids'][0]['price']),
                        'ask': float(data['asks'][0]['price']),
                        'ts': datetime.fromisoformat(data['time'].replace('Z', '+00:00')),
                    }
                elif data.get('type') == 'HEARTBEAT':
                    pass
    
    def _convert_symbol(self, symbol: str) -> str:
        """EURUSD → EUR_USD"""
        if '_' in symbol:
            return symbol
        return f'{symbol[:3]}_{symbol[3:]}'
```

## IBKR Broker Skeleton

**Location:** `src/execution/ibkr.py`
**Purpose:** Interactive Brokers via ib_insync. Used for institutional-grade execution.

```python
from datetime import datetime
import logging

from src.execution.broker import (Broker, Order, OrderSide, OrderType,
                                    Position, Account, Fill)

logger = logging.getLogger(__name__)


class IBKRBroker(Broker):
    """Interactive Brokers via ib_insync."""
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, 
                  client_id: int = 1):
        from ib_insync import IB
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
    
    async def connect(self):
        await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
    
    async def submit_order(self, order: Order) -> str:
        from ib_insync import Forex, MarketOrder, LimitOrder
        
        contract = Forex(order.symbol)
        action = 'BUY' if order.side == OrderSide.BUY else 'SELL'
        
        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, order.quantity)
        elif order.order_type == OrderType.LIMIT:
            ib_order = LimitOrder(action, order.quantity, order.limit_price)
        else:
            raise NotImplementedError(f"Order type {order.order_type} not supported")
        
        trade = self.ib.placeOrder(contract, ib_order)
        return str(trade.order.orderId)
    
    async def cancel_order(self, order_id: str) -> bool:
        for trade in self.ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self.ib.cancelOrder(trade.order)
                return True
        return False
    
    def get_account(self) -> Account:
        summary = self.ib.accountSummary()
        values = {x.tag: float(x.value) for x in summary 
                   if x.value.replace('.', '').replace('-', '').isdigit()}
        return Account(
            account_id=self.ib.client.accounts[0],
            balance=values.get('CashBalance', 0),
            equity=values.get('NetLiquidation', 0),
            margin_used=values.get('MaintMarginReq', 0),
            margin_available=values.get('AvailableFunds', 0),
        )
    
    def get_positions(self) -> list[Position]:
        positions = []
        for p in self.ib.positions():
            symbol = p.contract.symbol + p.contract.currency
            positions.append(Position(
                symbol=symbol,
                quantity=p.position,
                avg_price=p.avgCost,
                unrealized_pnl=0,  # Computed elsewhere
                margin_used=0,
            ))
        return positions
```

## Order Management System (OMS)

**Location:** `src/execution/oms.py`
**Purpose:** Translates strategy intents into broker orders. Handles reconciliation.

```python
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging
from collections import defaultdict

from src.execution.broker import Broker, Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


@dataclass
class OrderIntent:
    strategy_id: str
    symbol: str
    target_position: float          # Absolute target in base units
    urgency: str = 'normal'         # 'urgent', 'normal', 'passive'
    max_slippage_bps: float = 5.0


class OMS:
    def __init__(self, broker: Broker):
        self.broker = broker
        self.pending_intents: list[OrderIntent] = []
        self._intent_lock = asyncio.Lock()
        self._known_positions: dict[str, float] = {}
        self._strategy_attribution: dict[str, dict[str, float]] = defaultdict(dict)
    
    def submit_intent(self, intent: OrderIntent):
        self.pending_intents.append(intent)
        logger.info(f"Intent submitted: {intent.strategy_id} → {intent.symbol} "
                   f"target={intent.target_position}")
    
    async def reconcile(self):
        async with self._intent_lock:
            if not self.pending_intents:
                return
            
            current_positions = {p.symbol: p.quantity 
                                  for p in self.broker.get_positions()}
            
            target_positions = self._compute_target_positions()
            
            for symbol, target in target_positions.items():
                current = current_positions.get(symbol, 0)
                diff = target - current
                
                if abs(diff) < 1:
                    continue
                
                await self._execute_diff(symbol, diff, target)
            
            self.pending_intents = []
    
    def _compute_target_positions(self) -> dict[str, float]:
        """Sum target positions across all intents per symbol."""
        targets: dict[str, float] = defaultdict(float)
        for intent in self.pending_intents:
            targets[intent.symbol] += intent.target_position
            self._strategy_attribution[intent.symbol][intent.strategy_id] = \
                intent.target_position
        return dict(targets)
    
    async def _execute_diff(self, symbol: str, diff: float, target: float):
        side = OrderSide.BUY if diff > 0 else OrderSide.SELL
        urgency = max((i.urgency for i in self.pending_intents 
                       if i.symbol == symbol),
                      key=lambda u: {'passive': 0, 'normal': 1, 'urgent': 2}[u])
        
        if urgency == 'urgent':
            order = Order(
                order_id='',
                symbol=symbol,
                side=side,
                quantity=abs(diff),
                order_type=OrderType.MARKET,
                strategy_id='oms',
            )
        else:
            bid, ask = self.broker.get_price(symbol)
            limit_price = bid if side == OrderSide.BUY else ask
            order = Order(
                order_id='',
                symbol=symbol,
                side=side,
                quantity=abs(diff),
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
                strategy_id='oms',
            )
        
        order_id = await self.broker.submit_order(order)
        logger.info(f"Order submitted: {symbol} {side.value} {abs(diff)} → {order_id}")
        return order_id
```

## Paper Broker

**Location:** `src/execution/paper_broker.py`
**Purpose:** Simulated broker for testing. Implements full Broker interface against synthetic prices.

```python
from datetime import datetime
import uuid
from collections import defaultdict

from src.execution.broker import (Broker, Order, OrderSide, OrderType,
                                    Position, Account, Fill)


class PaperBroker(Broker):
    def __init__(self, starting_balance: float = 100000):
        self._account = Account(
            account_id='paper',
            balance=starting_balance,
            equity=starting_balance,
            margin_used=0,
            margin_available=starting_balance * 50,  # 50:1 leverage
        )
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._prices: dict[str, dict] = {}
    
    def update_prices(self, symbol: str, bid: float, ask: float):
        self._prices[symbol] = {
            'bid': bid, 'ask': ask, 'ts': datetime.utcnow()
        }
        # Update unrealized P&L
        if symbol in self._positions:
            pos = self._positions[symbol]
            mid = (bid + ask) / 2
            pos.unrealized_pnl = (mid - pos.avg_price) * pos.quantity
    
    async def submit_order(self, order: Order) -> str:
        order.order_id = str(uuid.uuid4())[:8]
        order.submitted_at = datetime.utcnow()
        self._orders[order.order_id] = order
        
        # Immediate fill at current ask/bid
        prices = self._prices.get(order.symbol)
        if not prices:
            order.status = 'rejected'
            return order.order_id
        
        fill_price = prices['ask'] if order.side == OrderSide.BUY else prices['bid']
        order.fill_price = fill_price
        order.filled_at = datetime.utcnow()
        order.status = 'filled'
        
        # Update position
        signed_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        existing = self._positions.get(order.symbol)
        
        if existing is None:
            self._positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=signed_qty,
                avg_price=fill_price,
                unrealized_pnl=0,
                margin_used=abs(signed_qty) * fill_price * 0.02,  # 50:1 = 2%
            )
        else:
            new_qty = existing.quantity + signed_qty
            if new_qty == 0:
                # Closing
                pnl = (fill_price - existing.avg_price) * existing.quantity
                self._account.balance += pnl
                del self._positions[order.symbol]
            elif (existing.quantity > 0) == (signed_qty > 0):
                # Adding
                existing.avg_price = (
                    (existing.avg_price * existing.quantity + 
                     fill_price * signed_qty) / new_qty
                )
                existing.quantity = new_qty
            else:
                # Reducing
                pnl = (fill_price - existing.avg_price) * (-signed_qty)
                self._account.balance += pnl
                existing.quantity = new_qty
        
        self._fills.append(Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=signed_qty,
            price=fill_price,
            commission=abs(signed_qty) * fill_price * 0.0001,  # 1 bp
            timestamp=order.filled_at,
        ))
        
        return order.order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = 'cancelled'
            return True
        return False
    
    def get_account(self) -> Account:
        upl = sum(p.unrealized_pnl for p in self._positions.values())
        self._account.equity = self._account.balance + upl
        self._account.margin_used = sum(p.margin_used for p in self._positions.values())
        self._account.margin_available = self._account.equity * 50 - self._account.margin_used
        return self._account
    
    def get_positions(self) -> list[Position]:
        return list(self._positions.values())
    
    def get_price(self, symbol: str) -> tuple[float, float]:
        if symbol not in self._prices:
            raise ValueError(f"No price for {symbol}")
        p = self._prices[symbol]
        return p['bid'], p['ask']
    
    async def stream_prices(self, symbols: list[str]):
        # In paper broker, prices are pushed in via update_prices
        while True:
            await asyncio.sleep(1)
```
