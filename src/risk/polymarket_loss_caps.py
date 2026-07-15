"""Per-market and per-UTC-day loss caps for Polymarket (CL-983f).

The Kelly sizer (``polymarket_sizer.py``) caps position SIZE via
``max_market_fraction`` — it says nothing about realized losses. These
caps are the LOSS brake the CL-983f acceptance gate requires before
mainnet: once a market has burned ``per_market_loss_cap_usd`` (realized
+ mark-to-market), no new orders go to that market; once the UTC day's
cumulative loss hits ``per_day_loss_cap_usd``, no new orders go
anywhere. Position-REDUCING orders are always allowed — a loss cap
that blocks the exit is a trap, not a brake.

P&L model:
  * ``record_fill`` books realized P&L on position reductions (weighted
    average cost basis; fees count against realized) and accumulates
    realized into a per-UTC-date bucket.
  * ``record_mark`` updates the mark price; unrealized P&L is
    ``(mark - avg_price) * signed_qty``. The last fill price doubles as
    the initial mark.
  * per-market P&L = lifetime realized + current unrealized for that
    market key. Deliberately lifetime — a market that ate its cap stays
    blocked; it doesn't get a fresh budget at midnight.
  * per-day P&L = realized booked today (UTC) + current TOTAL
    unrealized. Counting all open unrealized against today is the
    conservative choice: open risk blocks new risk.

Breach rule: P&L <= -cap blocks (reaching the cap halts, not just
exceeding it).

Wiring (the choke point): both ``PolymarketBroker`` and
``PolymarketPaperBroker`` call ``check_order_allowed`` at the top of
``place_order`` — a breach raises ``LossCapExceededError`` before anything
is signed or simulated. That mirrors the other pre-trade rejections:
the OMS RejectionHandler classifies the raise (UNKNOWN -> ABORT, no
retry), same as ``PreTradeValidator`` rejections stop an intent before
submit. The paper broker records its own fills so paper runs exercise
the exact same code; the live broker exposes ``.loss_caps`` for the
fill-reconciliation loop to feed.

Market keys are caller-defined strings. The brokers default to the
CLOB token_id; pass a ``market_key_fn`` that maps token_id ->
condition_id to make YES/NO tokens of one market share a bucket.

Caps come from the active risk profile's ``kill_switches`` block
(``configs/risk_profile.yaml``):
  polymarket_per_market_loss_cap_usd   (default 25)
  polymarket_per_day_loss_cap_usd      (default 50)
Defaults are sized against the $100 initial mainnet cap: one market can
burn at most a quarter of the bankroll, one day at most half.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal

logger = logging.getLogger(__name__)


# Defaults sized against the $100 initial mainnet cap (CL-983f).
_DEFAULT_PER_MARKET_LOSS_CAP_USD: Decimal = Decimal("25")
_DEFAULT_PER_DAY_LOSS_CAP_USD: Decimal = Decimal("50")


class LossCapExceededError(RuntimeError):
    """New-order submission refused: a Polymarket loss cap is breached.

    Raised by ``PolymarketLossCapTracker.check_order_allowed`` — i.e. at
    the brokers' ``place_order`` choke point, before anything is signed.
    ``scope`` is "market" (per-market cap) or "day" (daily cap).
    """

    def __init__(
        self,
        market_key: str,
        scope: str,
        pnl_usd: Decimal,
        cap_usd: Decimal,
    ) -> None:
        self.market_key = market_key
        self.scope = scope
        self.pnl_usd = pnl_usd
        self.cap_usd = cap_usd
        super().__init__(
            f"polymarket loss cap breached: scope={scope} "
            f"market={market_key} pnl={pnl_usd} cap={cap_usd} — "
            f"new orders refused",
        )


@dataclass(frozen=True)
class LossCapConfig:
    """Caps in USD (USDC). Positive numbers; breach at pnl <= -cap."""

    per_market_loss_cap_usd: Decimal = _DEFAULT_PER_MARKET_LOSS_CAP_USD
    per_day_loss_cap_usd: Decimal = _DEFAULT_PER_DAY_LOSS_CAP_USD

    def __post_init__(self) -> None:
        if self.per_market_loss_cap_usd <= 0:
            msg = (
                f"per_market_loss_cap_usd must be positive, got "
                f"{self.per_market_loss_cap_usd}"
            )
            raise ValueError(msg)
        if self.per_day_loss_cap_usd <= 0:
            msg = (
                f"per_day_loss_cap_usd must be positive, got "
                f"{self.per_day_loss_cap_usd}"
            )
            raise ValueError(msg)

    @classmethod
    def from_active_profile(cls) -> LossCapConfig:
        """Build from the active risk profile's kill_switches block."""
        from src.risk.risk_profile import load_active_profile

        ks = load_active_profile().kill_switches
        return cls(
            per_market_loss_cap_usd=Decimal(
                str(ks.polymarket_per_market_loss_cap_usd),
            ),
            per_day_loss_cap_usd=Decimal(
                str(ks.polymarket_per_day_loss_cap_usd),
            ),
        )


@dataclass
class _MarketState:
    """Position + P&L accumulator for one market key."""

    qty: Decimal = Decimal("0")        # signed shares (+ = long the token)
    avg_price: Decimal = Decimal("0")  # weighted-average entry
    realized: Decimal = Decimal("0")   # lifetime realized P&L (fees included)
    last_mark: Decimal | None = None   # last observed price

    @property
    def unrealized(self) -> Decimal:
        if self.qty == 0 or self.last_mark is None:
            return Decimal("0")
        return (self.last_mark - self.avg_price) * self.qty


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class PolymarketLossCapTracker:
    """Tracks realized + mark-to-market P&L and gates new orders.

    ``clock`` is injectable so tests can drive the UTC-day rollover.
    Not thread-safe — brokers call it from their (single) order path.
    """

    config: LossCapConfig = field(default_factory=LossCapConfig)
    clock: Callable[[], datetime] = field(default=_utc_now)

    def __post_init__(self) -> None:
        self._markets: dict[str, _MarketState] = {}
        self._realized_by_day: dict[date, Decimal] = defaultdict(
            lambda: Decimal("0"),
        )

    # --- recording ----------------------------------------------------

    def record_fill(
        self,
        market_key: str,
        side: str,
        quantity: Decimal | float,
        price: Decimal | float,
        fee: Decimal | float = Decimal("0"),
    ) -> None:
        """Book a fill: update position, realize P&L on reductions.

        ``side`` is "buy" | "sell"; ``quantity`` is unsigned shares.
        Fees always count against realized (and today's bucket).
        """
        if side not in {"buy", "sell"}:
            msg = f"side must be 'buy' or 'sell', got {side!r}"
            raise ValueError(msg)
        qty = Decimal(str(quantity))
        px = Decimal(str(price))
        fee_d = Decimal(str(fee))
        if qty <= 0:
            msg = f"fill quantity must be positive, got {qty}"
            raise ValueError(msg)

        signed = qty if side == "buy" else -qty
        st = self._markets.setdefault(market_key, _MarketState())

        realized_delta = -fee_d
        if st.qty != 0 and (st.qty > 0) != (signed > 0):
            # Reducing (possibly flipping) — realize on the closed part.
            closed = min(abs(signed), abs(st.qty))
            direction = Decimal("1") if st.qty > 0 else Decimal("-1")
            realized_delta += (px - st.avg_price) * closed * direction

        new_qty = st.qty + signed
        if new_qty == 0:
            st.avg_price = Decimal("0")
        elif st.qty == 0 or (st.qty > 0) == (signed > 0):
            # Opening or adding — weighted-average cost.
            st.avg_price = (
                (st.avg_price * abs(st.qty) + px * abs(signed)) / abs(new_qty)
                if st.qty != 0 else px
            )
        elif (new_qty > 0) != (st.qty > 0):
            # Flipped through zero — remainder opens at the fill price.
            st.avg_price = px
        # else: partial reduction — avg unchanged.

        st.qty = new_qty
        st.realized += realized_delta
        st.last_mark = px
        today = self.clock().date()
        self._realized_by_day[today] += realized_delta

        logger.info(
            "polymarket loss caps: fill %s %s %s @ %s fee=%s -> "
            "qty=%s realized=%s (today=%s)",
            side, qty, market_key, px, fee_d,
            st.qty, st.realized, self._realized_by_day[today],
        )

    def record_mark(self, market_key: str, price: Decimal | float) -> None:
        """Update the mark price used for unrealized P&L."""
        st = self._markets.setdefault(market_key, _MarketState())
        st.last_mark = Decimal(str(price))

    # --- P&L views ----------------------------------------------------

    def market_pnl(self, market_key: str) -> Decimal:
        """Lifetime realized + current unrealized for one market key."""
        st = self._markets.get(market_key)
        if st is None:
            return Decimal("0")
        return st.realized + st.unrealized

    def day_pnl(self) -> Decimal:
        """Realized booked today (UTC) + total current unrealized."""
        realized_today = self._realized_by_day[self.clock().date()]
        unrealized_total = sum(
            (st.unrealized for st in self._markets.values()),
            start=Decimal("0"),
        )
        return realized_today + unrealized_total

    # --- the gate -----------------------------------------------------

    def check_order_allowed(
        self, market_key: str, side: str | None = None,
    ) -> None:
        """Raise ``LossCapExceededError`` if a new order in ``market_key`` is
        blocked by either cap. Returns None when allowed.

        When ``side`` is given and the order REDUCES the tracked
        position (sell against a long, buy against a short), a breach
        is logged but the order is allowed — operators must always be
        able to flatten. (A reduce larger than the position would flip
        it; v1 doesn't split that hair — the flip's open risk is
        bounded by the position just closed.)
        """
        reducing = False
        if side is not None:
            st = self._markets.get(market_key)
            if st is not None and (
                (side == "sell" and st.qty > 0)
                or (side == "buy" and st.qty < 0)
            ):
                reducing = True

        m_pnl = self.market_pnl(market_key)
        m_cap = self.config.per_market_loss_cap_usd
        if m_pnl <= -m_cap:
            if reducing:
                logger.warning(
                    "polymarket loss caps: market %s breached "
                    "(pnl=%s cap=%s) but order reduces position — allowed",
                    market_key, m_pnl, m_cap,
                )
            else:
                logger.warning(
                    "polymarket loss caps: REFUSING order in %s — "
                    "market pnl=%s <= -cap=%s", market_key, m_pnl, m_cap,
                )
                raise LossCapExceededError(market_key, "market", m_pnl, m_cap)

        d_pnl = self.day_pnl()
        d_cap = self.config.per_day_loss_cap_usd
        if d_pnl <= -d_cap:
            if reducing:
                logger.warning(
                    "polymarket loss caps: daily cap breached "
                    "(pnl=%s cap=%s) but order reduces position — allowed",
                    d_pnl, d_cap,
                )
            else:
                logger.warning(
                    "polymarket loss caps: REFUSING order in %s — "
                    "day pnl=%s <= -cap=%s", market_key, d_pnl, d_cap,
                )
                raise LossCapExceededError(market_key, "day", d_pnl, d_cap)
