"""Per-market and per-UTC-day loss caps for Polymarket (CL-983f).

The Kelly sizer (``polymarket_sizer.py``) caps position SIZE via
``max_market_fraction`` — it says nothing about realized losses. These
caps are the LOSS brake the CL-983f acceptance gate requires before
mainnet: once a market has burned ``per_market_loss_cap_usd`` (realized
+ mark-to-market), no new orders go to that market; once the UTC day's
cumulative loss hits ``per_day_loss_cap_usd``, no new orders go
anywhere. Position-REDUCING orders are still allowed — a loss cap that
blocks the exit is a trap, not a brake — but only up to the size of the
position being reduced (see ``check_order_allowed``).

P&L model:
  * Position books are kept PER TOKEN (CLOB token_id). ``record_fill``
    books realized P&L on position reductions (weighted average cost
    basis; fees count against realized) and accumulates realized into a
    per-UTC-date bucket.
  * ``record_mark`` updates the token's mark price; unrealized P&L is
    ``(mark - avg_price) * signed_qty``. The last fill price doubles as
    the initial mark.
  * ``market_key_fn`` (token_id -> market key, default identity) is used
    for CAP AGGREGATION ONLY — never for position bookkeeping. Pass a
    condition_id resolver to make the YES/NO tokens of one market share
    a cap bucket: per-market P&L is then the SUM of each member token's
    realized + unrealized, so a hedged YES 0.7 / NO 0.3 pair nets to ~0
    instead of corrupting a pooled qty/avg_price book.
  * per-market P&L = sum over the market's tokens of lifetime realized
    + current unrealized. Deliberately lifetime — a market that ate its
    cap stays blocked; it doesn't get a fresh budget at midnight.
  * per-day P&L = realized booked today (UTC) + current TOTAL
    unrealized. Counting all open unrealized against today is the
    conservative choice: open risk blocks new risk.

Breach rule: P&L <= -cap blocks (reaching the cap halts, not just
exceeding it).

Persistence: when ``LossCapConfig.state_path`` is set (default
``data/polymarket_loss_caps_state.json``), the tracker loads its state
at construction and saves after every mutating call (atomic
tmp+rename, stdlib json). A process restart therefore does NOT reset
the caps: a burned market stays blocked and a part-burned day keeps its
running total. Processed fill ids (idempotency keys) persist too, so
re-reconciliation after a restart cannot double-count. A corrupt state
file raises at construction — a loss brake must not silently reset.
Pass ``state_path=None`` for a purely in-memory tracker (unit tests,
paper runs whose bankroll resets anyway).

Wiring (the choke point): both ``PolymarketBroker`` and
``PolymarketPaperBroker`` call ``check_order_allowed`` at the top of
``place_order`` — a breach raises ``LossCapExceededError`` before anything
is signed or simulated. That mirrors the other pre-trade rejections:
the OMS RejectionHandler classifies the raise (UNKNOWN -> ABORT, no
retry), same as ``PreTradeValidator`` rejections stop an intent before
submit. The paper broker records its own fills and marks; the live
broker records immediate CLOB matches itself and exposes
``ingest_reconciled_fills`` + ``.loss_caps.record_mark`` for the
runtime fill/price loop (see ``polymarket_broker.py`` for the runtime
contract).

All recording/gating calls take the raw token_id; the tracker applies
``market_key_fn`` internally where caps are evaluated.

Caps come from the active risk profile's ``kill_switches`` block
(``configs/risk_profile.yaml``):
  polymarket_per_market_loss_cap_usd   (default 25)
  polymarket_per_day_loss_cap_usd      (default 50)
Defaults are sized against the $100 initial mainnet cap: one market can
burn at most a quarter of the bankroll, one day at most half.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

logger = logging.getLogger(__name__)


# Defaults sized against the $100 initial mainnet cap (CL-983f).
_DEFAULT_PER_MARKET_LOSS_CAP_USD: Decimal = Decimal("25")
_DEFAULT_PER_DAY_LOSS_CAP_USD: Decimal = Decimal("50")

# Default persistence location. Lives under the gitignored ``data/``
# runtime tree next to the other engine state.
_DEFAULT_STATE_PATH: Path = Path("data/polymarket_loss_caps_state.json")

_STATE_VERSION: int = 1


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
    """Caps in USD (USDC). Positive numbers; breach at pnl <= -cap.

    ``state_path`` is where the tracker persists its P&L state (JSON;
    atomic writes). ``None`` disables persistence — used by unit tests
    and the paper broker, whose bankroll resets every run anyway.
    """

    per_market_loss_cap_usd: Decimal = _DEFAULT_PER_MARKET_LOSS_CAP_USD
    per_day_loss_cap_usd: Decimal = _DEFAULT_PER_DAY_LOSS_CAP_USD
    state_path: Path | None = _DEFAULT_STATE_PATH

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
        if self.state_path is not None and not isinstance(self.state_path, Path):
            object.__setattr__(self, "state_path", Path(self.state_path))

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
class TokenState:
    """Position + P&L accumulator for ONE CLOB token.

    Always per-token — market-level pooling happens only when caps are
    evaluated (``market_pnl`` sums member tokens). The paper broker
    reads these as its position book (single source of truth); treat
    instances as read-only outside the tracker.
    """

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


def _identity_key(token_id: str) -> str:
    return token_id


@dataclass
class PolymarketLossCapTracker:
    """Tracks realized + mark-to-market P&L and gates new orders.

    ``clock`` is injectable so tests can drive the UTC-day rollover.
    ``market_key_fn`` maps token_id -> market key for CAP AGGREGATION
    only (default identity; use a condition_id resolver to pool YES/NO
    tokens of one market into one cap bucket).
    Not thread-safe — brokers call it from their (single) order path.
    """

    config: LossCapConfig = field(default_factory=LossCapConfig)
    clock: Callable[[], datetime] = field(default=_utc_now)
    market_key_fn: Callable[[str], str] = field(default=_identity_key)

    def __post_init__(self) -> None:
        self._tokens: dict[str, TokenState] = {}
        self._realized_by_day: dict[date, Decimal] = defaultdict(
            lambda: Decimal("0"),
        )
        # Idempotency keys of externally-sourced fills already booked
        # (immediate CLOB matches + reconciled on-chain fills). Persisted
        # so a restart + re-reconcile cannot double-count.
        self._processed_fill_ids: set[str] = set()
        self._load()

    # --- recording ----------------------------------------------------

    def record_fill(
        self,
        token_id: str,
        side: str,
        quantity: Decimal | float,
        price: Decimal | float,
        fee: Decimal | float = Decimal("0"),
        fill_id: str | None = None,
    ) -> bool:
        """Book a fill: update the token's position, realize P&L on
        reductions. Returns True when booked.

        ``side`` is "buy" | "sell"; ``quantity`` is unsigned shares.
        Fees always count against realized (and today's bucket).
        ``fill_id`` is an optional idempotency key: a fill whose id was
        already recorded is skipped (returns False), which makes
        re-ingesting reconciled fills safe. The key set persists with
        the rest of the state.
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
        if fill_id is not None and fill_id in self._processed_fill_ids:
            logger.info(
                "polymarket loss caps: fill %s already recorded — skipped",
                fill_id,
            )
            return False

        signed = qty if side == "buy" else -qty
        st = self._tokens.setdefault(token_id, TokenState())

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
        if fill_id is not None:
            self._processed_fill_ids.add(fill_id)

        logger.info(
            "polymarket loss caps: fill %s %s %s @ %s fee=%s -> "
            "qty=%s realized=%s (today=%s)",
            side, qty, token_id, px, fee_d,
            st.qty, st.realized, self._realized_by_day[today],
        )
        self._save()
        return True

    def record_mark(self, token_id: str, price: Decimal | float) -> None:
        """Update the token's mark price used for unrealized P&L."""
        st = self._tokens.setdefault(token_id, TokenState())
        st.last_mark = Decimal(str(price))
        self._save()

    def has_recorded_fill(self, fill_id: str) -> bool:
        """True if this idempotency key was already booked/registered."""
        return fill_id in self._processed_fill_ids

    def register_processed_fill(self, fill_id: str) -> bool:
        """Mark a fill id as processed WITHOUT booking P&L.

        Used when one booked fill settles across several transactions:
        the aggregate is booked once under the first key and the sibling
        keys are registered here so reconciliation skips them. Returns
        False if the id was already known.
        """
        if fill_id in self._processed_fill_ids:
            return False
        self._processed_fill_ids.add(fill_id)
        self._save()
        return True

    # --- state views --------------------------------------------------

    @property
    def has_activity(self) -> bool:
        """True once ANY fill or mark has been recorded (including state
        loaded from disk). The live broker uses this to warn when it is
        trading with an unfed tracker."""
        return bool(self._tokens) or bool(self._realized_by_day)

    def open_positions(self) -> dict[str, TokenState]:
        """token_id -> TokenState for tokens with a nonzero position.

        The paper broker derives its ``get_positions`` view from this so
        there is exactly ONE position book. Read-only by convention.
        """
        return {tid: st for tid, st in self._tokens.items() if st.qty != 0}

    # --- P&L views ----------------------------------------------------

    def market_pnl(self, market_key: str) -> Decimal:
        """Lifetime realized + current unrealized summed over every
        token that ``market_key_fn`` maps into ``market_key``."""
        total = Decimal("0")
        for token_id, st in self._tokens.items():
            if self.market_key_fn(token_id) == market_key:
                total += st.realized + st.unrealized
        return total

    def day_pnl(self) -> Decimal:
        """Realized booked today (UTC) + total current unrealized."""
        realized_today = self._realized_by_day.get(
            self.clock().date(), Decimal("0"),
        )
        unrealized_total = sum(
            (st.unrealized for st in self._tokens.values()),
            start=Decimal("0"),
        )
        return realized_today + unrealized_total

    # --- the gate -----------------------------------------------------

    def check_order_allowed(
        self,
        token_id: str,
        side: str | None = None,
        quantity: Decimal | float | None = None,
    ) -> None:
        """Raise ``LossCapExceededError`` if a new order in ``token_id``'s
        market is blocked by either cap. Returns None when allowed.

        Reduce-only exemption: when ``side`` (and ``quantity``) are
        given and the order REDUCES the tracked token position (sell
        against a long, buy against a short) WITHOUT flipping it —
        i.e. ``quantity <= |position|`` — a breach is logged but the
        order is allowed: operators must always be able to flatten. An
        opposing order LARGER than the position would flip through zero
        into brand-new risk in a capped market, so the whole order is
        rejected — brokers submit atomic orders, there is no partial
        exemption. The flip's open risk is bounded only by order size,
        not position size, which is exactly what the cap must stop.

        ``quantity=None`` (ops tooling / legacy callers) treats any
        opposing order as reducing; both brokers always pass the order
        quantity.
        """
        reducing = False
        if side is not None:
            st = self._tokens.get(token_id)
            if (
                st is not None
                and st.qty != 0
                and (
                    (side == "sell" and st.qty > 0)
                    or (side == "buy" and st.qty < 0)
                )
            ):
                reducing = (
                    quantity is None
                    or Decimal(str(quantity)) <= abs(st.qty)
                )

        market_key = self.market_key_fn(token_id)
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

    # --- persistence --------------------------------------------------

    def _save(self) -> None:
        """Atomic (tmp+rename) JSON snapshot of all cap state."""
        path = self.config.state_path
        if path is None:
            return
        payload = {
            "version": _STATE_VERSION,
            "tokens": {
                tid: {
                    "qty": str(st.qty),
                    "avg_price": str(st.avg_price),
                    "realized": str(st.realized),
                    "last_mark": (
                        str(st.last_mark) if st.last_mark is not None else None
                    ),
                }
                for tid, st in self._tokens.items()
            },
            "realized_by_day": {
                d.isoformat(): str(v) for d, v in self._realized_by_day.items()
            },
            "processed_fill_ids": sorted(self._processed_fill_ids),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True))
        os.replace(tmp, path)

    def _load(self) -> None:
        """Load persisted state. Missing file -> fresh tracker; a corrupt
        file raises — a silently reset loss brake is worse than a crash
        at boot (operator must repair or remove the file)."""
        path = self.config.state_path
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text())
            if raw.get("version") != _STATE_VERSION:
                msg = f"unsupported state version {raw.get('version')!r}"
                raise ValueError(msg)
            tokens = {
                str(tid): TokenState(
                    qty=Decimal(body["qty"]),
                    avg_price=Decimal(body["avg_price"]),
                    realized=Decimal(body["realized"]),
                    last_mark=(
                        Decimal(body["last_mark"])
                        if body.get("last_mark") is not None else None
                    ),
                )
                for tid, body in (raw.get("tokens") or {}).items()
            }
            by_day = {
                date.fromisoformat(k): Decimal(v)
                for k, v in (raw.get("realized_by_day") or {}).items()
            }
            fill_ids = {str(x) for x in (raw.get("processed_fill_ids") or [])}
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            msg = (
                f"polymarket loss-cap state at {path} is corrupt or "
                f"unreadable ({exc}) — refusing to start with reset caps; "
                f"repair or remove the file"
            )
            raise ValueError(msg) from exc
        self._tokens = tokens
        self._realized_by_day.update(by_day)
        self._processed_fill_ids = fill_ids
        logger.info(
            "polymarket loss caps: loaded state from %s "
            "(%d tokens, %d day buckets, %d processed fills)",
            path, len(tokens), len(by_day), len(fill_ids),
        )
