"""Retail-broker (Robinhood) execution proxies (CL-vowz).

curLit's event pipeline emits trade ideas on OANDA CFDs and FX pairs
(``XAU_USD``, ``BCO_USD``, ``SPX500_USD``, ``USD_JPY``, ...). The
operator executes on Robinhood, which has NO forex, NO CFDs, and NO
futures — only US equities, ETFs, and (with approval) options on them.
A digest idea reading ``XAU_USD — LONG`` or ``BCO_USD — LONG`` is not
executable there.

This module maps each event instrument to the closest Robinhood-tradable
proxy (``configs/retail_proxies.yaml``) so every notification can also
show the version the operator can actually place:

  * commodity / metal CFDs → the tracking ETF (gold → GLD/IAU, Brent →
    BNO, WTI → USO, gas → UNG, ...);
  * equity-index CFDs → the index ETF long, and an INVERSE ETF (or
    "buy puts") for the short/bearish side (SPX500_USD short → SH/SDS);
  * FX pairs → flagged NOT tradable (no clean retail proxy);
  * plain equities (``DHT``, ``TEVA`` — already Robinhood tickers) →
    "trades directly (stock + options)".

Honesty is baked in (surfaced in the caveats): ETF proxies track
imperfectly (USO/BNO/UNG have roll/contango drift), leveraged/inverse
ETFs decay on multi-day holds, and FX is unmappable. Advisory only —
NOT financial advice.

Consumers:
  * :func:`compact_label` — the one-line digest sub-line
    (:mod:`src.events.digest` ``_idea_line``): ``GLD/IAU`` /
    ``SPY→short via SH`` / ``FX — n/a`` / ``trades directly``.
  * :func:`full_detail` — the multi-line ``Robinhood:`` section in the
    ``idea <id>`` card (:mod:`src.research.telegram_approvals`).

Both take the idea's ``instrument`` (its ``ticker``) and its trade
``direction``/``action`` so the short side surfaces the inverse ETFs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_PROXIES_PATH = Path("configs/retail_proxies.yaml")

#: A valid US-listed equity/ETF ticker: 1-5 uppercase letters.
_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

#: An OANDA/CFD/FX-shaped instrument carries an underscore (``XAU_USD``,
#: ``SPX500_USD``, ``USD_JPY``); a plain equity ticker (``DHT``) does not.
#: Used to tell "unknown CFD" from "already a Robinhood equity".
_OANDA_SHAPE_RE = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+$")

#: Trade directions / actions that express a BEARISH (short) view — these
#: surface the inverse ETFs (or "buy puts") instead of the long proxy.
_BEARISH = frozenset({"short", "sell", "bearish", "buy_puts", "puts", "down"})
#: ...and the bullish side, for symmetry / clarity.
_BULLISH = frozenset({"long", "buy", "bullish", "buy_calls", "calls", "up"})


@dataclass(frozen=True)
class RetailProxy:
    """A resolved Robinhood-tradable mapping for one event instrument.

    Exactly one of these describes what the operator can actually place:

      * a proxy set (``proxies`` populated, ``tradable`` True) — a
        commodity/index CFD mapped to its ETF(s);
      * an FX pair (``tradable`` False, ``proxies`` empty) — no clean
        retail proxy;
      * a plain equity (``trades_directly`` True) — the instrument IS a
        Robinhood ticker; buy the stock or its options.

    ``inverse`` holds the short-side ETFs for an index (SH/SDS, PSQ/SQQQ);
    :func:`retail_proxy` copies them into ``active`` when the requested
    direction is bearish so callers do not re-derive the side.
    """

    instrument: str
    proxies: tuple[str, ...] = ()
    inverse: tuple[str, ...] = ()
    note: str = ""
    leveraged_warning: bool = False
    tradable: bool = True
    trades_directly: bool = False
    #: Direction requested (``"long"``/``"short"``/``""``) — drives which
    #: side of an index proxy is surfaced.
    direction: str = ""
    #: The ETFs appropriate to ``direction``: ``inverse`` for a bearish
    #: call on an instrument that has an inverse list, else ``proxies``.
    active: tuple[str, ...] = field(default=())

    @property
    def is_bearish(self) -> bool:
        return self.direction == "short"


def _normalise_direction(direction: str | None) -> str:
    """Fold a raw direction/action token to ``"long"`` / ``"short"`` /
    ``""`` (unknown). Never raises — malformed input yields ``""``."""
    tok = str(direction or "").strip().lower()
    if tok in _BEARISH:
        return "short"
    if tok in _BULLISH:
        return "long"
    return ""


def _clean_ticker_list(raw: object, ctx: str) -> tuple[str, ...]:
    """Validate a YAML ETF list: every entry must be ``[A-Z]{1,5}``.
    Raises on a bad ticker — a silently-dropped proxy would show the
    operator an incomplete (or empty) tradable set (fail-loud, per the
    project convention)."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        msg = f"{ctx}: expected a list of tickers, got {type(raw).__name__}"
        raise ValueError(msg)
    out: list[str] = []
    for item in raw:
        ticker = str(item).strip()
        if not _TICKER_RE.match(ticker):
            msg = f"{ctx}: invalid ETF ticker {item!r} (must match [A-Z]{{1,5}})"
            raise ValueError(msg)
        out.append(ticker)
    return tuple(out)


def _parse_entry(instrument: str, body: object) -> RetailProxy:
    """One YAML instrument body → a base :class:`RetailProxy` (no
    direction applied yet). Raises on structural problems."""
    if not isinstance(body, dict):
        msg = f"retail proxy {instrument!r}: body must be a mapping"
        raise ValueError(msg)
    tradable = bool(body.get("tradable_on_robinhood", True))
    proxies = _clean_ticker_list(body.get("proxies"), f"{instrument}.proxies")
    inverse = _clean_ticker_list(body.get("inverse"), f"{instrument}.inverse")
    if tradable and not proxies:
        # A tradable instrument with no proxies is a config error — the
        # whole point of the entry is to name a tradable proxy.
        msg = f"retail proxy {instrument!r}: tradable entry has no 'proxies'"
        raise ValueError(msg)
    return RetailProxy(
        instrument=instrument,
        proxies=proxies,
        inverse=inverse,
        note=str(body.get("note", "")).strip(),
        leveraged_warning=bool(body.get("leveraged_warning", False)),
        tradable=tradable,
    )


def load_retail_proxies(
    path: Path | str = DEFAULT_PROXIES_PATH,
) -> dict[str, RetailProxy]:
    """Load + validate the retail-proxy config.

    Raises on structural problems (missing file, non-mapping body, an
    ETF ticker that is not ``[A-Z]{1,5}``, a tradable entry with no
    proxies) — a silently-broken map would mis-advise a real trade.
    """
    path = Path(path)
    if not path.exists():
        msg = f"retail proxies config not found: {path}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(path.read_text()) or {}
    instruments = raw.get("instruments")
    if not isinstance(instruments, dict) or not instruments:
        msg = f"{path}: missing or empty 'instruments' mapping"
        raise ValueError(msg)
    return {
        str(key).strip(): _parse_entry(str(key).strip(), body) for key, body in instruments.items()
    }


@lru_cache(maxsize=1)
def _cached_proxies() -> dict[str, RetailProxy]:
    """Process-lifetime cache of the default config. Fail-soft: a load
    error degrades to an empty map (no proxy lines) rather than killing
    the digest — the mapping is a convenience, not a hard dependency."""
    try:
        return load_retail_proxies()
    except Exception:
        logger.warning(
            "retail proxies config unavailable; notifications render without RH proxy lines",
            exc_info=True,
        )
        return {}


def _with_direction(base: RetailProxy, direction: str) -> RetailProxy:
    """Return ``base`` specialised to ``direction`` — a bearish call on
    an instrument WITH an inverse list surfaces the inverse ETFs in
    ``active``; everything else surfaces ``proxies``."""
    active = base.inverse if direction == "short" and base.inverse else base.proxies
    return replace(base, direction=direction, active=active)


def retail_proxy(
    instrument: str,
    direction: str | None = None,
    proxies: dict[str, RetailProxy] | None = None,
) -> RetailProxy | None:
    """Resolve a Robinhood-tradable mapping for ``instrument``.

    * A mapped CFD/index/FX instrument → its :class:`RetailProxy`, with
      the short side (inverse ETFs) surfaced when ``direction`` is
      bearish (``short``/``sell``/``buy_puts``/``bearish``).
    * A plain equity ticker (``[A-Z]{1,5}``, NOT in the config and NOT
      OANDA-shaped — e.g. ``DHT``, ``TEVA``) → a ``trades_directly``
      proxy: it IS a Robinhood ticker.
    * Anything else (unknown OANDA-shaped instrument, empty, junk) →
      ``None`` (None-safe — callers omit the proxy line).

    ``proxies`` overrides the loaded config (tests inject a fixture).
    """
    symbol = str(instrument or "").strip()
    if not symbol:
        return None
    dir_norm = _normalise_direction(direction)
    table = proxies if proxies is not None else _cached_proxies()

    base = table.get(symbol) or table.get(symbol.upper())
    if base is not None:
        return _with_direction(base, dir_norm)

    # Not in the config. A plain equity ticker (no underscore, 1-5 caps)
    # is already a Robinhood symbol — trade it directly.
    upper = symbol.upper()
    if _TICKER_RE.match(upper) and not _OANDA_SHAPE_RE.match(upper):
        return RetailProxy(
            instrument=upper,
            proxies=(upper,),
            note="trades directly on Robinhood (stock + options)",
            tradable=True,
            trades_directly=True,
            direction=dir_norm,
            active=(upper,),
        )

    # Unknown OANDA/CFD-shaped instrument (e.g. a pair we did not map) —
    # None-safe: the caller simply omits the proxy line.
    return None


def _short_instruction(proxy: RetailProxy) -> str:
    """The bearish-side instruction for an index proxy: inverse ETFs when
    listed, else the fallback of buying puts on the long proxy."""
    if proxy.inverse:
        return "/".join(proxy.inverse)
    if proxy.proxies:
        return f"{proxy.proxies[0]} puts"
    return "n/a"


def compact_label(instrument: str, direction: str | None = None) -> str:
    """One compact token for the digest idea sub-line.

    ``GLD/IAU`` (commodity long) · ``SPY→short via SH/SDS`` (index short)
    · ``FX — n/a`` (unmappable pair) · ``trades directly`` (plain equity)
    · ``n/a`` (unknown). NOT HTML-escaped here — the caller escapes it
    (tickers are safe, but the ``→``/``—`` glyphs and any note text ride
    the same escaping path as every interpolated value).
    """
    proxy = retail_proxy(instrument, direction)
    if proxy is None:
        return "n/a"
    if proxy.trades_directly:
        return "trades directly"
    if not proxy.tradable:
        return "FX — n/a"
    if proxy.is_bearish and proxy.inverse:
        long_sym = proxy.proxies[0] if proxy.proxies else instrument
        return f"{long_sym}→short via {'/'.join(proxy.inverse)}"
    if proxy.is_bearish and proxy.proxies:
        # Bearish but no inverse fund (commodity ETF): puts on the proxy.
        return f"{proxy.proxies[0]} puts (or short)"
    return "/".join(proxy.active or proxy.proxies)


def full_detail(instrument: str, direction: str | None = None) -> list[str]:
    """Multi-line ``Robinhood:`` detail for the ``idea <id>`` card.

    Returns a list of plain-text lines (the idea card is plain text, no
    parse_mode — nothing needs HTML escaping). Empty list when there is
    no mapping (unknown instrument → caller omits the section).

    Content, direction-aware:
      * the tradable proxy set (or inverse ETFs for a short);
      * the exact instruction (``LONG → buy GLD or IAU`` /
        ``SHORT SPX500_USD → buy SH/SDS or SPY puts``);
      * the config note, plus the standing caveats (ETF tracking,
        leveraged/inverse decay, options approval) and the advisory
        footer.
    """
    proxy = retail_proxy(instrument, direction)
    if proxy is None:
        return []

    dir_word = {"long": "LONG", "short": "SHORT"}.get(proxy.direction, "")
    lines: list[str] = []

    if not proxy.tradable:
        # FX pair — no clean retail proxy.
        lines.append(f"Robinhood: {instrument} is FX — not tradable on Robinhood.")
        if proxy.note:
            lines.append(f"  {proxy.note}")
        lines.append("  Skip the FX leg, or express the view via a correlated ETF.")
        lines.append("  Advisory — not financial advice.")
        return lines

    if proxy.trades_directly:
        lines.append(
            f"Robinhood: {proxy.instrument} trades directly (buy the stock; "
            f"options with broker approval)."
        )
        act = f"{dir_word} " if dir_word else ""
        if proxy.direction == "short":
            lines.append(
                f"  {act}{proxy.instrument} → short shares (margin) or buy {proxy.instrument} puts."
            )
        elif proxy.direction == "long":
            lines.append(f"  {act}{proxy.instrument} → buy {proxy.instrument} shares or calls.")
        lines.append("  Advisory — not financial advice; confirm strikes/expiries on your broker.")
        return lines

    # A mapped CFD/index/commodity proxy.
    proxy_word = "proxy" if len(proxy.proxies) == 1 else "proxies"
    lines.append(f"Robinhood {proxy_word}: {', '.join(proxy.proxies) or 'n/a'}")
    if proxy.inverse:
        lines.append(f"  Inverse (for shorts): {', '.join(proxy.inverse)}")

    # Direction-appropriate instruction.
    if proxy.direction == "short":
        if proxy.inverse:
            long_or = f" or {proxy.proxies[0]} puts" if proxy.proxies else ""
            lines.append(f"  SHORT {instrument} → buy {'/'.join(proxy.inverse)}{long_or}.")
        elif proxy.proxies:
            lines.append(
                f"  SHORT {instrument} → buy {proxy.proxies[0]} puts (no "
                f"clean inverse ETF; shorting the ETF is the alternative)."
            )
    elif proxy.direction == "long" and proxy.proxies:
        joined = " or ".join(proxy.proxies)
        lines.append(f"  LONG {instrument} → buy {joined}.")
    # Unknown direction: proxies already listed above; no side instruction.

    if proxy.note:
        lines.append(f"  {proxy.note}")

    caveats = [
        "ETF proxies track imperfectly (futures-based funds have roll/contango drift).",
    ]
    if proxy.leveraged_warning:
        caveats.append(
            "Leveraged/inverse ETFs rebalance daily and DECAY on multi-day holds — tactical only.",
        )
    caveats.append(
        "Options need broker approval; no live options/IV data here — "
        "confirm strikes/expiries on your broker.",
    )
    caveats.append("Advisory — not financial advice.")
    lines.extend(f"  {c}" for c in caveats)
    return lines
