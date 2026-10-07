"""Explicit account-currency conversion for event risk and P&L (CL-vfw7).

Why this exists: the event strategy sized ``equity * risk_pct /
stop_distance`` where equity is ACCOUNT currency (USD) but the stop
distance is QUOTE-currency price units per unit. That is only right for
USD-quoted pairs. A 2026-08-03 USD_JPY short was sized 318 units (~150x
too small) and its reported "+44.29" P&L was ¥44 (~$0.28). Every amount
that is compared against account equity (risk per unit, concentration
notionals, realized P&L feeding the loss cap) must first be converted
from the instrument's quote currency to the account currency — with the
rate's provenance recorded, and never with a fabricated 1.0.

Contract:

* :meth:`AccountCurrencyConverter.rate` resolves ``quote -> account``:
  identity; a direct pair (``QUOTE_ACCOUNT`` → mid, or ``ACCOUNT_QUOTE`` →
  1/mid) from FRESH live ticks; else ONE cross via USD (both legs direct
  and fresh). It returns a :class:`Conversion` (rate, source pair,
  observed_at) so callers can persist provenance.
* Anything stale, missing, unparseable, non-positive, crossed, or
  future-dated raises :class:`ConversionUnavailable`. Callers fail CLOSED
  (skip the entry / do not credit the P&L) — there is no default rate.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

#: Maximum age of a tick used as a conversion rate. 15 minutes matches the
#: event strategy's existing intraday price fallback
#: (``get_intraday_value(..., max_staleness_minutes=15)``) and is 7.5x the
#: intraday pricer's 120 s poll, so a healthy stream never trips it during
#: FX hours. A major-pair rate rarely drifts more than ~0.3% in 15 min, so
#: the worst-case sizing error from an in-bound rate is ~0.3% of the 50 bps
#: risk budget — negligible — while weekend/closed-market or dead-stream
#: quotes (hours old) are excluded. Tighter than this would block entries
#: on quiet crosses (USD_MXN overnight) for no material accuracy gain.
MAX_RATE_AGE = timedelta(minutes=15)

#: Tolerated forward clock skew between the venue's tick timestamp and our
#: clock. A tick stamped further in the future than this is nonsense and is
#: refused rather than treated as maximally fresh.
MAX_FUTURE_SKEW = timedelta(seconds=60)

#: Env override for the account currency when the broker Account does not
#: expose one (the OANDA/paper :class:`~src.execution.broker.Account`
#: dataclass currently has no currency field).
ACCOUNT_CURRENCY_ENV = "CURLIT_ACCOUNT_CURRENCY"

#: Default account currency: the OANDA practice account is USD-denominated
#: (docs/CURRENT_OPERATIONS.md §1a, started at $100,000). Logged at boot.
DEFAULT_ACCOUNT_CURRENCY = "USD"

#: The single intermediate currency allowed for a cross rate.
_CROSS_VIA = "USD"

PriceSource = Callable[[], Mapping[str, Any]]


class ConversionUnavailable(Exception):  # noqa: N818 — domain name, not an "Error"
    """No fresh, valid rate exists for the requested conversion. Callers
    must fail closed: block the entry, never substitute a default rate."""


@dataclass(frozen=True)
class Conversion:
    """One resolved ``quote -> account`` rate: 1 unit of ``quote_ccy`` is
    worth ``rate`` units of ``account_ccy``. ``source_pair`` names the
    tick(s) used (``"identity"`` for same currency; ``"A*B"`` for a cross);
    ``observed_at`` is the OLDEST tick timestamp involved."""

    quote_ccy: str
    account_ccy: str
    rate: float
    source_pair: str
    observed_at: datetime

    def __post_init__(self) -> None:
        assert math.isfinite(self.rate) and self.rate > 0, "conversion rate must be finite > 0"
        assert self.observed_at.tzinfo is not None, "observed_at must be tz-aware"

    def to_payload(self) -> dict[str, Any]:
        return {
            "quote_ccy": self.quote_ccy,
            "account_ccy": self.account_ccy,
            "rate": self.rate,
            "source_pair": self.source_pair,
            "observed_at": self.observed_at.isoformat(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Conversion:
        """Parse a persisted conversion; raises ValueError on nonsense so a
        corrupt state file fails loud (CL-74u9 posture)."""
        try:
            observed = datetime.fromisoformat(str(payload["observed_at"]))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=UTC)
            rate = float(payload["rate"])
            quote = _valid_ccy(str(payload["quote_ccy"]))
            account = _valid_ccy(str(payload["account_ccy"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"unparseable conversion record {payload!r}") from exc
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"conversion record has invalid rate {rate!r}")
        return cls(
            quote_ccy=quote,
            account_ccy=account,
            rate=rate,
            source_pair=str(payload.get("source_pair", "")),
            observed_at=observed,
        )


def _valid_ccy(code: str) -> str:
    code = code.strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError(f"not a 3-letter currency code: {code!r}")
    return code


def quote_currency(symbol: str) -> str | None:
    """Quote currency of an instrument in either symbol dialect, or None.

    OANDA ids end in ``_<QUOTE>`` for FX, metals and CFDs alike
    (``USD_JPY`` → JPY, ``XAU_USD`` → USD, ``SPX500_USD`` → USD). A compact
    6-letter pair (``USDJPY``) splits 3/3. Anything else is unknown — the
    caller must treat that as conversion-unavailable, never as USD."""
    s = str(symbol).strip().upper()
    if "_" in s:
        tail = s.rsplit("_", 1)[1]
        return tail if len(tail) == 3 and tail.isalpha() else None
    if len(s) == 6 and s.isalpha():
        return s[3:]
    return None


def parse_tick_ts(tick: Mapping[str, Any]) -> datetime | None:
    """Tick timestamp (``ts``/``time``/``timestamp``; OANDA RFC3339 with a
    ``Z`` suffix or ISO). Same parsing as the risk context's price-age
    check (src/risk/risk_context.py ``_parse_tick_ts``); naive → UTC."""
    raw = tick.get("ts") or tick.get("time") or tick.get("timestamp")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    else:
        text = str(raw).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def risk_sized_units(
    risk_budget_account: float,
    stop_distance_quote: float,
    rate: float,
) -> float:
    """Unsigned units whose loss at the stop equals the risk budget, in
    ACCOUNT currency (CL-vfw7).

    ``stop_distance_quote`` is quote-currency price units per unit;
    ``rate`` converts 1 quote unit to account currency. Post-condition
    (the invariant the pre-fix ``equity*risk/stop_distance`` violated for
    every non-account-quoted pair): ``units * stop_distance_quote * rate
    == risk_budget_account``."""
    assert math.isfinite(risk_budget_account) and risk_budget_account > 0
    assert math.isfinite(stop_distance_quote) and stop_distance_quote > 0
    assert math.isfinite(rate) and rate > 0
    risk_per_unit_account = stop_distance_quote * rate
    units = risk_budget_account / risk_per_unit_account
    assert math.isclose(units * risk_per_unit_account, risk_budget_account, rel_tol=1e-9)
    return units


def resolve_account_currency(account: Any = None) -> str:
    """Account currency: the broker Account's ``currency`` attribute if it
    exposes a valid one, else ``$CURLIT_ACCOUNT_CURRENCY``, else USD.
    Raises ValueError on an invalid explicit value (fail loud, never
    silently fall back past a typo). Logs the resolved source."""
    broker_ccy = getattr(account, "currency", None) if account is not None else None
    if broker_ccy:
        ccy = _valid_ccy(str(broker_ccy))
        logger.info("Account currency %s (from broker account)", ccy)
        return ccy
    env = os.environ.get(ACCOUNT_CURRENCY_ENV, "").strip()
    if env:
        ccy = _valid_ccy(env)
        logger.info("Account currency %s (from $%s)", ccy, ACCOUNT_CURRENCY_ENV)
        return ccy
    logger.warning(
        "Account currency %s ASSUMED (broker account exposes none and $%s unset) "
        "— set %s if the account is not %s-denominated",
        DEFAULT_ACCOUNT_CURRENCY,
        ACCOUNT_CURRENCY_ENV,
        ACCOUNT_CURRENCY_ENV,
        DEFAULT_ACCOUNT_CURRENCY,
    )
    return DEFAULT_ACCOUNT_CURRENCY


class AccountCurrencyConverter:
    """Quote → account currency rates from live ticks (CL-vfw7).

    ``price_source`` returns the current tick mapping (the engine's
    ``_last_prices``: canonical keys like ``"USDJPY"`` → ``{bid, ask, ts}``);
    only ``.get`` lookups are performed on it, so passing the engine's live
    dict is safe from the strategy's worker thread. Tests inject a fake."""

    def __init__(
        self,
        price_source: PriceSource,
        account_currency: str,
        *,
        max_age: timedelta = MAX_RATE_AGE,
    ) -> None:
        assert max_age > timedelta(0), "max_age must be positive"
        self._price_source = price_source
        self.account_currency = _valid_ccy(account_currency)
        self._max_age = max_age

    def rate(self, quote_ccy: str, as_of: datetime) -> Conversion:
        """Resolve ``quote_ccy -> account currency`` at ``as_of``. Raises
        :class:`ConversionUnavailable` when no fresh valid rate exists."""
        assert as_of.tzinfo is not None, "as_of must be tz-aware"
        try:
            quote = _valid_ccy(quote_ccy)
        except ValueError as exc:
            raise ConversionUnavailable(str(exc)) from exc
        account = self.account_currency
        if quote == account:
            return Conversion(quote, account, 1.0, "identity", as_of)
        prices = self._price_source()
        direct = self._direct(prices, quote, account, as_of)
        if direct is not None:
            return direct
        if _CROSS_VIA not in (quote, account):
            leg1 = self._direct(prices, quote, _CROSS_VIA, as_of)
            leg2 = self._direct(prices, _CROSS_VIA, account, as_of)
            if leg1 is not None and leg2 is not None:
                return Conversion(
                    quote,
                    account,
                    leg1.rate * leg2.rate,
                    f"{leg1.source_pair}*{leg2.source_pair}",
                    min(leg1.observed_at, leg2.observed_at),
                )
        logger.warning(
            "currency conversion %s->%s UNAVAILABLE at %s (no fresh direct or "
            "USD-cross tick within %s) — failing closed",
            quote,
            account,
            as_of.isoformat(),
            self._max_age,
        )
        raise ConversionUnavailable(f"no fresh rate for {quote}->{account} at {as_of.isoformat()}")

    def _direct(
        self,
        prices: Mapping[str, Any],
        frm: str,
        to: str,
        as_of: datetime,
    ) -> Conversion | None:
        """``frm -> to`` from the ``FRM_TO`` (mid) or ``TO_FRM`` (1/mid)
        tick, or None if neither is fresh and valid."""
        for base, quote, invert in ((frm, to, False), (to, frm, True)):
            mid_ts = self._fresh_mid(prices, base, quote, as_of)
            if mid_ts is None:
                continue
            mid, observed = mid_ts
            rate = 1.0 / mid if invert else mid
            return Conversion(frm, to, rate, f"{base}_{quote}", observed)
        return None

    def _fresh_mid(
        self,
        prices: Mapping[str, Any],
        base: str,
        quote: str,
        as_of: datetime,
    ) -> tuple[float, datetime] | None:
        tick = prices.get(f"{base}_{quote}")
        if tick is None:
            tick = prices.get(f"{base}{quote}")
        if not isinstance(tick, Mapping):
            return None
        try:
            bid = float(tick["bid"])
            ask = float(tick["ask"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(bid) and math.isfinite(ask)) or bid <= 0 or ask <= 0 or bid > ask:
            logger.debug("conversion tick %s_%s invalid bid=%r ask=%r", base, quote, bid, ask)
            return None
        observed = parse_tick_ts(tick)
        if observed is None:
            logger.debug("conversion tick %s_%s has no timestamp — unusable", base, quote)
            return None
        age = as_of - observed
        if age > self._max_age or age < -MAX_FUTURE_SKEW:
            logger.debug(
                "conversion tick %s_%s stale/future (age %.0fs) — unusable",
                base,
                quote,
                age.total_seconds(),
            )
            return None
        return (bid + ask) / 2.0, observed
