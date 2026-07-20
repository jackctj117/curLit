"""Instrument extraction for research-gate notifications (CL-frn7).

Every operator approval shown in Telegram must say WHICH tickers /
instruments the proposed plan would trade. Two sources:

  * **Candidate strategy code** (GATE 2) — the Implementer writes
    ``src/strategies/_experimental/{slug}.py`` whose strategy class
    carries a ``symbols`` list and optionally ``execution_symbol`` /
    ``execution_symbols``. The canonical read logic lives in
    ``src.research.backtest_runner`` (``_read_symbols`` /
    ``_read_execution_symbol``); we import and wrap it rather than
    duplicating so notification text can never disagree with what the
    backtest actually traded.
  * **Hypothesis briefs** (GATE 1) — ``docs/research/hypotheses/
    {slug}.md`` has a ``## Data requirements`` section listing series
    identifiers (``prices.{SYM}``, bare FX pairs like ``EURUSD``,
    ``POLY:slug`` prediction-market ids, FRED series ids). FX pairs and
    POLY entries are tradable; macro series (FRED ids) are inputs — the
    two are returned separately so the notification can render
    ``Trades:`` and ``Inputs:`` lines.

Everything here is fail-safe: extraction problems log a warning and
return empty results. A notification with ``Trades: (unknown)`` beats a
research loop killed by a parse error.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Candidate strategy code (GATE 2)
# --------------------------------------------------------------------- #


def extract_candidate_instruments(code_path: Path | str) -> list[str]:
    """Symbols a candidate strategy file would trade.

    Wraps the backtest runner's canonical extraction (import module →
    find strategy class → read ``symbols`` + execution symbol(s)).
    Execution symbols come first — they're the ones P&L is computed on;
    the rest of ``symbols`` are cross-symbol signal inputs.

    Fail-safe: any problem (missing file, import error, no class) logs
    a warning and returns ``[]``.
    """
    # Imported lazily: backtest_runner pulls in pandas + the backtest
    # stack, which callers like the Telegram bot don't otherwise need
    # at import time.
    from src.research.backtest_runner import (
        _find_strategy_class,
        _import_strategy_module,
        _read_execution_symbol,
        _read_symbols,
    )

    path = Path(code_path)
    module = None
    try:
        module = _import_strategy_module(path)
        strategy_cls = _find_strategy_class(module)
        symbols = _read_symbols(strategy_cls)
        execution: list[str] = []
        # Plural form first (some candidates declare a basket), then
        # the canonical singular via the backtest runner's helper.
        plural = getattr(strategy_cls, "execution_symbols", None)
        if isinstance(plural, (list, tuple)):
            execution = [str(s) for s in plural]
        else:
            execution = [_read_execution_symbol(strategy_cls, symbols)]
        ordered = execution + symbols
        # Dedup preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for sym in ordered:
            s = str(sym)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out
    except Exception as exc:
        logger.warning(
            "could not extract instruments from candidate %s: %s: %s",
            path, type(exc).__name__, exc,
        )
        return []
    finally:
        # _import_strategy_module registers a unique module name per
        # call; drop it so repeated 'pending' listings don't accumulate
        # dead modules in sys.modules.
        if module is not None:
            sys.modules.pop(module.__name__, None)


# --------------------------------------------------------------------- #
# Hypothesis briefs (GATE 1)
# --------------------------------------------------------------------- #


# Currency codes accepted in 6-letter FX-pair tokens (both halves must
# match). G10 + the liquid crosses the ingest layer knows about.
_CCY_CODES = frozenset({
    "USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF",
    "NOK", "SEK", "DKK", "CNH", "CNY", "MXN", "ZAR", "PLN",
    "HUF", "TRY", "SGD", "HKD", "KRW", "INR", "BRL",
})

_FX_PAIR_RE = re.compile(r"\b([A-Z]{6})\b")
_POLY_RE = re.compile(r"\bPOLY:[A-Za-z0-9_-]+")
# Single backticked tokens like `DGS10`, `VIXCLS`, `T10Y2Y` — FRED-ish
# series ids: uppercase alnum, letter-first, no spaces/lowercase.
_BACKTICK_TOKEN_RE = re.compile(r"`([A-Z][A-Z0-9]{2,11})`")
# Words that match the series-id shape but are prose, not data series.
_INPUT_STOPWORDS = frozenset({
    "FRED", "POLY", "NOT", "YET", "INGESTED", "OHLCV", "CSV",
    "JSON", "API", "OOS", "TODO", "NONE",
}) | _CCY_CODES

_SECTION_RE = re.compile(
    r"^##\s+Data requirements\s*$(.*?)(?=^##\s|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _is_fx_pair(token: str) -> bool:
    return token[:3] in _CCY_CODES and token[3:] in _CCY_CODES


def extract_brief_instruments(
    brief_path: Path | str,
) -> tuple[list[str], list[str]]:
    """``(tradable, inputs)`` series identifiers from a hypothesis brief.

    Parses the ``## Data requirements`` section (whole document as a
    fallback if the section is missing):

      * FX pairs (``EURUSD`` — 6 letters, both halves valid currency
        codes) and ``POLY:slug`` entries → **tradable**
      * backticked FRED-style series ids (``DGS10``, ``VIXCLS``) that
        aren't FX pairs → **inputs**

    Fail-safe: unreadable/missing brief logs a warning and returns
    ``([], [])``.
    """
    path = Path(brief_path)
    try:
        text = path.read_text()
    except OSError as exc:
        logger.warning(
            "could not read hypothesis brief %s: %s: %s",
            path, type(exc).__name__, exc,
        )
        return [], []
    section_match = _SECTION_RE.search(text)
    if section_match:
        section = section_match.group(1)
    else:
        logger.warning(
            "brief %s has no '## Data requirements' section; scanning "
            "whole document for instruments", path,
        )
        section = text

    tradable: list[str] = []
    inputs: list[str] = []
    seen: set[str] = set()

    def add(bucket: list[str], token: str) -> None:
        if token not in seen:
            seen.add(token)
            bucket.append(token)

    for match in _POLY_RE.finditer(section):
        add(tradable, match.group(0))
    for match in _FX_PAIR_RE.finditer(section):
        token = match.group(1)
        if _is_fx_pair(token):
            add(tradable, token)
    for match in _BACKTICK_TOKEN_RE.finditer(section):
        token = match.group(1)
        if _is_fx_pair(token) or token in _INPUT_STOPWORDS:
            continue
        add(inputs, token)
    return tradable, inputs
