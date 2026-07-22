"""Scenario stress tester — run strategies through historical FX crises.

Fail-visible policy (CL-qyav): a scenario whose data cannot be fetched,
is empty, or lacks close prices is SKIPPED with a WARNING naming what
was missing — it is never reported as a zero-loss result. If no scenario
at all can be priced the whole stress test raises, because an empty
result would silently understate risk.
"""

import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

SCENARIOS = {
    "gfc_2008": (date(2008, 9, 1), date(2008, 12, 31)),
    "eurocrisis_2011": (date(2011, 7, 1), date(2011, 12, 31)),
    "taper_tantrum_2013": (date(2013, 5, 1), date(2013, 9, 30)),
    "chf_shock_2015": (date(2015, 1, 1), date(2015, 2, 28)),
    "brexit_2016": (date(2016, 6, 1), date(2016, 7, 31)),
    "covid_2020": (date(2020, 2, 15), date(2020, 4, 30)),
    "inflation_shock_2022": (date(2022, 1, 1), date(2022, 6, 30)),
    "svb_2023": (date(2023, 3, 1), date(2023, 3, 31)),
    "yen_intervention_2024": (date(2024, 7, 1), date(2024, 8, 15)),
}


def run_stress_test(
    strategies: list[Any], data_provider: Any,
) -> dict[str, Any]:
    """Replay each crisis scenario through the strategies.

    Returns per-scenario metrics for every scenario that could be
    priced. Unpriceable scenarios (fetch failure, no rows, no ``close``
    column) are omitted from the result and logged as WARNINGs.

    Raises:
        TypeError: ``data_provider`` has no ``get_range`` method.
        RuntimeError: no scenario could be priced at all.
    """
    import pandas as pd

    if not hasattr(data_provider, "get_range"):
        msg = (
            f"stress test: data provider {type(data_provider).__name__!r} "
            "has no get_range(start, end) — cannot price any scenario"
        )
        raise TypeError(msg)

    results: dict[str, dict[str, float]] = {}
    skipped: list[str] = []
    for name, (start, end) in SCENARIOS.items():
        try:
            data = data_provider.get_range(start, end)
        except Exception:
            # Broad by design: provider backends (DB, HTTP, cache) raise
            # heterogeneous errors and one broken scenario window must
            # not kill the rest of the stress run — but it must be loud.
            logger.warning(
                "stress test %s: data fetch failed for %s..%s — "
                "scenario skipped, NOT counted as zero loss",
                name, start, end, exc_info=True,
            )
            skipped.append(name)
            continue
        if data is None or data.empty:
            logger.warning(
                "stress test %s: no price data for %s..%s — "
                "scenario skipped, NOT counted as zero loss",
                name, start, end,
            )
            skipped.append(name)
            continue
        if "close" not in data.columns:
            logger.warning(
                "stress test %s: 'close' column missing for %s..%s "
                "(columns: %s) — scenario skipped, NOT counted as "
                "zero loss",
                name, start, end, list(data.columns),
            )
            skipped.append(name)
            continue
        close = data["close"]
        returns = pd.Series(0.0, index=data.index)
        for strategy in strategies:
            if hasattr(strategy, "generate_signals"):
                sig = strategy.generate_signals(data)
                if sig is not None:
                    ret = sig.shift(1).fillna(0) * close.pct_change().fillna(0)
                    returns = returns.add(ret, fill_value=0)
        cum = (1 + returns).prod() - 1
        dd = ((1 + returns).cumprod().div((1 + returns).cumprod().cummax()) - 1).min()
        results[name] = {"total_return": float(cum), "max_drawdown": float(dd), "worst_day": float(returns.min())}

    if not results:
        msg = (
            "stress test: no scenario could be priced "
            f"(skipped: {', '.join(skipped)}) — refusing to return an "
            "empty result that would understate risk"
        )
        raise RuntimeError(msg)
    return results
