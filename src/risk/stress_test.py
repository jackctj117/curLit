"""Scenario stress tester — run strategies through historical FX crises."""

from datetime import date


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


def run_stress_test(strategies, data_provider) -> dict:
    import pandas as pd
    import numpy as np
    results = {}
    for name, (start, end) in SCENARIOS.items():
        try:
            data = data_provider.get_range(start, end) if hasattr(data_provider, "get_range") else pd.DataFrame()
        except Exception:
            data = pd.DataFrame()
        if data.empty:
            results[name] = {"total_return": 0, "max_drawdown": 0, "worst_day": 0}
            continue
        returns = pd.Series(0.0, index=data.index)
        for strategy in strategies:
            if hasattr(strategy, "generate_signals"):
                sig = strategy.generate_signals(data)
                if sig is not None:
                    ret = sig.shift(1).fillna(0) * (data.get("close", pd.Series(0)).pct_change().fillna(0))
                    returns = returns.add(ret, fill_value=0)
        cum = (1 + returns).prod() - 1
        dd = ((1 + returns).cumprod().div((1 + returns).cumprod().cummax()) - 1).min()
        results[name] = {"total_return": float(cum), "max_drawdown": float(dd), "worst_day": float(returns.min())}
    return results
