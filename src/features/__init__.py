"""Feature computation — signal engineering for FX strategies."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    name: str
    dependencies: list[str]
    compute_fn: Callable[..., Any]
    lookback_days: int


class FeatureStore:
    def __init__(self) -> None:
        self.specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        self.specs[spec.name] = spec

    def compute(self, data: dict[str, pd.Series]) -> pd.DataFrame:
        results = {}
        for name, series in data.items():
            if series is not None:
                results[name] = series
        for name in self._topo_sort():
            spec = self.specs[name]
            inputs = {d: results[d] for d in spec.dependencies if d in results}
            if inputs:
                results[name] = spec.compute_fn(**inputs)
        return pd.DataFrame({k: v for k, v in results.items() if isinstance(v, pd.Series)})

    def _topo_sort(self) -> list[str]:
        resolved = set()
        order = []

        def visit(name: str) -> None:
            if name in resolved:
                return
            spec = self.specs.get(name)
            if spec is None:
                return
            for dep in spec.dependencies:
                if dep not in resolved and dep in self.specs:
                    visit(dep)
            resolved.add(name)
            order.append(name)

        for name in self.specs:
            visit(name)
        return order


# -- individual feature functions -------------------------------------


def rate_differential_zscore(
    us_2y: pd.Series, de_2y: pd.Series, window: int = 252,
) -> pd.Series:
    diff = us_2y - de_2y
    roll_mean = diff.rolling(window).mean()
    roll_std = diff.rolling(window).std()
    return (diff - roll_mean) / roll_std


def realized_vol(price: pd.Series, window: int = 20) -> pd.Series:
    log_ret = np.log(price / price.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def cot_zscore(net_position: pd.Series, window: int = 156) -> pd.Series:
    roll_mean = net_position.rolling(window).mean()
    roll_std = net_position.rolling(window).std()
    return (net_position - roll_mean) / roll_std


def carry(target_rate: pd.Series, base_rate: pd.Series) -> pd.Series:
    return target_rate - base_rate


def momentum_signal(price: pd.Series, lookback: int = 63) -> pd.Series:
    return price / price.shift(lookback) - 1.0


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    return (series - roll_mean) / roll_std


# -- registry factory --------------------------------------------------

def make_default_store() -> FeatureStore:
    store = FeatureStore()
    store.register(FeatureSpec("rate_diff_zscore", ["us_2y", "de_2y"], rate_differential_zscore, 252))
    store.register(FeatureSpec("realized_vol", ["price"], realized_vol, 20))
    store.register(FeatureSpec("cot_zscore", ["net_position"], cot_zscore, 156))
    store.register(FeatureSpec("carry", ["target_rate", "base_rate"], carry, 1))
    store.register(FeatureSpec("momentum", ["price"], momentum_signal, 63))
    return store
