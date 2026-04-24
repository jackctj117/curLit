"""Bootstrap confidence intervals for performance metrics."""

import numpy as np
import pandas as pd


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    n = len(returns)
    sharpes = []
    for _ in range(n_bootstrap):
        sample = returns.sample(n, replace=True)
        if sample.std() > 0:
            sharpes.append(
                sample.mean() / sample.std() * np.sqrt(periods_per_year),
            )
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(sharpes, alpha)),
        float(np.quantile(sharpes, 1 - alpha)),
    )


def stationary_bootstrap(
    returns: pd.Series,
    block_mean_len: int = 20,
    n_bootstrap: int = 10000,
    periods_per_year: int = 252,
) -> np.ndarray:
    n = len(returns)
    p = 1.0 / block_mean_len
    results = np.zeros(n_bootstrap)

    values = returns.values
    for b in range(n_bootstrap):
        indices = []
        i = np.random.randint(n)
        while len(indices) < n:
            indices.append(i)
            if np.random.random() < p:
                i = np.random.randint(n)
            else:
                i = (i + 1) % n
        sample = values[np.array(indices[:n])]
        if np.std(sample) > 0:
            results[b] = np.mean(sample) / np.std(sample) * np.sqrt(periods_per_year)

    return results


def stationary_bootstrap_sharpe_ci(
    returns: pd.Series,
    block_mean_len: int = 20,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    sharpes = stationary_bootstrap(returns, block_mean_len, n_bootstrap, periods_per_year)
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(sharpes, alpha)),
        float(np.quantile(sharpes, 1 - alpha)),
    )
