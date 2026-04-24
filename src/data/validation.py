"""Data quality validation — Great Expectations-style checks on ingested data."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_price_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    failures = []
    if "close" in df.columns:
        if df["close"].isnull().any():
            failures.append("null close values")
        if (df["close"] < 0).any():
            failures.append("negative close values")
    if "ts" in df.columns and "symbol" in df.columns:
        dupes = df.duplicated(subset=["ts", "symbol"]).sum()
        if dupes > 0:
            failures.append(f"{dupes} duplicate ts+symbol rows")
    return len(failures) == 0, failures


def validate_macro_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    failures = []
    if df["value"].isnull().any():
        failures.append("null values in macro data")
    if "release_date" in df.columns and "observation_date" in df.columns:
        bad = (pd.to_datetime(df["release_date"]) < pd.to_datetime(df["observation_date"])).sum()
        if bad > 0:
            failures.append(f"{bad} rows with release_date before observation_date")
    return len(failures) == 0, failures


def validate_cot_positioning(df: pd.DataFrame) -> tuple[bool, list[str]]:
    failures = []
    for col in ["longs", "shorts"]:
        if col in df.columns and (df[col] < 0).any():
            failures.append(f"negative {col}")
    return len(failures) == 0, failures


def validate_rate_curves(df: pd.DataFrame) -> tuple[bool, list[str]]:
    failures = []
    if "rate" in df.columns:
        if (df["rate"] < 0).any():
            failures.append("negative rates")
        if (df["rate"] > 0.25).any():
            failures.append("rate exceeds 25%")
    if "tenor_days" in df.columns and (df["tenor_days"] <= 0).any():
        failures.append("non-positive tenor_days")
    return len(failures) == 0, failures


def run_all_validations(
    prices: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    cot: pd.DataFrame | None = None,
    rate_curves: pd.DataFrame | None = None,
) -> dict[str, bool]:
    results = {}
    for name, df, fn in [
        ("prices", prices, validate_price_data),
        ("macro_data", macro, validate_macro_data),
        ("cot_positioning", cot, validate_cot_positioning),
        ("rate_curves", rate_curves, validate_rate_curves),
    ]:
        if df is not None:
            ok, failures = fn(df)
            results[name] = ok
            if not ok:
                logger.warning("Validation failed for %s: %s", name, failures)
    return results
