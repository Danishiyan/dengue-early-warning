"""Clean the weekly dengue data and build model features."""

import numpy as np
import pandas as pd

from src.config import load_config

CONFIG = load_config()
RAW_CSV = CONFIG["paths"]["raw_csv"]
FEATURES_CSV = CONFIG["paths"]["features_csv"]

START_YEAR = CONFIG["data"]["start_year"]
RECENT_WEEKS = CONFIG["data"]["recent_weeks"]
LAGS = range(1, CONFIG["data"]["max_lag"] + 1)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions from the EDA notebook."""
    df = df[df["year"] >= START_YEAR].copy()
    df["week_of_year"] = df["week"].clip(upper=52)

    last_dates = sorted(df["start_date"].unique())[-RECENT_WEEKS:]
    df["is_recent"] = df["start_date"].isin(last_dates)

    return df.sort_values(["district", "start_date"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, seasonal and target columns for each area."""
    df = df.copy()
    by_area = df.groupby("district")["cases"]

    for k in LAGS:
        df[f"lag_{k}"] = by_area.shift(k)
    df["rolling_mean_4"] = df[["lag_1", "lag_2", "lag_3", "lag_4"]].mean(axis=1)

    angle = 2 * np.pi * df["week_of_year"] / 52
    df["week_sin"] = np.sin(angle)
    df["week_cos"] = np.cos(angle)

    df["target"] = by_area.shift(-1)
    return df.dropna(subset=[f"lag_{k}" for k in LAGS]).reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(RAW_CSV, parse_dates=["start_date", "end_date"])
    data = add_features(clean(raw))

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {FEATURES_CSV.name}: {len(data):,} rows, {data.shape[1]} columns")

    cols = ["start_date", "cases", "lag_1", "lag_2", "rolling_mean_4", "target"]
    print(data.loc[data["district"] == "Colombo", cols].tail(3).to_string(index=False))
