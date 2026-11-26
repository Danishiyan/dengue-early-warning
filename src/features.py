"""Clean the weekly dengue data and build model features."""

import logging

import numpy as np
import pandas as pd

from src.config import load_config
from src.log import setup_logging

logger = logging.getLogger(__name__)

CONFIG = load_config()
RAW_CSV = CONFIG["paths"]["raw_csv"]
FEATURES_CSV = CONFIG["paths"]["features_csv"]

START_YEAR = CONFIG["data"]["start_year"]
RECENT_WEEKS = CONFIG["data"]["recent_weeks"]
LAGS = range(1, CONFIG["data"]["max_lag"] + 1)


def add_missing_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Give every area a row for every reporting week.

    lag and target columns use shift(), which counts rows, not weeks. If an area has no
    row for a week, its lags would silently use the wrong weeks. A missing week gets
    cases = NaN, so the rows that depend on it are dropped later instead of being wrong.
    """
    weeks = df[["year", "week", "start_date", "end_date"]].drop_duplicates("start_date")
    areas = df[["district"]].drop_duplicates()
    full = areas.merge(weeks, how="cross")
    out = full.merge(
        df[["district", "start_date", "cases"]], on=["district", "start_date"], how="left"
    )

    missing = out[out["cases"].isna()]
    for _, row in missing.iterrows():
        logger.warning(
            "Missing week: %s, %s (week %d of %d)",
            row["district"],
            row["start_date"].date(),
            row["week"],
            row["year"],
        )
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions from the EDA notebook."""
    df = add_missing_weeks(df[df["year"] >= START_YEAR])
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
    # Drop rows whose own week or any lag week is missing (see add_missing_weeks).
    return df.dropna(subset=["cases", *[f"lag_{k}" for k in LAGS]]).reset_index(drop=True)


if __name__ == "__main__":
    setup_logging()
    raw = pd.read_csv(RAW_CSV, parse_dates=["start_date", "end_date"])
    data = add_features(clean(raw))

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(FEATURES_CSV, index=False)
    logger.info("Saved %s: %d rows, %d columns", FEATURES_CSV.name, len(data), data.shape[1])

    cols = ["start_date", "cases", "lag_1", "lag_2", "rolling_mean_4", "target"]
    logger.info(
        "Colombo, last 3 weeks:\n%s",
        data.loc[data["district"] == "Colombo", cols].tail(3).to_string(index=False),
    )
