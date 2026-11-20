"""Clean the weekly dengue data and build model features."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "dengue_weekly.csv"

START_YEAR = 2010
RECENT_WEEKS = 4


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions from the EDA notebook."""
    df = df[df["year"] >= START_YEAR].copy()
    df["week_of_year"] = df["week"].clip(upper=52)

    last_dates = sorted(df["start_date"].unique())[-RECENT_WEEKS:]
    df["is_recent"] = df["start_date"].isin(last_dates)

    return df.sort_values(["district", "start_date"]).reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(RAW_CSV, parse_dates=["start_date", "end_date"])
    data = clean(raw)
    print(f"{len(raw):,} raw rows -> {len(data):,} clean rows")
    print(f"Years: {data['year'].min()}-{data['year'].max()}")
    print(f"Recent (not trusted) rows: {data['is_recent'].sum()}")
