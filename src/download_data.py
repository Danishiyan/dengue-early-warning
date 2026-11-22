"""Download the raw Sri Lanka weekly dengue data.

Source: denguedatahub R package by Thiyanga Talagala (GPL-3)
https://github.com/thiyangt/denguedatahub
"""

from pathlib import Path

import pandas as pd
import pyreadr
import requests

RAW_URL = (
    "https://raw.githubusercontent.com/thiyangt/denguedatahub/main/data/srilanka_weekly_data.rda"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RDA_PATH = PROJECT_ROOT / "data" / "raw" / "srilanka_weekly_data.rda"
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "dengue_weekly.csv"


def download_rda(url: str = RAW_URL, dest: Path = RDA_PATH) -> Path:
    """Download the .rda file once; skip if it is already there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already downloaded: {dest.name}")
        return dest

    print(f"Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Saved {dest.name} ({len(response.content) / 1024:.0f} KB)")
    return dest


def rda_to_csv(rda_path: Path = RDA_PATH, csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Convert the R data file into a tidy CSV that pandas can read."""
    df = pyreadr.read_r(rda_path)["srilanka_weekly_data"]

    df = df.rename(columns={"start.date": "start_date", "end.date": "end_date"})
    for col in ["start_date", "end_date"]:
        df[col] = pd.to_datetime(df[col], format="mixed")
    df[["year", "week", "cases"]] = df[["year", "week", "cases"]].astype(int)

    df = df.sort_values(["district", "start_date"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path.name}: {len(df):,} rows, {df['district'].nunique()} areas")
    return df


if __name__ == "__main__":
    download_rda()
    rda_to_csv()
