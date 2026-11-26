"""Download the raw Sri Lanka weekly dengue data.

Source: denguedatahub R package by Thiyanga Talagala (GPL-3)
https://github.com/thiyangt/denguedatahub
"""

import logging
from pathlib import Path

import pandas as pd
import pyreadr
import requests

from src.config import load_config
from src.log import setup_logging

logger = logging.getLogger(__name__)

RAW_URL = (
    "https://raw.githubusercontent.com/thiyangt/denguedatahub/main/data/srilanka_weekly_data.rda"
)
CONFIG = load_config()
RDA_PATH = CONFIG["paths"]["raw_rda"]
CSV_PATH = CONFIG["paths"]["raw_csv"]


def download_rda(url: str = RAW_URL, dest: Path = RDA_PATH) -> Path:
    """Download the .rda file once; skip if it is already there."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest

    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    logger.info("Saved %s (%.0f KB)", dest.name, len(response.content) / 1024)
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
    logger.info("Saved %s: %d rows, %d areas", csv_path.name, len(df), df["district"].nunique())
    return df


if __name__ == "__main__":
    setup_logging()
    download_rda()
    rda_to_csv()
