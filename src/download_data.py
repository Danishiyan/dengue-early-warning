"""Download the raw Sri Lanka weekly dengue data.

Source: denguedatahub R package by Thiyanga Talagala (GPL-3)
https://github.com/thiyangt/denguedatahub
"""
from pathlib import Path

import requests

RAW_URL = (
    "https://raw.githubusercontent.com/thiyangt/denguedatahub/"
    "main/data/srilanka_weekly_data.rda"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RDA_PATH = PROJECT_ROOT / "data" / "raw" / "srilanka_weekly_data.rda"


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


if __name__ == "__main__":
    download_rda()