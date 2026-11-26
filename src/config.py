"""Load project settings from configs/config.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache
def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read the YAML file once and turn every entry under `paths` into a full Path."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["paths"] = {name: PROJECT_ROOT / p for name, p in config["paths"].items()}
    return config
