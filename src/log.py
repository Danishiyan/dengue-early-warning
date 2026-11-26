"""Logging setup shared by every script in the project."""

import logging

from src.config import load_config

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging() -> None:
    """Send log messages to the terminal with a timestamp, level and module name."""
    level = load_config()["logging"]["level"]
    logging.basicConfig(level=level, format=FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
