"""Tests for the feature-building step."""

import pandas as pd

from src.features import add_features


def make_data() -> pd.DataFrame:
    """Two areas, 10 weeks each, with easy-to-check case numbers."""
    dates = pd.date_range("2020-01-06", periods=10, freq="7D")
    rows = []
    for area, base in [("A", 0), ("B", 100)]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "district": area,
                    "start_date": date,
                    "week_of_year": i + 1,
                    "cases": base + i,
                }
            )
    return pd.DataFrame(rows)


def test_lag_and_target_values():
    out = add_features(make_data())
    row = out[out["district"] == "A"].iloc[0]

    assert row["cases"] == 8
    assert row["lag_1"] == 7
    assert row["lag_8"] == 0
    assert row["target"] == 9


def test_lags_never_mix_areas():
    out = add_features(make_data())
    area_b = out[out["district"] == "B"]

    assert (area_b["lag_8"] >= 100).all()
