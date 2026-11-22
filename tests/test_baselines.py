"""Tests for baselines and time-series folds."""

import pandas as pd

from src.baselines import add_baselines, mae, time_series_folds


def make_data() -> pd.DataFrame:
    """One area, 60 weeks, cases = week number (0..59)."""
    return pd.DataFrame(
        {
            "district": "A",
            "year": [2020] * 30 + [2021] * 30,
            "cases": range(60),
        }
    )


def test_baseline_predictions():
    out = add_baselines(make_data())
    row = out.iloc[55]

    assert row["pred_last_week"] == 55
    assert row["pred_mean_4"] == (52 + 53 + 54 + 55) / 4
    assert row["pred_last_year"] == 55 - 51


def test_folds_never_train_on_the_future():
    for year, train, test in time_series_folds(make_data(), test_years=[2021]):
        assert train["year"].max() < year
        assert (test["year"] == year).all()


def test_mae():
    assert mae(pd.Series([10, 20]), pd.Series([12, 16])) == 3
