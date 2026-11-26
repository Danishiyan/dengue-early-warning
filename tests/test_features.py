"""Tests for the feature-building step."""

import pandas as pd

from src.features import add_features, add_missing_weeks


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


def make_raw_with_gap() -> pd.DataFrame:
    """Raw-style rows for two areas; area B has no row for the 5th week."""
    dates = pd.date_range("2020-01-06", periods=20, freq="7D")
    rows = []
    for area in ["A", "B"]:
        for i, date in enumerate(dates):
            if area == "B" and i == 4:
                continue
            rows.append(
                {
                    "year": 2020,
                    "week": i + 1,
                    "start_date": date,
                    "end_date": date + pd.Timedelta(days=6),
                    "district": area,
                    "cases": i,
                }
            )
    return pd.DataFrame(rows)


def test_missing_week_is_added_as_unknown():
    out = add_missing_weeks(make_raw_with_gap())
    area_b = out[out["district"] == "B"]

    assert len(area_b) == 20
    assert area_b["cases"].isna().sum() == 1


def test_lags_never_skip_a_missing_week():
    out = add_features(add_missing_weeks(make_raw_with_gap()).assign(week_of_year=1))

    assert out[["cases", *[f"lag_{k}" for k in range(1, 9)]]].notna().all().all()
    # A keeps weeks 9-20 (12 rows). B loses the 8 weeks whose lags include the gap.
    assert (out["district"] == "A").sum() == 12
    assert (out["district"] == "B").sum() == 7
    for _, rows in out.groupby("district"):
        gaps = rows["start_date"].diff().dropna().dt.days
        assert (gaps == 7).all()
