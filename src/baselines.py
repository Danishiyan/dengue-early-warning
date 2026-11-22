"""Simple forecasts that every ML model must beat."""

import pandas as pd

from src.features import FEATURES_CSV

TEST_YEARS = range(2021, 2026)


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Add one prediction column per baseline."""
    df = df.copy()
    by_area = df.groupby("district")["cases"]
    df["pred_last_week"] = df["cases"]
    df["pred_mean_4"] = by_area.transform(lambda s: s.rolling(4).mean())
    df["pred_last_year"] = by_area.shift(51)
    return df


def time_series_folds(df: pd.DataFrame, test_years=TEST_YEARS):
    """Yield (year, train, test): train on all earlier years, test on one year."""
    for year in test_years:
        yield year, df[df["year"] < year], df[df["year"] == year]


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error: average size of the mistake, in cases."""
    return (y_true - y_pred).abs().mean()


if __name__ == "__main__":
    data = pd.read_csv(FEATURES_CSV, parse_dates=["start_date"])
    data = add_baselines(data)
    data = data[~data["is_recent"]].dropna(subset=["target", "pred_last_year"])

    baselines = ["pred_last_week", "pred_mean_4", "pred_last_year"]
    rows = []
    for year, _train, test in time_series_folds(data):
        row = {"test_year": year}
        for name in baselines:
            row[name] = mae(test["target"], test[name])
        rows.append(row)

    results = pd.DataFrame(rows).set_index("test_year").round(1)
    print("MAE (cases per area per week), lower is better:\n")
    print(results.to_string())
    print("\nAverage:", results.mean().round(1).to_dict())
