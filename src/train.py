"""Train a gradient boosting model and compare it with the best baseline."""

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.baselines import REPORTS_DIR, mae, time_series_folds
from src.config import PROJECT_ROOT, load_config
from src.features import FEATURES_CSV, LAGS

CONFIG = load_config()
MODEL_PATH = CONFIG["paths"]["model"]
FEATURES = [
    "cases",
    *[f"lag_{k}" for k in LAGS],
    "rolling_mean_4",
    "week_sin",
    "week_cos",
    "district",
]


def load_training_data() -> pd.DataFrame:
    """Rows we can learn from: known target, not in the untrusted recent weeks."""
    df = pd.read_csv(FEATURES_CSV, parse_dates=["start_date"])
    df = df[~df["is_recent"]].dropna(subset=["target"])
    df["district"] = df["district"].astype("category")
    return df


def make_model() -> HistGradientBoostingRegressor:
    """Gradient boosting that optimises absolute error (the metric we report)."""
    return HistGradientBoostingRegressor(
        loss=CONFIG["model"]["loss"],
        categorical_features="from_dtype",
        random_state=CONFIG["model"]["random_state"],
    )


def cross_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Train on earlier years, test on each year; compare with 'same as last week'."""
    rows = []
    for year, train, test in time_series_folds(df):
        model = make_model().fit(train[FEATURES], train["target"])
        rows.append(
            {
                "test_year": year,
                "baseline_last_week": mae(test["target"], test["cases"]),
                "model": mae(test["target"], model.predict(test[FEATURES])),
            }
        )
    return pd.DataFrame(rows).set_index("test_year").round(1)


if __name__ == "__main__":
    data = load_training_data()
    results = cross_validate(data)
    print("MAE (cases per area per week), lower is better:\n")
    print(results.to_string())

    avg = results.mean()
    gain = (1 - avg["model"] / avg["baseline_last_week"]) * 100
    print(f"\nAverage: baseline {avg['baseline_last_week']:.1f} | model {avg['model']:.1f}")
    print(f"Model error is {gain:.1f}% lower than 'same as last week'")
    results.to_csv(REPORTS_DIR / "model_results.csv")

    final_model = make_model().fit(data[FEATURES], data["target"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH.relative_to(PROJECT_ROOT)}")
