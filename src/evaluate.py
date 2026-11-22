"""Evaluate the model on the latest full year: error per area and forecast charts."""

import matplotlib.pyplot as plt
import pandas as pd

from src.baselines import REPORTS_DIR, mae
from src.train import FEATURES, load_training_data, make_model

TEST_YEAR = 2025
SHOW_AREAS = ["Colombo", "Kandy", "Jaffna"]


def predict_test_year(df: pd.DataFrame, year: int = TEST_YEAR) -> pd.DataFrame:
    """Train on years before `year`, predict every week of `year`."""
    train, test = df[df["year"] < year], df[df["year"] == year].copy()
    model = make_model().fit(train[FEATURES], train["target"])
    test["pred_model"] = model.predict(test[FEATURES])
    test["pred_last_week"] = test["cases"]
    return test


def error_by_area(test: pd.DataFrame) -> pd.DataFrame:
    """MAE per area for the model and the baseline, sorted by total cases."""
    rows = []
    for area, g in test.groupby("district", observed=True):
        rows.append(
            {
                "district": area,
                "total_cases": int(g["target"].sum()),
                "baseline_mae": mae(g["target"], g["pred_last_week"]),
                "model_mae": mae(g["target"], g["pred_model"]),
            }
        )
    out = pd.DataFrame(rows).sort_values("total_cases", ascending=False)
    out["model_better"] = out["model_mae"] < out["baseline_mae"]
    return out.round(1).reset_index(drop=True)


def plot_forecasts(test: pd.DataFrame, areas=SHOW_AREAS) -> None:
    """Actual vs predicted next-week cases for a few areas."""
    fig, axes = plt.subplots(len(areas), 1, figsize=(11, 3 * len(areas)), sharex=True)
    for ax, area in zip(axes, areas, strict=True):
        g = test[test["district"] == area]
        ax.plot(g["start_date"], g["target"], label="Actual", color="black")
        ax.plot(g["start_date"], g["pred_model"], label="Model", color="tab:orange")
        ax.plot(
            g["start_date"],
            g["pred_last_week"],
            label="Same as last week",
            color="tab:blue",
            alpha=0.5,
            linestyle="--",
        )
        ax.set_title(f"{area}: next-week dengue cases, {TEST_YEAR}")
        ax.set_ylabel("Cases")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "07_forecasts_2025.png", dpi=150)


if __name__ == "__main__":
    test = predict_test_year(load_training_data())
    by_area = error_by_area(test)
    by_area.to_csv(REPORTS_DIR / "error_by_area_2025.csv", index=False)
    print(by_area.to_string(index=False))
    wins = by_area["model_better"].sum()
    print(f"\nModel beats the baseline in {wins} of {len(by_area)} areas ({TEST_YEAR})")
    plot_forecasts(test)
