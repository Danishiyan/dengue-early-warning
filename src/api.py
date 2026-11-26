"""Web API that serves next-week dengue forecasts."""

import logging
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features import FEATURES_CSV, RECENT_WEEKS
from src.train import FEATURES, MODEL_PATH

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sri Lanka Dengue Early Warning",
    description=(
        "Forecasts dengue cases for 26 Sri Lankan health areas, one week after the latest "
        "fully reported week."
    ),
    version="0.1.0",
)


class Forecast(BaseModel):
    district: str
    based_on_week: str
    cases_that_week: int
    forecast_week_starting: str
    forecast_cases: float
    weeks_still_being_reported: int


@lru_cache
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache
def load_latest_rows() -> pd.DataFrame:
    """The latest fully reported week for every area, with the model's feature columns.

    The newest weeks are still being reported (`is_recent`), so their counts are too low.
    Forecasting from them would look like a sudden drop or rise, so they are skipped.
    """
    df = pd.read_csv(FEATURES_CSV, parse_dates=["start_date"])
    df["district"] = df["district"].astype("category")
    df = df.sort_values(["district", "start_date"])
    df["next_week_start"] = df.groupby("district", observed=True)["start_date"].shift(-1)
    complete = df[~df["is_recent"]]
    return complete.groupby("district", observed=True).tail(1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/areas")
def areas() -> list[str]:
    return sorted(load_latest_rows()["district"].astype(str))


@app.get("/forecast/{district}", response_model=Forecast)
def forecast(district: str) -> Forecast:
    rows = load_latest_rows()
    row = rows[rows["district"].astype(str).str.lower() == district.lower()]
    if row.empty:
        logger.warning("Forecast requested for unknown area: %s", district)
        raise HTTPException(status_code=404, detail=f"Unknown area: {district}")

    prediction = float(load_model().predict(row[FEATURES])[0])
    latest = row.iloc[0]
    return Forecast(
        district=str(latest["district"]),
        based_on_week=latest["start_date"].date().isoformat(),
        cases_that_week=int(latest["cases"]),
        forecast_week_starting=latest["next_week_start"].date().isoformat(),
        forecast_cases=round(max(prediction, 0.0), 1),
        weeks_still_being_reported=RECENT_WEEKS,
    )
