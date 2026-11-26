"""Web API that serves next-week dengue forecasts."""

import logging
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features import FEATURES_CSV
from src.train import FEATURES, MODEL_PATH

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sri Lanka Dengue Early Warning",
    description="Forecasts next week's dengue cases for 26 health areas.",
    version="0.1.0",
)


class Forecast(BaseModel):
    district: str
    week_starting: str
    cases_this_week: int
    forecast_next_week: float
    data_may_be_incomplete: bool


@lru_cache
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache
def load_latest_rows() -> pd.DataFrame:
    """The most recent week for every area, with the model's feature columns."""
    df = pd.read_csv(FEATURES_CSV, parse_dates=["start_date"])
    df["district"] = df["district"].astype("category")
    return df.sort_values("start_date").groupby("district", observed=True).tail(1)


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
        week_starting=latest["start_date"].date().isoformat(),
        cases_this_week=int(latest["cases"]),
        forecast_next_week=round(max(prediction, 0.0), 1),
        data_may_be_incomplete=bool(latest["is_recent"]),
    )
