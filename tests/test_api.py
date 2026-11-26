"""Tests for the forecast API."""

import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.features import FEATURES_CSV

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_areas_lists_all_26():
    response = client.get("/areas")
    assert response.status_code == 200
    assert len(response.json()) == 26


def test_forecast_for_known_area():
    response = client.get("/forecast/colombo")
    body = response.json()
    assert response.status_code == 200
    assert body["district"] == "Colombo"
    assert body["forecast_cases"] >= 0
    assert body["forecast_week_starting"] > body["based_on_week"]


def test_forecast_uses_a_fully_reported_week():
    df = pd.read_csv(FEATURES_CSV)
    complete = df[(df["district"] == "Colombo") & ~df["is_recent"]]
    body = client.get("/forecast/colombo").json()

    assert body["based_on_week"] == complete["start_date"].max()
    assert body["weeks_still_being_reported"] == 4


def test_unknown_area_returns_404():
    response = client.get("/forecast/atlantis")
    assert response.status_code == 404
