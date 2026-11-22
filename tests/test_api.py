"""Tests for the forecast API."""

from fastapi.testclient import TestClient

from src.api import app

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
    assert body["forecast_next_week"] >= 0


def test_unknown_area_returns_404():
    response = client.get("/forecast/atlantis")
    assert response.status_code == 404
