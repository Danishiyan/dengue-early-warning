"""Log cross-validation results to MLflow so runs can be compared over time.

Run:   python -m src.experiment
View:  mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   (then open http://127.0.0.1:5000)
"""

import logging
from pathlib import Path

import mlflow
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.log import setup_logging
from src.train import FEATURES, cross_validate, load_training_data

logger = logging.getLogger(__name__)

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT = "dengue-next-week-cases"


def log_results(
    results: pd.DataFrame,
    params: dict[str, dict],
    tracking_uri: str,
    artifact_dir: Path,
    version: str = "v1",
) -> dict[str, str]:
    """Log one MLflow run per method (baseline and model). Returns {method: run_id}.

    `params` maps each method to its own settings, so the baseline run doesn't claim
    model settings it never used.
    """
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(EXPERIMENT)

    run_ids = {}
    for method in results.columns:
        maes = results[method]
        with mlflow.start_run(run_name=f"{version}-{method}") as run:
            mlflow.set_tags({"version": version, "method": method})
            mlflow.log_params(params[method])
            for year, value in maes.items():
                mlflow.log_metric(f"mae_{year}", float(value))
            mlflow.log_metric("mae_avg", float(maes.mean()))
            run_ids[method] = run.info.run_id
        logger.info("Logged %s: average MAE %.1f", method, maes.mean())
    return run_ids


def experiment_params() -> dict[str, dict]:
    """The settings behind each run, read from the config."""
    config = load_config()
    data = {
        "start_year": config["data"]["start_year"],
        "test_years": ",".join(str(y) for y in config["evaluation"]["test_years"]),
    }
    return {
        "baseline_last_week": {**data, "rule": "next week = this week"},
        "model": {
            **data,
            "model": "HistGradientBoostingRegressor",
            "loss": config["model"]["loss"],
            "random_state": config["model"]["random_state"],
            "max_lag": config["data"]["max_lag"],
            "n_features": len(FEATURES),
        },
    }


if __name__ == "__main__":
    setup_logging()
    results = cross_validate(load_training_data())
    MLRUNS_DIR.mkdir(exist_ok=True)
    run_ids = log_results(
        results,
        experiment_params(),
        tracking_uri=f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}",
        artifact_dir=MLRUNS_DIR / "artifacts",
    )
    logger.info("Runs: %s", run_ids)
