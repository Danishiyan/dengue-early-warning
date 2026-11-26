import mlflow
import pandas as pd

from src.experiment import EXPERIMENT, log_results


def test_log_results_creates_one_run_per_method(tmp_path):
    results = pd.DataFrame(
        {"baseline_last_week": [10.0, 12.0], "model": [9.0, 11.0]},
        index=pd.Index([2024, 2025], name="test_year"),
    )
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    run_ids = log_results(results, {"loss": "absolute_error"}, uri, tmp_path / "artifacts")

    assert set(run_ids) == {"baseline_last_week", "model"}
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    model_run = runs[runs["run_id"] == run_ids["model"]].iloc[0]
    assert model_run["metrics.mae_avg"] == 10.0
    assert model_run["metrics.mae_2025"] == 11.0
    assert model_run["params.loss"] == "absolute_error"
    assert model_run["tags.version"] == "v1"
