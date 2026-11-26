from pathlib import Path

from src.config import PROJECT_ROOT, load_config


def test_config_has_all_sections():
    config = load_config()
    assert {"paths", "data", "evaluation", "model"} <= config.keys()


def test_paths_are_absolute_and_inside_project():
    for path in load_config()["paths"].values():
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents


def test_values_match_current_code():
    config = load_config()
    assert config["data"]["start_year"] == 2010
    assert config["data"]["max_lag"] == 8
    assert config["evaluation"]["test_years"] == [2021, 2022, 2023, 2024, 2025]
    assert config["model"]["random_state"] == 42
