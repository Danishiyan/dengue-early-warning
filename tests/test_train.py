"""Guard the lessons learned while building the model."""

from src.train import FEATURES


def test_model_sees_current_week():
    # Lesson from PR #6: without this week's cases the model lost to the baseline.
    assert "cases" in FEATURES


def test_target_is_never_a_feature():
    # Using next week's value as an input would be data leakage.
    assert "target" not in FEATURES
