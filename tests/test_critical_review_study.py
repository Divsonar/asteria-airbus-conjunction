"""Offline review overlay evaluation and threshold selection."""
import numpy as np
import pandas as pd
import pytest

from experiments.critical_review_study import evaluate, selection_threshold


def test_threshold_catches_validation_misses_without_reflagging_point_criticals():
    frame = pd.DataFrame({"event_id": [1, 1, 2, 3, 4],
                          "risk": [-3., -3., -3., -5., -5.],
                          "miss_distance": [1200., 2000., 5000., 2000., 2000.]})
    point = np.array([-7., -7., -3., -7., -7.])
    score = np.array([.8, .4, .9, .5, .2])
    threshold = selection_threshold(frame, point, score)
    assert threshold == .4
    result = evaluate(frame, point, score, threshold)
    assert result["point_model_missed_pc_rows"] == 2
    assert result["point_model_missed_pc_events"] == 1
    assert result["true_misses_flagged_rows"] == 2
    assert result["false_review_rows"] == 1
    assert result["review_precision"] == pytest.approx(2/3)
    assert result["additional_review_rate_all_rows"] == pytest.approx(3/5)
    assert result["remaining_pc_driven_missed_rows"] == 0


def test_no_misses_cannot_select_threshold():
    frame = pd.DataFrame({"event_id": [1], "risk": [-3.], "miss_distance": [900.]})
    assert selection_threshold(frame, np.array([-7.]), np.array([.9])) is None


def test_unscorable_critical_cases_cannot_disappear_from_recall():
    frame = pd.DataFrame({"event_id": [1, 2], "risk": [-3., -3.], "miss_distance": [2000., 2000.]})
    point, score = np.array([np.nan, -7.]), np.array([.9, np.nan])
    result = evaluate(frame, point, score, .5)
    assert result["point_model_missed_pc_rows"] == 2
    assert result["true_misses_flagged_rows"] == 1
    assert result["remaining_pc_driven_missed_rows"] == 1
    assert result["unscorable_point_rows"] == 1
    assert result["unscorable_review_score_rows"] == 1
    with pytest.raises(ValueError, match="Missing review scores"):
        selection_threshold(frame, point, score)
