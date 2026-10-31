"""Tests for the experimental study; this does not test production calibration."""
import numpy as np
import pandas as pd
import pytest

from experiments.conformal_study import conformal_quantile, summarize


def test_finite_sample_order_statistic_and_small_sample_guard():
    assert conformal_quantile([8, 2, 7, 1], .8) == 8
    with pytest.raises(ValueError):
        conformal_quantile([8, 2, 7, 1], .99)
    with pytest.raises(ValueError):
        conformal_quantile([np.nan], .8)


def test_coverage_uses_whole_events_and_pc_driven_subset():
    rows = pd.DataFrame({"event_id": [1, 1, 2], "risk": [-5., -2., -20.],
                         "miss_distance": [2000., 3000., 500.]})
    result = summarize(rows, np.array([-5., -10., -20.]), 1.)
    assert result["row_coverage"] == pytest.approx(2 / 3)
    assert result["all_messages_event_coverage"] == .5
    assert result["pc_driven_critical_rows"] == 1
    assert result["point_estimate_pc_driven_misses"] == 1
    assert result["point_misses_with_interval_straddling_critical"] == 0


def test_unscorable_prediction_does_not_disappear_from_coverage():
    rows = pd.DataFrame({"event_id": [1, 2], "risk": [-3., -7.], "miss_distance": [2000., 2000.]})
    result = summarize(rows, np.array([np.nan, -7.]), 1.)
    assert result["n_rows"] == 2
    assert result["n_scored_rows"] == 1
    assert result["n_unscorable_predictions"] == 1
    assert result["row_coverage"] == .5
    assert result["point_estimate_pc_driven_misses"] == 1
    assert result["pc_driven_critical_row_coverage"] == 0
