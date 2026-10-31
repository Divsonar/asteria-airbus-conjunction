"""Updated challenge leakage boundaries and final-tier forecast contract."""
import numpy as np
import pandas as pd
import pytest

from src.pipeline import (CLASS_LABELS, FEATURES, clean_features, latest_message,
                          load_event_labels, metrics, predict_messages,
                          prepare_training_events, split_training_events)


class ReorderedClasses:
    classes_ = np.array(["Safe", "Critical", "Elevated"])

    def predict_proba(self, features):
        assert list(features.columns) == FEATURES
        assert (features.time_to_tca >= 2).all()
        return np.tile([.2, .7, .1], (len(features), 1))


def telemetry():
    frame = pd.DataFrame({key: [10.] * 5 for key in FEATURES})
    frame["event_id"] = ["001", "NA", "001", "late", "bad"]
    frame["source_row_id"] = [5, 7, 9, 11, 13]
    frame["c_object_type"] = "DEBRIS"
    frame["time_to_tca"] = [3., 2., 1., -.1, np.nan]
    frame["miss_distance"] = 6000.
    frame["risk"] = 999.
    frame["final_risk_log10_pc"] = 999.
    frame["final_tier"] = "Safe"
    return frame


def test_prediction_reorders_classes_and_excludes_late_messages():
    frame = telemetry()
    result = predict_messages(frame, model=ReorderedClasses())
    assert len(result) == len(frame)
    assert result.source_row_id.tolist() == [5, 7, 9, 11, 13]
    assert result.event_id.tolist() == ["001", "NA", "001", "late", "bad"]
    assert result.predicted_final_tier.tolist() == ["Critical", "Critical", "Incomplete", "Incomplete", "Incomplete"]
    np.testing.assert_allclose(result.loc[:1, ["prob_critical", "prob_elevated", "prob_safe"]], [[.7, .1, .2]] * 2)
    assert result.loc[2:, ["prob_critical", "prob_elevated", "prob_safe"]].isna().all().all()
    assert set(result.risk_tier) == {"Critical", "Incomplete"}
    assert not {"risk", "final_tier", "final_risk_log10_pc", "estimated_pc", "predicted_log10_pc"} & set(result.columns)
    assert "not Pc" in result.triggered_rules.iloc[0]


def test_latest_snapshot_stops_at_cutoff_and_breaks_ties_by_source_order():
    frame = pd.DataFrame({"event_id": ["a", "a", "a", "b", "b", "c"],
                          "source_row_id": [0, 1, 2, 3, 4, 5],
                          "time_to_tca": [4., 2., .01, 2., 2., 1.]})
    latest = latest_message(frame).set_index("event_id")
    assert latest.source_row_id.to_dict() == {"a": 1, "b": 3}
    assert "c" not in latest.index


def test_late_message_mutation_cannot_change_training_features():
    frame = telemetry().iloc[:3].copy()
    labels = pd.DataFrame({"event_id": ["001", "NA"], "final_tier": ["Elevated", "Safe"],
                           "cdm_count": [2, 1], "cdm_count_2d_or_more": [1, 1],
                           "usable_for_training": ["yes", "yes"]})
    before = prepare_training_events(frame, labels)
    frame.loc[frame.time_to_tca.lt(2), "miss_distance"] = 0
    frame["risk"] = -2
    after = prepare_training_events(frame, labels)
    pd.testing.assert_frame_equal(clean_features(before), clean_features(after))
    assert before.final_tier.tolist() == ["Elevated", "Safe"]
    assert "final_risk_log10_pc" not in clean_features(before)


def test_label_join_rejects_wrong_file_ids_and_count_mismatches():
    frame = telemetry().iloc[:2]
    labels = pd.DataFrame({"event_id": ["001", "other"], "final_tier": ["Critical", "Safe"],
                           "cdm_count": [1, 1], "cdm_count_2d_or_more": [1, 1],
                           "usable_for_training": ["yes", "yes"]})
    with pytest.raises(ValueError, match="never join test"):
        prepare_training_events(frame, labels)
    labels["event_id"] = ["001", "NA"]
    labels.loc[0, "cdm_count"] = 2
    with pytest.raises(ValueError, match="CDM counts"):
        prepare_training_events(frame, labels)


def test_split_is_reproducible_event_disjoint_and_class_stratified():
    frame = pd.DataFrame({"event_id": [f"event-{i}" for i in range(150)],
                          "final_tier": np.repeat(CLASS_LABELS, 50)})
    first = split_training_events(frame)
    second = split_training_events(frame)
    assert [len(part) for part in first] == [90, 30, 30]
    assert [part.event_id.tolist() for part in first] == [part.event_id.tolist() for part in second]
    group_ids = [set(part.event_id) for part in first]
    assert not (group_ids[0] & group_ids[1] or group_ids[0] & group_ids[2] or group_ids[1] & group_ids[2])
    assert all(set(part.final_tier) == set(CLASS_LABELS) for part in first)


def test_metrics_compare_forecast_to_final_tier_not_current_distance():
    frame = pd.DataFrame({"event_id": ["a", "b", "c"], "final_tier": CLASS_LABELS, "miss_distance": [9000, 9000, 9000]})
    result = metrics(frame, ["Safe", "Elevated", "Safe"], [[.1, .1, .8], [.1, .8, .1], [.1, .1, .8]])
    assert result["critical_fn"] == 1
    assert result["critical_recall"] == 0
    assert result["critical_true"] == 1
    assert result["macro_f1"] == pytest.approx((0 + 1 + 2 / 3) / 3)
    assert result["log_loss"] > 0
    assert result["multiclass_brier"] > 0


def test_organizer_workbook_is_complete_and_unique():
    labels = load_event_labels()
    assert len(labels) == 13154
    assert labels.event_id.is_unique
    assert labels.final_tier.value_counts().to_dict() == {"Safe": 9843, "Elevated": 2403, "Critical": 908}
    assert labels.usable_for_training.eq("yes").sum() == 11942
