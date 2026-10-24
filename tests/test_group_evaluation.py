"""Fixed group protocol and checkpoint-selection contracts (CPU only)."""
import copy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


SHA256 = "784722c2d57667a681e61b5637bbe68acd3532ca8bc158a924e2b58eb9f70189"
ROOT = Path(__file__).resolve().parents[1]
COUNT_KEYS = ("n_gt", "detected", "false_alarms", "correct_class")
SELECTORS = ("best_recall", "best_accuracy", "lowest_false_alarms", "best_controlled_recall")


def api():
    assert importlib.util.find_spec("vdcp.group_evaluation") is not None, "Group evaluation adapter is missing"
    return importlib.import_module("vdcp.group_evaluation")


def masks():
    return np.zeros((4, 100, 100), dtype=np.uint8)


def counts(detected=2607, false_alarms=698, correct_class=2574, n_gt=3029, **extra):
    return dict(n_gt=n_gt, detected=detected, false_alarms=false_alarms,
                correct_class=correct_class, **extra)


def assert_counts(result, expected):
    assert {key: result[key] for key in COUNT_KEYS} == dict(zip(COUNT_KEYS, expected))
    assert all(type(result[key]) is int for key in COUNT_KEYS)


def test_reference_source_is_exact_supplied_bytes():
    api()
    source = (ROOT / "reference/group_unified_v2.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == SHA256
    assert source == (ROOT / "reports/v9/unified_protocol_20261019/supplied/evaluate_v2.py").read_bytes()


def test_perfect_single_pixel_gt_is_not_filtered():
    gt = masks()
    gt[0, 40, 40] = 1
    result = api().score_final_masks(gt, gt.copy())
    assert_counts(result, (1, 1, 0, 1))
    assert result["per_class"] == {
        str(c): dict(n_gt=int(c == 1), detected=int(c == 1), false_alarms=0)
        for c in range(1, 5)
    }
    json.dumps(result, allow_nan=False)


def test_eight_connected_diagonal_pixels_form_one_gt():
    gt = masks()
    gt[3, 40, 40] = gt[3, 41, 41] = 1
    assert_counts(api().score_final_masks(gt, gt), (1, 1, 0, 1))


@pytest.mark.parametrize("covered,detected", [(2, 0), (3, 1)])
def test_thirty_percent_coverage_integer_boundary(covered, detected):
    gt, pred = masks(), masks()
    gt[0, 40, 40:50] = 1
    pred[0, 40, 40:40 + covered] = 1
    assert_counts(api().score_final_masks(gt, pred), (1, detected, 0, detected))


@pytest.mark.parametrize("length,expected", [(49, (1, 0, 1, 0)), (50, (1, 1, 0, 1))])
def test_fifty_percent_inside_integer_boundary(length, expected):
    gt, pred = masks(), masks()
    gt[0, 40, 40] = 1  # The 31-wide dilated union begins at x=25.
    pred[0, 40, :length] = 1
    assert_counts(api().score_final_masks(gt, pred), expected)


def test_square_dilation_includes_diagonal_corner_at_distance_fifteen():
    gt, pred = masks(), masks()
    gt[0, 40, 40] = 1
    pred[1, 25, 25] = 1
    assert_counts(api().score_final_masks(gt, pred), (1, 0, 0, 0))
    pred[1, 25, 25] = 0
    pred[1, 24, 24] = 1
    assert_counts(api().score_final_masks(gt, pred), (1, 0, 1, 0))


def test_wrong_class_still_detects_gt_and_preserves_class_counters():
    gt, pred = masks(), masks()
    gt[2, 40:42, 40:45] = 1
    pred[1] = gt[2]
    pred[3, 0, 0] = 1
    result = api().score_final_masks(gt, pred)
    assert_counts(result, (1, 1, 1, 0))
    assert result["per_class"]["3"] == dict(n_gt=1, detected=1, false_alarms=0)
    assert result["per_class"]["4"] == dict(n_gt=0, detected=0, false_alarms=1)


def test_majority_class_counts_only_pixels_inside_gt():
    gt, pred = masks(), masks()
    gt[2, 40, 40:50] = 1
    pred[2, 40, 40:46] = 1
    pred[0, 40, 46:60] = 1  # Larger component, only four pixels inside GT.
    assert_counts(api().score_final_masks(gt, pred), (1, 1, 0, 1))


def test_class_vote_tie_matches_supplied_lowest_class_argmax():
    gt, pred = masks(), masks()
    gt[1, 40, 40:50] = 1
    pred[0, 40, 40:45] = 1
    pred[1, 40, 45:50] = 1
    assert_counts(api().score_final_masks(gt, pred), (1, 1, 0, 0))


def test_false_positive_components_are_removed_before_detection():
    gt, pred = masks(), masks()
    gt[0, 40, 40] = 1
    pred[:] = 1
    result = api().score_final_masks(gt, pred)
    assert_counts(result, (1, 0, 4, 0))
    summary = api().summarize_counts(result)
    assert summary["recognized"] == 4
    assert summary["false_alarm_rate"] == 4
    assert summary["classification_accuracy"] == 0


def test_one_merged_prediction_can_detect_multiple_gt():
    gt, pred = masks(), masks()
    gt[0, 40, 40:43] = gt[0, 40, 50:53] = 1
    pred[0, 40, 40:53] = 1
    assert_counts(api().score_final_masks(gt, pred), (2, 2, 0, 2))


def test_split_predictions_count_one_detected_gt():
    gt, pred = masks(), masks()
    gt[0, 40:42, 40:50] = 1
    pred[0, 40:42, 40:43] = pred[0, 40:42, 47:50] = 1
    assert_counts(api().score_final_masks(gt, pred), (1, 1, 0, 1))


def test_empty_masks_and_empty_gt_use_supplied_nan_denominators():
    gt, pred = masks(), masks()
    result = api().summarize_counts(api().score_final_masks(gt, pred))
    assert_counts(result, (0, 0, 0, 0))
    assert result["recognized"] == 0
    assert all(math.isnan(result[key]) for key in ("recall", "false_alarm_rate", "classification_accuracy"))
    pred[0, 10, 10] = 1
    result = api().summarize_counts(api().score_final_masks(gt, pred))
    assert_counts(result, (0, 0, 1, 0))
    assert result["classification_accuracy"] == 0


@pytest.mark.parametrize("bad", [np.zeros((4, 100)), np.zeros((3, 100, 100)),
                                  np.zeros((4, 99, 100)), np.full((4, 100, 100), .5),
                                  np.full((4, 100, 100), np.nan)])
def test_rejects_invalid_mask_shape_and_nonbinary_values(bad):
    evaluator = api()
    for gt, pred in [(bad, masks()), (masks(), bad)]:
        with pytest.raises(ValueError):
            evaluator.score_final_masks(gt, pred)


def test_rejects_overlapping_gt_but_allows_overlapping_predictions():
    evaluator = api()
    gt, pred = masks(), masks()
    gt[0, 40, 40] = gt[1, 40, 40] = 1
    with pytest.raises(ValueError, match="overlap"):
        evaluator.score_final_masks(gt, pred)
    gt[1, 40, 40] = 0
    pred[0, 40, 40] = pred[1, 40, 40] = 1
    assert_counts(evaluator.score_final_masks(gt, pred), (1, 1, 0, 1))


def test_summary_uses_accumulated_dataset_counts_with_3029_gt():
    evaluator = api()
    parts = [counts(1, 1, 1, n_gt=1), counts(2606, 697, 2573, n_gt=3028)]
    total = {key: np.int64(sum(part[key] for part in parts)) for key in COUNT_KEYS}
    total["epoch"] = 24
    original = copy.deepcopy(total)
    result = evaluator.summarize_counts(total)
    assert_counts(result, (3029, 2607, 698, 2574))
    assert result["recognized"] == 3305
    assert result["recall"] == 2607 / 3029
    assert result["false_alarm_rate"] == 698 / 3029
    assert result["classification_accuracy"] == 2574 / 3305
    assert result["epoch"] == 24
    assert total == original
    json.dumps(result, allow_nan=False)


def test_selectors_initialize_and_preserve_epoch_metadata_without_mutation():
    evaluator = api()
    current = counts(epoch=24, note={"run": ["baseline"]})
    result, selected = evaluator.selection_updates({}, current, 698)
    assert selected == list(SELECTORS)
    assert all(value == current for value in result.values())
    result["best_recall"]["note"]["run"].append("changed")
    assert current["note"]["run"] == ["baseline"]
    assert result["best_accuracy"]["note"]["run"] == ["baseline"]


def test_recall_can_improve_without_meeting_controlled_false_alarm_limit():
    evaluator = api()
    previous, _ = evaluator.selection_updates({}, counts(epoch=24), 698)
    original = copy.deepcopy(previous)
    result, selected = evaluator.selection_updates(previous, counts(2700, 700, 2600, epoch=25), 698)
    assert selected == ["best_recall"]
    assert result["best_recall"]["epoch"] == 25
    assert result["best_controlled_recall"]["epoch"] == 24
    assert previous == original


def test_controlled_selector_initializes_only_at_or_below_baseline():
    evaluator = api()
    result, selected = evaluator.selection_updates({}, counts(false_alarms=699), 698)
    assert "best_controlled_recall" not in result
    assert "best_controlled_recall" not in selected
    result, selected = evaluator.selection_updates(result, counts(false_alarms=698), 698)
    assert "best_controlled_recall" in selected


@pytest.mark.parametrize("selector,incumbent,candidate,better", [
    ("best_recall", counts(10, 2, 10), counts(11, 100, 0), True),
    ("best_recall", counts(10, 2, 9), counts(10, 2, 10), True),
    ("best_recall", counts(10, 10, 10), counts(10, 8, 9), True),
    ("best_recall", counts(10, 10, 10), counts(10, 12, 11), False),
    ("best_accuracy", counts(10, 10, 10), counts(9, 1, 9), True),
    ("best_accuracy", counts(10, 10, 10), counts(11, 11, 11), True),
    ("best_accuracy", counts(10, 10, 10), counts(10, 8, 9), True),
    ("lowest_false_alarms", counts(10, 10, 10), counts(1, 9, 0), True),
    ("lowest_false_alarms", counts(10, 10, 10), counts(11, 10, 0), True),
    ("lowest_false_alarms", counts(10, 10, 9), counts(10, 10, 10), True),
    ("lowest_false_alarms", counts(10, 10, 10), counts(100, 11, 100), False),
    ("best_controlled_recall", counts(10, 10, 10), counts(10, 8, 9), True),
])
def test_selector_orderings_and_ratio_ties(selector, incumbent, candidate, better):
    evaluator = api()
    incumbent["epoch"] = 24
    candidate["epoch"] = 25
    result, selected = evaluator.selection_updates({selector: incumbent}, candidate, 100)
    assert (selector in selected) is better
    assert result[selector]["epoch"] == (25 if better else 24)


def test_complete_selection_ties_retain_existing_checkpoint():
    evaluator = api()
    previous, _ = evaluator.selection_updates({}, counts(epoch=24), 698)
    result, selected = evaluator.selection_updates(previous, counts(epoch=25), 698)
    assert selected == []
    assert result == previous


def test_selection_uses_raw_counts_even_when_summary_fields_are_stale_and_fp_exceeds_gt():
    evaluator = api()
    current = counts(10, 4000, 10, classification_accuracy=1.0, epoch=24)
    previous = {"best_accuracy": current}
    candidate = counts(9, 3000, 9, classification_accuracy=0.0, epoch=25)
    result, selected = evaluator.selection_updates(previous, candidate, 698)
    assert "best_accuracy" in selected
    assert result["best_accuracy"] == candidate
