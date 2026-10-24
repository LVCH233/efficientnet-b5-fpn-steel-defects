"""Fixed group evaluation of final masks and count-based checkpoint selection.

Prediction thresholds and component filtering belong to inference, before this
adapter. The bundled group evaluator always runs with no additional area filter.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path

import numpy as np


EVALUATOR_SHA256 = "784722c2d57667a681e61b5637bbe68acd3532ca8bc158a924e2b58eb9f70189"
COUNT_KEYS = ("n_gt", "detected", "false_alarms", "correct_class")
SELECTION_KEYS = ("best_recall", "best_accuracy", "lowest_false_alarms", "best_controlled_recall")

_repository_reference = Path(__file__).resolve().parents[2] / "reference" / "group_unified_v2.py"
_packaged_reference = Path(__file__).resolve().parent / "_reference" / "group_unified_v2.py"
_reference_path = _repository_reference if _repository_reference.is_file() else _packaged_reference
if hashlib.sha256(_reference_path.read_bytes()).hexdigest() != EVALUATOR_SHA256:
    raise RuntimeError("Bundled group evaluator SHA256 does not match the fixed protocol")
_spec = importlib.util.spec_from_file_location("_vdcp_group_unified_v2", _reference_path)
_supplied = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_supplied)


def _binary_masks(value, name):
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] != 4 or min(array.shape[1:]) < 1:
        raise ValueError(f"{name} must have shape (4, H, W) with positive H and W")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return array.astype(np.uint8, copy=False)


def score_final_masks(gt, pred):
    """Score binary (4,H,W) masks with the exact supplied group protocol.

    GT classes must not overlap. Prediction classes may overlap. Per-class keys
    are strings 1..4; their false alarms are attributed to prediction classes,
    while GT and detection counts are attributed to annotation classes.
    """
    gt = _binary_masks(gt, "gt")
    pred = _binary_masks(pred, "pred")
    if gt.shape != pred.shape:
        raise ValueError("gt and pred must have identical shapes")
    if np.any(gt.sum(axis=0) > 1):
        raise ValueError("GT classes must not overlap")
    result = _supplied.eval_image(gt, pred, cov=0.3, inside=0.5, tol=15, min_area=0)
    return dict(
        n_gt=int(result["n_gt"]),
        detected=int(result["found"]),
        false_alarms=int(result["fp"]),
        correct_class=int(result["correct"]),
        per_class={
            str(c): dict(n_gt=int(result["pc_gt"][c]),
                         detected=int(result["pc_found"][c]),
                         false_alarms=int(result["pc_fp"][c]))
            for c in range(1, 5)
        },
    )


def summarize_counts(counts):
    """Add rates to accumulated counts, preserving metadata such as epoch.

    R=D/G, F=FP/G, A=C/(D+FP). Undefined rates use NaN, as in the supplied
    implementation. Accumulate raw counts over images before calling this.
    """
    result = deepcopy(counts)
    for key in COUNT_KEYS:
        result[key] = int(counts[key])
    ng, detected = result["n_gt"], result["detected"]
    recognized = detected + result["false_alarms"]
    result.update(
        recognized=recognized,
        recall=detected / ng if ng else float("nan"),
        false_alarm_rate=result["false_alarms"] / ng if ng else float("nan"),
        classification_accuracy=result["correct_class"] / recognized if recognized else float("nan"),
    )
    return result


def _selection_rank(counts, selector):
    detected, false_alarms, correct = (int(counts[key]) for key in
                                     ("detected", "false_alarms", "correct_class"))
    recognized = detected + false_alarms
    accuracy = Fraction(correct, recognized) if recognized else Fraction(0)
    if selector == "best_accuracy":
        return accuracy, detected, -false_alarms
    if selector == "lowest_false_alarms":
        return -false_alarms, detected, accuracy
    return detected, accuracy, -false_alarms


def selection_updates(previous, counts, baseline_false_alarms):
    """Return (selector->count snapshot, selected keys), without mutating inputs.

    Exact rational comparisons resolve accuracy ties. Complete ties retain the
    incumbent. Only the controlled-recall selector requires FP <= baseline.
    Additional count fields, including epoch and summary rates, are preserved.
    """
    updated = deepcopy(previous)
    selected = []
    for key in SELECTION_KEYS:
        if key == "best_controlled_recall" and counts["false_alarms"] > baseline_false_alarms:
            continue
        incumbent = previous.get(key)
        if incumbent is None or _selection_rank(counts, key) > _selection_rank(incumbent, key):
            snapshot = deepcopy(counts)
            for count_key in COUNT_KEYS:
                snapshot[count_key] = int(counts[count_key])
            updated[key] = snapshot
            selected.append(key)
    return updated, selected
