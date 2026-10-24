"""Defect-region matching with the assignment's exact denominators."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


@dataclass
class Region:
    class_id: int
    box: tuple
    mask: np.ndarray
    area: int


def extract_regions(masks, min_areas=None):
    array = np.asarray(masks)
    if array.ndim != 3 or array.shape[0] != 4:
        raise ValueError('Expected four binary masks in C,H,W order')
    limits = [1] * 4 if min_areas is None else list(min_areas)
    regions = []
    for channel in range(4):
        if not np.any(array[channel]):
            continue
        labels, _ = ndimage.label(array[channel] > 0, structure=np.ones((3, 3)))
        for identity, slices in enumerate(ndimage.find_objects(labels), start=1):
            if slices is None:
                continue
            y, x = slices
            crop = labels[slices] == identity
            area = int(crop.sum())
            if area >= limits[channel]:
                regions.append(Region(channel + 1, (x.start, y.start, x.stop, y.stop), crop, area))
    return regions


def region_iou(first, second):
    ax, ay, ar, ab = first.box
    bx, by, br, bb = second.box
    left, top, right, bottom = max(ax, bx), max(ay, by), min(ar, br), min(ab, bb)
    if left >= right or top >= bottom:
        return 0.0
    overlap = np.logical_and(first.mask[top-ay:bottom-ay, left-ax:right-ax],
                             second.mask[top-by:bottom-by, left-bx:right-bx]).sum()
    return float(overlap / (first.area + second.area - overlap))


def match_iou_matrix(ious, threshold=0.5):
    matrix = np.asarray(ious, dtype=float)
    if matrix.ndim != 2:
        raise ValueError('Expected a truth-by-prediction IoU matrix')
    truths, predictions = matrix.shape
    if truths == 0 or predictions == 0:
        return []
    # A count increment is always more valuable than the entire IoU tie-break.
    rewards = np.zeros((truths, predictions + truths), dtype=float)
    rewards[:, :predictions] = np.where(matrix >= threshold,
        1.0 + matrix / (min(truths, predictions) + 1), -1.0)
    rows, cols = linear_sum_assignment(-rewards)
    return [(int(i), int(j)) for i, j in zip(rows, cols)
            if j < predictions and matrix[i, j] >= threshold]


def region_counts(truth_masks, prediction_masks, threshold=0.5, min_areas=None):
    truth = extract_regions(truth_masks)
    prediction = extract_regions(prediction_masks, min_areas)
    ious = np.zeros((len(truth), len(prediction)), dtype=float)
    for i, actual in enumerate(truth):
        for j, estimated in enumerate(prediction):
            ious[i, j] = region_iou(actual, estimated)
    matches = match_iou_matrix(ious, threshold)
    correct = sum(truth[i].class_id == prediction[j].class_id for i, j in matches)
    matched_truth, matched_prediction = {i for i, _ in matches}, {j for _, j in matches}
    by_class = []
    for class_id in range(1, 5):
        gt_ids = {i for i, region in enumerate(truth) if region.class_id == class_id}
        pred_ids = {j for j, region in enumerate(prediction) if region.class_id == class_id}
        by_class.append(dict(class_id=class_id, n_gt=len(gt_ids), n_pred=len(pred_ids),
            tp=len(gt_ids & matched_truth), fn=len(gt_ids - matched_truth),
            fp=len(pred_ids - matched_prediction), correct_class=sum(
                truth[i].class_id == class_id and prediction[j].class_id == class_id for i, j in matches)))
    return dict(n_gt=len(truth), n_pred=len(prediction), tp=len(matches),
        fp=len(prediction)-len(matches), fn=len(truth)-len(matches),
        correct_class=int(correct), by_class=by_class)


def summarize_counts(counts):
    gt, pred = counts['n_gt'], counts['n_pred']
    return dict(recall=counts['tp']/gt if gt else None,
        false_alarm_rate=counts['fp']/gt if gt else None,
        classification_accuracy=counts['correct_class']/pred if pred else None)
