"""Training-only difficult background crops and explicit region error diagnostics."""
import json
from pathlib import Path
import cv2
import numpy as np
from vdcp.regions import extract_regions, region_iou, match_iou_matrix, region_counts


def hard_negative_windows(probability, truth, width=512, threshold=.6, min_pixels=64, margin=4, max_windows=2):
    probability, truth = np.asarray(probability), np.asarray(truth)
    if (probability.shape != truth.shape or truth.ndim != 3 or truth.shape[0] != 4
            or not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1)
            or width < 1 or width > truth.shape[2] or not 0 < threshold < 1
            or min_pixels < 1 or margin < 0 or max_windows < 1):
        raise ValueError('Invalid hard-background probability, mask or crop settings')
    union = truth.any(axis=0).astype(np.uint8)
    protected = cv2.dilate(union, np.ones((2*margin+1, 2*margin+1), np.uint8)) > 0
    confidence = probability.max(axis=0)
    false = (confidence >= threshold) & ~protected
    packed = np.zeros_like(truth, dtype=np.uint8)
    packed[0] = false
    regions = sorted(extract_regions(packed, [min_pixels]*4), key=lambda r: (-r.area, r.box))
    windows = []
    for region in regions:
        center = (region.box[0]+region.box[2])//2
        left = min(max(center-width//2, 0), truth.shape[2]-width)
        if any(abs(left-item['left']) < width//4+1 for item in windows):
            continue
        crop = false[:, left:left+width]
        windows.append({'left':int(left), 'false_pixels':int(crop.sum()),
                        'mean_false_confidence':float(confidence[:,left:left+width][crop].mean())})
        if len(windows) == max_windows:
            break
    return windows


def load_hard_manifest(path, records, train_sha, parent_sha, width):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if (manifest.get('split') != 'train' or manifest.get('train_csv_sha256') != train_sha
            or manifest.get('parent_checkpoint_sha256') != parent_sha or manifest.get('crop_width') != width):
        raise ValueError('Hard-background manifest identity differs from training configuration')
    expected=dict(width=width,threshold=.6,min_pixels=64,margin=4,max_windows=2)
    if manifest.get('settings')!=expected:
        raise ValueError('Hard-background mining settings differ from the approved experiment')
    allowed = {row['image_id']:int(row['width']) for row in records}
    result = {}
    for row in manifest.get('records', []):
        identity = row['image_id']
        if identity not in allowed or identity in result:
            raise ValueError('Hard-background manifest contains non-training or duplicate image ID')
        lefts = [item['left'] for item in row['windows']]
        if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= allowed[identity]-width for x in lefts):
            raise ValueError('Hard-background crop is outside image bounds')
        if len(lefts)>expected['max_windows'] or len(set(lefts))!=len(lefts):
            raise ValueError('Hard-background candidates must be distinct and at most two per image')
        result[identity] = lefts
    if set(result) != set(allowed):
        raise ValueError('Hard-background manifest is missing training image IDs')
    return result


def diagnose_regions(truth, prediction, raw_prediction=None):
    actual, estimated = extract_regions(truth), extract_regions(prediction)
    matrix = np.array([[region_iou(a,p) for p in estimated] for a in actual],dtype=float).reshape(len(actual),len(estimated))
    matches = match_iou_matrix(matrix, .5)
    matched_gt, matched_pred = {i for i,j in matches}, {j for i,j in matches}
    fp = dict(background=0, low_iou=0, matching_conflict=0)
    fn = dict(no_overlap=0, low_iou=0, matching_conflict=0, area_filtered=0)
    raw = extract_regions(raw_prediction) if raw_prediction is not None else []
    for j in set(range(len(estimated)))-matched_pred:
        maximum = float(matrix[:,j].max()) if actual else 0
        fp['background' if maximum == 0 else 'low_iou' if maximum < .5 else 'matching_conflict'] += 1
    for i in set(range(len(actual)))-matched_gt:
        maximum = float(matrix[i].max()) if estimated else 0
        raw_max = max((region_iou(actual[i],p) for p in raw),default=0)
        reason = ('matching_conflict' if maximum >= .5 else 'area_filtered' if raw_max >= .5
                  else 'no_overlap' if maximum == 0 else 'low_iou')
        fn[reason] += 1
    return {'counts':region_counts(truth,prediction), 'fp_reasons':fp, 'fn_reasons':fn,
            'fragmentation_candidates':int(((matrix>0).sum(axis=1)>1).sum()),
            'merge_candidates':int(((matrix>0).sum(axis=0)>1).sum())}
