"""Validation-only classwise calibration; final metrics use global matching."""
import csv
import json
from pathlib import Path
import numpy as np
from vdcp.data import read_masks, read_records
from vdcp.evaluate import sha256, load_probabilities, validate_cache_metadata, validate_postprocessing, validate_protocol
from vdcp.regions import extract_regions, match_iou_matrix, region_iou
from vdcp.inference import validate_inference_settings


def select_candidate(candidates):
    scored = []
    for candidate in candidates:
        row = dict(candidate)
        gt, pred, tp = row['n_gt'],row['n_pred'],row['tp']
        row['recall'] = tp/gt if gt else None
        row['false_alarm_rate'] = row['fp']/gt if gt else None
        row['classwise_precision'] = tp/pred if pred else None
        denominator = 5*tp+4*row['fn']+row['fp']
        row['f2'] = 5*tp/denominator if denominator else 0.0
        row['meets_proxy_targets'] = bool(gt and pred and row['recall']>=.98
            and row['false_alarm_rate']<=.10 and row['classwise_precision']>=.90)
        scored.append(row)
    if not scored:
        raise ValueError('No validation candidates')
    return max(scored,key=lambda row:(row['meets_proxy_targets'],row['f2'],
        row['classwise_precision'] or 0.0,-row['fp'],row['threshold']))


def calibrate(root,folder,output,thresholds=(.15,.25,.35,.45,.55,.65,.75,.85,.95),areas=(1,16,64,256,1024,4096)):
    root,folder,output = Path(root),Path(folder),Path(output)
    metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    validate_cache_metadata(root, metadata)
    if metadata['split'] != 'val':
        raise ValueError('Threshold calibration is restricted to the validation split')
    if metadata['split_csv_sha256'] != sha256(root/'data/splits/val.csv'):
        raise ValueError('Validation manifest has changed')
    for threshold in thresholds:
        for area in areas:
            validate_postprocessing([threshold]*4, [area]*4)
    grid = [[dict(class_id=channel+1,threshold=float(threshold),min_area=int(area),
        n_gt=0,n_pred=0,tp=0,fp=0,fn=0) for threshold in thresholds for area in areas]
        for channel in range(4)]
    records = read_records(root,'val')
    for number,row in enumerate(records,start=1):
        truth = read_masks(root,row['image_id'])
        actual_regions = extract_regions(truth)
        probability = load_probabilities(folder/(Path(row['image_id']).stem+'.npz'), truth.shape)
        binary = np.zeros_like(truth)
        for channel in range(4):
            actual = [region for region in actual_regions if region.class_id==channel+1]
            for threshold_index,threshold in enumerate(thresholds):
                binary.fill(0)
                binary[channel] = probability[channel]>=threshold
                predicted = extract_regions(binary)
                full_iou = np.array([[region_iou(first,second) for second in predicted]
                                     for first in actual],dtype=float).reshape(len(actual),len(predicted))
                for area_index,area in enumerate(areas):
                    kept = [index for index,region in enumerate(predicted) if region.area>=area]
                    matches = match_iou_matrix(full_iou[:,kept],.5)
                    counts = grid[channel][threshold_index*len(areas)+area_index]
                    counts['n_gt'] += len(actual); counts['n_pred'] += len(kept)
                    counts['tp'] += len(matches); counts['fp'] += len(kept)-len(matches)
                    counts['fn'] += len(actual)-len(matches)
        if number%100==0:
            print(f'Calibrated {number}/{len(records)} validation images',flush=True)
    chosen = [select_candidate(candidates) for candidates in grid]
    output.mkdir(parents=True,exist_ok=True)
    with (output/'calibration_grid.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(grid[0][0]))
        writer.writeheader()
        writer.writerows(row for candidates in grid for row in candidates)
    protocol = dict(locked=True,checkpoint_sha256=metadata['checkpoint_sha256'],
        validation_csv_sha256=metadata['split_csv_sha256'],thresholds=[row['threshold'] for row in chosen],
        min_areas=[row['min_area'] for row in chosen],region_iou_threshold=.5,connectivity=8,
        gt_min_area=1,probability_dtype=metadata['probability_dtype'],
        inference=validate_inference_settings(metadata['inference']),
        selection_method='Classwise same-class region F2; prefer feasible proxy targets; global metrics scored separately',
        calibration_split='val',selected_classwise_proxy_results=chosen,
        threshold_grid=list(thresholds),min_area_grid=list(areas))
    validate_protocol(protocol, checkpoint=metadata['checkpoint'], root=root)
    (output/'locked_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    return protocol
