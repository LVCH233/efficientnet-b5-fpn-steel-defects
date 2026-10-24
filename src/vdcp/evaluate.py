"""Aggregate actual defect counts, never average per-image percentages."""
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from zipfile import BadZipFile
import cv2
import numpy as np
import torch
from vdcp.data import read_masks, read_records
from vdcp.inference import (predict_image, read_rgb, probability_masks, inference_settings,
                            validate_inference_settings, prediction_kwargs)
from vdcp.regions import extract_regions, region_counts, summarize_counts

COUNT_KEYS = ('n_gt','n_pred','tp','fp','fn','correct_class')


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            result.update(block)
    return result.hexdigest()


def verify_checkpoint(checkpoint, expected_hash):
    try:
        actual = sha256(checkpoint)
    except (OSError, TypeError) as error:
        raise ValueError(f'Cannot verify checkpoint: {checkpoint}') from error
    if actual != expected_hash:
        raise ValueError('Actual checkpoint SHA256 differs from cache or protocol')


def validate_postprocessing(thresholds, min_areas):
    if not isinstance(thresholds, (list, tuple)) or len(thresholds) != 4 or any(
            isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError('Expected four finite probability thresholds in [0,1]')
    if not isinstance(min_areas, (list, tuple)) or len(min_areas) != 4 or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1 for value in min_areas):
        raise ValueError('Expected four positive integer minimum areas')


def validate_protocol(protocol, checkpoint=None, root=None, inference=None):
    if not isinstance(protocol, dict) or protocol.get('locked') is not True or protocol.get('calibration_split') != 'val':
        raise ValueError('Expected a locked validation protocol')
    validate_postprocessing(protocol.get('thresholds'), protocol.get('min_areas'))
    if (protocol.get('region_iou_threshold') != .5 or protocol.get('connectivity') != 8
            or protocol.get('gt_min_area') != 1 or protocol.get('probability_dtype') != 'float16'):
        raise ValueError('Protocol requires IoU 0.5, connectivity 8, GT minimum area 1 and float16 probabilities')
    geometry = validate_inference_settings(protocol['inference']) if 'inference' in protocol else inference_settings()
    if inference is not None and geometry != validate_inference_settings(inference):
        raise ValueError('Protocol inference settings differ from checkpoint or prediction cache')
    if checkpoint is not None:
        verify_checkpoint(checkpoint, protocol.get('checkpoint_sha256'))
    if root is not None and protocol.get('validation_csv_sha256') != sha256(Path(root)/'data/splits/val.csv'):
        raise ValueError('Protocol validation manifest has changed')


def validate_cache_metadata(root, metadata):
    verify_checkpoint(metadata.get('checkpoint'), metadata.get('checkpoint_sha256'))
    split = metadata.get('split')
    if split not in ('val', 'test') or metadata.get('probability_dtype') != 'float16':
        raise ValueError('Expected validation or test cache with float16 probabilities')
    if metadata.get('split_csv_sha256') != sha256(Path(root)/'data/splits'/f'{split}.csv'):
        raise ValueError('Split manifest changed after inference')
    validate_inference_settings(metadata.get('inference'))


def validate_probabilities(probability, shape):
    if (probability.shape != tuple(shape) or probability.dtype != np.float16
            or not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1)):
        raise ValueError(f'Expected finite float16 probabilities in [0,1] with shape {tuple(shape)}')


def load_probabilities(path, shape):
    try:
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError('Expected an NPZ archive containing probabilities')
        with archive as data:
            probability = data['probabilities']
        validate_probabilities(probability, shape)
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
        raise ValueError(f'Invalid prediction cache: {path}: {error}') from error
    return probability.astype(np.float32)


def save_probabilities(path, probability, shape):
    probability = np.asarray(probability)
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError('Model returned invalid probabilities')
    packed = probability.astype(np.float16)
    validate_probabilities(packed, shape)
    temporary = None
    try:
        with NamedTemporaryFile(dir=Path(path).parent, prefix=Path(path).name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, probabilities=packed)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def postprocess_masks(probabilities, thresholds, min_areas):
    validate_postprocessing(thresholds, min_areas)
    # Calibration caches use float16; live inference must use the same values.
    probabilities = np.asarray(probabilities).astype(np.float16).astype(np.float32)
    binary = probability_masks(probabilities, thresholds)
    filtered = np.zeros_like(binary)
    for region in extract_regions(binary, min_areas):
        x1,y1,x2,y2 = region.box
        filtered[region.class_id-1,y1:y2,x1:x2] |= region.mask.astype(np.uint8)
    return filtered


def aggregate_counts(items):
    totals = {key:sum(int(item[key]) for item in items) for key in COUNT_KEYS}
    totals.update(summarize_counts(totals))
    totals['by_class'] = []
    for index in range(4):
        counts = {key:sum(int(item['by_class'][index][key]) for item in items) for key in COUNT_KEYS}
        totals['by_class'].append(dict(class_id=index+1, **counts, **summarize_counts(counts)))
    return totals


def load_model(checkpoint, device='cuda'):
    from vdcp.models import create_model
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = create_model(payload['config'], pretrained=False)
    model.load_state_dict(payload['model'])
    model.to(device).eval()
    return model, payload


def cache_predictions(root, checkpoint, split, batch_size=2, device='cuda'):
    root, checkpoint = Path(root), Path(checkpoint)
    records = read_records(root, split)
    identity = sha256(checkpoint)
    model, payload = load_model(checkpoint, device)
    geometry = inference_settings(payload.get('config', {}))
    verify_checkpoint(checkpoint, identity)
    folder = checkpoint.parent / 'predictions' / f'{split}-{identity[:12]}'
    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=identity, split=split,
        split_csv_sha256=sha256(root/'data/splits'/f'{split}.csv'), probability_dtype='float16',
        inference=dict(geometry, batch_size=batch_size), images=len(records))
    validate_inference_settings(metadata['inference'])
    meta_path = folder/'metadata.json'
    reusable = meta_path.exists()
    if reusable and json.loads(meta_path.read_text(encoding='utf-8')) != metadata:
        raise ValueError('Prediction cache identity or inference configuration changed')
    meta_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    start = time.perf_counter()
    for index, row in enumerate(records,start=1):
        output = folder/(Path(row['image_id']).stem+'.npz')
        shape = (4, int(row['height']), int(row['width']))
        try:
            if not reusable:
                raise ValueError('Existing predictions have no verified cache metadata')
            load_probabilities(output, shape)
        except ValueError:
            image = read_rgb(root/'data/raw/train_images'/row['image_id'])
            probability = predict_image(model,image,device=device,batch_size=batch_size, **prediction_kwargs(geometry))
            save_probabilities(output, probability, shape)
        if index%100==0:
            print(f'{split} cached {index}/{len(records)} images, {time.perf_counter()-start:.1f}s',flush=True)
    del model
    verify_checkpoint(checkpoint, identity)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return folder


def score_cache(root, folder, protocol, output):
    root, folder, output = Path(root), Path(folder), Path(output)
    metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    validate_cache_metadata(root, metadata)
    validate_protocol(protocol, checkpoint=metadata['checkpoint'], root=root, inference=metadata['inference'])
    if metadata['checkpoint_sha256'] != protocol['checkpoint_sha256']:
        raise ValueError('Protocol belongs to a different checkpoint')
    split = metadata['split']
    records = read_records(root,split)
    results, pixel_gt, pixel_pred, intersection = [], np.zeros(4), np.zeros(4), np.zeros(4)
    positive_dice = [[] for _ in range(4)]
    image_exact, image_gt_positive, image_tp, image_fp = 0, np.zeros(4), np.zeros(4), np.zeros(4)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'per_image.jsonl').open('w',encoding='utf-8') as stream:
        for index,row in enumerate(records,start=1):
            truth = read_masks(root,row['image_id'])
            probability = load_probabilities(folder/(Path(row['image_id']).stem+'.npz'), truth.shape)
            pred = postprocess_masks(probability,protocol['thresholds'],protocol['min_areas'])
            counts = region_counts(truth,pred,threshold=0.5)
            gt = truth.sum(axis=(1,2)).astype(np.float64)
            pp = pred.sum(axis=(1,2)).astype(np.float64)
            ii = (truth.astype(bool)&pred.astype(bool)).sum(axis=(1,2)).astype(np.float64)
            pixel_gt += gt; pixel_pred += pp; intersection += ii
            dice = [float(2*ii[c]/(gt[c]+pp[c])) if gt[c] else None for c in range(4)]
            for c in range(4):
                if dice[c] is not None: positive_dice[c].append(dice[c])
            actual, estimated = gt>0, pp>0
            image_exact += int(np.array_equal(actual,estimated))
            image_gt_positive += actual; image_tp += actual&estimated; image_fp += (~actual)&estimated
            sample = dict(image_id=row['image_id'],**counts,**summarize_counts(counts),positive_class_dice=dice)
            stream.write(json.dumps(sample)+'\n'); results.append(counts)
            if index%200==0: print(f'Scored {index}/{len(records)} {split} images',flush=True)
    summary = aggregate_counts(results)
    summary.update(split=split,images=len(records),checkpoint_sha256=metadata['checkpoint_sha256'],
        protocol=protocol,region_definition='8-connected per-class masks; GT min area 1; class-agnostic maximum-cardinality IoU>=0.5',
        global_pixel_dice=(2*intersection/np.maximum(pixel_gt+pixel_pred,1)).tolist(),
        global_pixel_iou=(intersection/np.maximum(pixel_gt+pixel_pred-intersection,1)).tolist(),
        positive_image_class_mean_dice=[float(np.mean(values)) if values else None for values in positive_dice],
        image_level=dict(exact_match_accuracy=image_exact/len(records),gt_positive=image_gt_positive.tolist(),
            true_positive=image_tp.tolist(),false_positive=image_fp.tolist()),
        target_checks=dict(recall_ge_98=summary['recall'] is not None and summary['recall']>=.98,
            false_alarm_le_10=summary['false_alarm_rate'] is not None and summary['false_alarm_rate']<=.10,
            classification_ge_90=summary['classification_accuracy'] is not None and summary['classification_accuracy']>=.90))
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def draw_prediction(image,masks):
    result = image.copy()
    colors = [(55,117,186),(66,148,158),(182,67,66),(154,77,142)]
    for region in extract_regions(masks):
        x1,y1,x2,y2 = region.box
        color = colors[region.class_id-1]
        crop = result[y1:y2,x1:x2]
        crop[region.mask] = (0.5*crop[region.mask]+0.5*np.asarray(color)).astype(np.uint8)
        cv2.rectangle(result,(x1,y1),(x2-1,y2-1),color,1)
        cv2.putText(result,f'C{region.class_id}',(x1,max(y1-3,12)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    return result


def benchmark(root,checkpoint,protocol,output,device='cuda',samples=30,batch_size=2):
    root,output = Path(root),Path(output)
    validate_protocol(protocol, checkpoint=checkpoint, root=root)
    records = read_records(root,'val')[:samples]
    model,payload = load_model(checkpoint,device)
    geometry = inference_settings(payload.get('config', {}))
    validate_protocol(protocol, inference=geometry)
    first = read_rgb(root/'data/raw/train_images'/records[0]['image_id'])
    for _ in range(3): predict_image(model,first,device=device,batch_size=batch_size, **prediction_kwargs(geometry))
    output.mkdir(parents=True,exist_ok=True)
    elapsed = []
    for row in records:
        if device=='cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        image = read_rgb(root/'data/raw/train_images'/row['image_id'])
        probability = predict_image(model,image,device=device,batch_size=batch_size, **prediction_kwargs(geometry))
        masks = postprocess_masks(probability,protocol['thresholds'],protocol['min_areas'])
        overlay = draw_prediction(image,masks)
        encoded = cv2.imencode('.png',cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))[1]
        encoded.tofile(output/(Path(row['image_id']).stem+'.png'))
        if device=='cuda': torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start)
    result = dict(samples=len(elapsed),warmup=3,seconds=elapsed,median_seconds=float(np.median(elapsed)),
        checkpoint_sha256=protocol['checkpoint_sha256'],postprocessing_protocol=protocol,
        p95_seconds=float(np.percentile(elapsed,95)),max_seconds=float(max(elapsed)),
        inference=dict(geometry, batch_size=batch_size),
        scope='Read original image, normalize, overlapping sliding windows, inference, fusion, regions, overlay and PNG write',
        device=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',torch_version=torch.__version__,
        industrial_full_plate_realtime_verified=False)
    (output/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
