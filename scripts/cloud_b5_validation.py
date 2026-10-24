"""Bounded, batched native FP32 validation using the immutable 768 protocol.

Only validation records and train_images/masks are read. CPU workers prepare
unchanged native inputs; independent region jobs retain frozen postprocessing.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
ENGINE_SHA = 'd061e49271210ede89233139f00a85d7588f5c7be52bee510f88157a9ca64979'
_path = ROOT / 'scripts/continue_b5_local_768.py'
if hashlib.sha256(_path.read_bytes()).hexdigest() != ENGINE_SHA:
    raise ValueError('Frozen 768 validation engine changed')
_spec = importlib.util.spec_from_file_location('cloud_validation_frozen_engine', _path)
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)

from vdcp.data import read_masks
from vdcp.evaluate import aggregate_counts, validate_postprocessing
from vdcp.kaggle_b5 import original_tensor
from vdcp.regions import region_counts


def _worker_init(_):
    cv2.setNumThreads(1)


class NativeValidation(Dataset):
    def __init__(self, root, records, expected_shape=(256, 1600)):
        self.root, self.records, self.expected_shape = Path(root), records, expected_shape
        names = [row['image_id'] for row in records]
        if len(names) != len(set(names)) or any(
                not name or name in ('.', '..') or Path(name).name != name
                or '/' in name or '\\' in name for name in names):
            raise ValueError('Duplicate or unsafe validation image identifier')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        name = self.records[index]['image_id']
        image = cv2.imread(str(self.root / 'data/raw/train_images' / name))
        if image is None or image.shape[:2] != self.expected_shape:
            raise ValueError('Full native validation image required')
        target = read_masks(self.root, name)
        if target.shape != (4, *self.expected_shape):
            raise ValueError('Full native four-class validation mask required')
        return index, original_tensor(image), target


def _score(job):
    probability, target, thresholds, min_areas = job
    masks = engine.fp32_postprocess(probability, thresholds, min_areas)
    counts = region_counts(target, masks)
    binary = probability >= .5
    intersection = (binary & target.astype(bool)).sum(axis=(1, 2))
    denominator = binary.sum(axis=(1, 2)) + target.sum(axis=(1, 2))
    dice = np.divide(2 * intersection, denominator, out=np.ones(4), where=denominator > 0).sum()
    return counts, float(dice)


def _validate_records(model, root, records, protocol, device, batch_size=8,
                      workers=4, progress=None, expected_shape=(256, 1600)):
    """Diagnostic core; production callers must use strict ``validate`` below."""
    if type(batch_size) is not int or batch_size < 1 or type(workers) is not int or workers < 0:
        raise ValueError('Positive batch size and nonnegative worker count required')
    if not records:
        raise ValueError('Validation requires records')
    validate_postprocessing(protocol['thresholds'], protocol['min_areas'])
    state = engine.capture_rng(device)
    modes = [(layer, layer.training) for layer in model.modules()]
    started = time.monotonic()
    try:
        dataset = NativeValidation(root, records, expected_shape)
        settings = dict(batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=torch.device(device).type == 'cuda', drop_last=False,
                        worker_init_fn=_worker_init, generator=torch.Generator().manual_seed(0))
        if workers:
            settings['prefetch_factor'] = 2
            settings['multiprocessing_context'] = 'spawn'
        loader = DataLoader(dataset, **settings)
        model.eval()
        counts, dice_sum, completed = [], 0., 0
        with torch.inference_mode(), ThreadPoolExecutor(max_workers=min(8, batch_size)) as pool:
            for indices, inputs, targets in loader:
                expected = list(range(completed, completed + len(indices)))
                if indices.tolist() != expected:
                    raise RuntimeError('Validation image order or coverage changed')
                with torch.autocast(torch.device(device).type, enabled=False):
                    logits = model(inputs.to(device, non_blocking=True))
                    if logits.dtype != torch.float32 or logits.shape != (len(inputs), 4, *expected_shape):
                        raise ValueError('Expected native four-class FP32 validation logits')
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError('Nonfinite FP32 validation logits')
                    probabilities = logits.sigmoid().cpu().numpy()
                jobs = ((p, y, protocol['thresholds'], protocol['min_areas'])
                        for p, y in zip(probabilities, targets.numpy()))
                for count, dice in pool.map(_score, jobs):
                    counts.append(count); dice_sum += dice
                completed += len(indices)
                if progress is not None:
                    progress(dict(phase='validation', images=completed, total_images=len(records),
                                  seconds=time.monotonic() - started))
                del logits, probabilities
        if completed != len(records):
            raise RuntimeError('Incomplete validation coverage')
        result = aggregate_counts(counts)
        result.update(region_f2=5 * result['tp'] / max(5 * result['tp'] + 4 * result['fn'] + result['fp'], 1),
            images=completed, kaggle_style_mean_dice_at_05=dice_sum / (4 * completed),
            seconds=time.monotonic() - started, reference_protocol_sha256=engine.PROTOCOL_SHA,
            canonical_precision='fp32_direct', probability_dtype='float32',
            inference_precision_policy='FP32_forward_sigmoid_direct_threshold_no_FP16_round_trip',
            derived_protocol=True, threshold_source='frozen reference; precision override FP32',
            inference=dict(window=1600, stride=1600, height=256, amp=False, batch_size=batch_size),
            test_accessed=False)
        return result
    finally:
        for layer, training in modes:
            layer.training = training
        engine.restore_rng(state, device)


def validate(model, root, records, base_config, device, batch_size=8, workers=4, progress=None):
    """Score all 1885 native validation images with the locked FP32 protocol."""
    if len(records) != 1885:
        raise ValueError('Every epoch requires complete 1885-image validation')
    path = Path(root) / base_config['validation_reference_protocol']
    if engine.sha(path) != engine.PROTOCOL_SHA:
        raise ValueError('Frozen threshold reference changed')
    protocol = json.loads(path.read_text(encoding='utf-8'))
    return _validate_records(model, root, records, protocol, device,
                             batch_size=batch_size, workers=workers, progress=progress)
