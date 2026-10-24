import importlib
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def driver():
    path = Path(__file__).resolve().parents[1] / 'scripts/cloud_b5_validation.py'
    assert path.is_file(), 'Separate cloud full validation module is required'
    return importlib.import_module('cloud_b5_validation')


@pytest.fixture
def data(tmp_path):
    images = tmp_path / 'data/raw/train_images'
    masks = tmp_path / 'data/raw/masks'
    images.mkdir(parents=True); masks.mkdir(parents=True)
    records = []
    for index in range(9):
        image = np.zeros((4, 7, 3), np.uint8)
        image[:, :, 0] = 80 + index * 3
        image[index % 4, :, 0] = 180
        image[:, :, 1:] = 255
        packed = np.zeros((4, 7), np.uint8)
        packed[index % 4, :] = 1 << (index % 4)
        name = f'{index}.png'
        assert cv2.imwrite(str(images / name), image)
        assert cv2.imwrite(str(masks / name), packed)
        records.append({'image_id': name})
    return tmp_path, records, dict(thresholds=[.3, .4, .6, .7], min_areas=[1, 2, 3, 4])


class Toy(torch.nn.Module):
    def __init__(self, invalid=None):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.invalid = invalid
        self.inputs = []

    def forward(self, x):
        assert torch.is_inference_mode_enabled()
        assert x.dtype == torch.float32
        assert not torch.is_autocast_enabled(x.device.type)
        self.inputs.append(x.cpu().clone())
        random.random(); np.random.random(); torch.rand(())
        logits = x.repeat(1, 4, 1, 1) - torch.arange(4)[None, :, None, None]
        if self.invalid == 'nonfinite': logits[0, 0, 0, 0] = float('inf')
        if self.invalid == 'shape': logits = logits[:, :3]
        if self.invalid == 'half': logits = logits.half()
        return logits


def run(module, model, data, batch_size=4, workers=0, progress=None):
    root, records, protocol = data
    return module._validate_records(model, root, records, protocol, 'cpu',
        batch_size=batch_size, workers=workers, progress=progress, expected_shape=(4, 7))


def stable(metrics):
    return {key: value for key, value in metrics.items() if key not in ('seconds', 'inference')}


def test_batches_match_frozen_per_image_metrics_without_resizing_or_mutation(data):
    m = driver()
    from vdcp.data import read_masks
    from vdcp.evaluate import aggregate_counts
    from vdcp.kaggle_b5 import original_tensor
    from vdcp.regions import region_counts
    model = Toy(); events = []
    got = run(m, model, data, progress=events.append)
    assert [len(x) for x in model.inputs] == [4, 4, 1]
    expected_inputs = torch.stack([original_tensor(cv2.imread(str(data[0] / 'data/raw/train_images' / row['image_id']))) for row in data[1]])
    assert torch.equal(torch.cat(model.inputs), expected_inputs)
    assert model.training and model.bn.training
    reference_model = Toy()
    one = run(m, reference_model, data, batch_size=1)
    assert stable(got) == stable(one)
    counts, dice = [], 0.
    with torch.inference_mode():
        probability = (expected_inputs.repeat(1, 4, 1, 1) - torch.arange(4)[None, :, None, None]).sigmoid().numpy()
    for row, p in zip(data[1], probability):
        target = read_masks(data[0], row['image_id'])
        counts.append(region_counts(target, m.engine.fp32_postprocess(p, **data[2])))
        binary = p >= .5
        denominator = binary.sum((1, 2)) + target.sum((1, 2))
        dice += np.divide(2*(binary & target.astype(bool)).sum((1, 2)), denominator, out=np.ones(4), where=denominator > 0).sum()
    expected = aggregate_counts(counts)
    for key, value in expected.items(): assert got[key] == value
    assert got['kaggle_style_mean_dice_at_05'] == dice / 36
    assert got['inference'] == dict(window=1600, stride=1600, height=256, amp=False, batch_size=4)
    assert got['probability_dtype'] == 'float32' and got['test_accessed'] is False
    assert events[-1]['images'] == events[-1]['total_images'] == 9
    assert all(set(event) == {'phase', 'images', 'total_images', 'seconds'} for event in events)


@pytest.mark.parametrize('invalid,error', [('nonfinite', FloatingPointError), ('shape', ValueError), ('half', ValueError), (None, None)])
def test_rng_and_mixed_module_modes_preserved_on_success_and_errors(data, invalid, error):
    m = driver(); model = Toy(invalid); model.bn.eval()
    before = m.engine.capture_rng('cpu')
    if error:
        with pytest.raises(error): run(m, model, data)
    else: run(m, model, data)
    assert m.engine.states_equal(before, m.engine.capture_rng('cpu'))
    assert model.training and not model.bn.training


def test_public_validation_requires_full_native_images_and_frozen_protocol(data):
    m = driver(); root, records, _ = data
    with pytest.raises(ValueError, match='1885'):
        m.validate(Toy(), root, records, {}, 'cpu')
    with pytest.raises(ValueError, match='native'):
        m.NativeValidation(root, records)[0]
    protocol = root / 'protocol.json'; protocol.write_text('{}')
    with pytest.raises(ValueError, match='reference'):
        m.validate(Toy(), root, [{'image_id': f'{i}.png'} for i in range(1885)],
                   {'validation_reference_protocol': 'protocol.json'}, 'cpu')


@pytest.mark.parametrize('name', ['../test.png', 'a/b.png', r'a\b.png'])
def test_dataset_rejects_unsafe_identifiers(data, name):
    m = driver()
    with pytest.raises(ValueError): m.NativeValidation(data[0], [{'image_id': name}])


def test_workers_deliver_every_image_once_and_preserve_rng(data):
    m = driver(); model = Toy(); before = m.engine.capture_rng('cpu')
    got = run(m, model, data, workers=2)
    assert got['images'] == 9 and [len(x) for x in model.inputs] == [4, 4, 1]
    assert m.engine.states_equal(before, m.engine.capture_rng('cpu'))
    assert stable(got) == stable(run(m, Toy(), data, workers=0))


def test_direct_fp32_threshold_does_not_round_probabilities_to_half():
    m = driver()
    probability = np.full((4, 2, 3), np.float32(.49999), np.float32)
    target = np.zeros((4, 2, 3), np.uint8)
    counts, dice = m._score((probability, target, [.5]*4, [1]*4))
    assert counts['n_pred'] == 0 and dice == 4
    assert probability.dtype == np.float32 and np.all(probability < .5)
    assert np.all(probability.astype(np.float16) == .5)


def test_progress_failure_preserves_rng_and_modes(data):
    m = driver(); model = Toy(); model.bn.eval()
    before = m.engine.capture_rng('cpu')
    def fail(_):
        raise RuntimeError('progress unavailable')
    with pytest.raises(RuntimeError, match='progress unavailable'):
        run(m, model, data, progress=fail)
    assert m.engine.states_equal(before, m.engine.capture_rng('cpu'))
    assert model.training and not model.bn.training


@pytest.mark.parametrize('batch_size,workers', [(0, 0), (True, 0), (2, -1), (2, True)])
def test_invalid_loader_settings_rejected(data, batch_size, workers):
    m = driver()
    with pytest.raises(ValueError): run(m, Toy(), data, batch_size=batch_size, workers=workers)
