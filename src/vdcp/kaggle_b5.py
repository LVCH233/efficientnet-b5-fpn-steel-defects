"""Pinned historical B5-FPN adaptation, isolated from existing experiments."""
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from vdcp.vendor.kh_fpn.encoders.efficientnet import efficientnet_encoders
from vdcp.vendor.kh_fpn.fpn.decoder import FPNDecoder


class HistoricalFPN(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        entry = efficientnet_encoders['efficientnet-b5']
        self.encoder = entry['encoder'](in_channels='r', **entry['params'])
        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            self.encoder.load_state_dict(state, strict=True)
        self.decoder = FPNDecoder(entry['out_shapes'], pyramid_channels=256,
                                  segmentation_channels=128, final_channels=4,
                                  dropout=.2, merge_policy='cat')

    def forward(self, images):
        return self.decoder(self.encoder(images))


def original_tensor(bgr):
    """Preserve the original dataset's OpenCV channel-zero convention."""
    normalized = (bgr[:, :, 0].astype(np.float32) / 255 - .3439) / .0383
    return torch.from_numpy(np.ascontiguousarray(normalized[None]))


def original_loss(logits, target):
    logits, target = logits.float(), target.float()
    probability = logits.sigmoid()
    dice = 2 * (probability * target).sum() / (probability.sum() + target.sum() + 1e-7)
    return .6 * F.binary_cross_entropy_with_logits(logits, target) + .4 * (1 - dice)


def verify_resume_identity(checkpoint, identity):
    if checkpoint.get('identity') != identity:
        raise ValueError('Resume identity differs: source, config, weights or data changed')


def can_start_epoch(remaining, epoch_seconds, validation_seconds):
    return remaining >= epoch_seconds * 1.15 + validation_seconds * 1.15 + 120


def check_records(train, val):
    def ids(rows):
        names = [row['image_id'] for row in rows]
        if len(names) != len(set(names)) or any(Path(x).name != x or '/' in x or '\\' in x for x in names):
            raise ValueError('Duplicate or unsafe image identifier')
        return set(names)
    if ids(train) & ids(val):
        raise ValueError('Training and validation images overlap')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_identity(root, config):
    root = Path(root)
    files = sorted((root / 'src').rglob('*.py')) + [root / 'scripts/train_kaggle_b5.py']
    source = hashlib.sha256()
    for file in files:
        source.update(file.relative_to(root).as_posix().encode())
        source.update(file.read_bytes())
    from vdcp.data import read_records
    selected = read_records(root,'train') + read_records(root,'val')
    check_records(read_records(root,'train'),read_records(root,'val'))
    data_digest = hashlib.sha256()
    for row in sorted(selected,key=lambda x:x['image_id']):
        for relative in (f"data/raw/train_images/{row['image_id']}",
                         f"data/raw/masks/{Path(row['image_id']).stem}.png"):
            data_digest.update(relative.encode())
            data_digest.update(file_sha(root/relative).encode())
    test_manifest = root/'data/splits/test.csv'
    internal_test_sha = file_sha(test_manifest) if test_manifest.exists() else json.loads(
        (root/'bundle_manifest.json').read_text())['internal_test_csv_sha256']
    return {'config': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            'splits': {name: file_sha(root / 'data/splits' / (name + '.csv')) for name in ('train', 'val')},
            'source': source.hexdigest(),
            'weights': file_sha(root / config['pretrained_file']),
            'selected_data': data_digest.hexdigest(),
            'reference_protocol': file_sha(root/config['validation_reference_protocol']),
            'internal_test_csv': internal_test_sha}


class OriginalCrops(Dataset):
    def __init__(self, root, records, width=416):
        self.root, self.records, self.width = Path(root), records, width

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from vdcp.data import read_masks
        row = self.records[index]
        image = cv2.imread(str(self.root / 'data/raw/train_images' / row['image_id']))
        if image is None:
            raise ValueError('Cannot read training image')
        mask = read_masks(self.root, row['image_id'])
        left = random.randint(0, image.shape[1] - self.width)
        if random.random() < .5:
            occupied = np.argwhere(mask.any(axis=0))
            if len(occupied):
                center = occupied[random.randrange(len(occupied)), 1]
                left = min(max(int(center) - random.randrange(self.width), 0), image.shape[1] - self.width)
        image = image[:, left:left + self.width, :1].copy()
        mask = mask[:, :, left:left + self.width].copy()
        effect = random.choices(('clahe', 'blur', 'sharpen'), weights=(.5, .3, .3))[0]
        if effect == 'clahe':
            image[:, :, 0] = cv2.createCLAHE(clipLimit=random.uniform(1, 4), tileGridSize=(8, 8)).apply(image[:, :, 0])
        elif effect == 'blur':
            image = cv2.GaussianBlur(image[:, :, 0], (3, 3), 0)[:, :, None]
        else:
            # OpenCV equivalent of imgaug's alpha blend; an explicitly recorded adaptation.
            alpha = random.uniform(.2, .3)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32)
            sharpened = cv2.filter2D(image[:, :, 0].astype(np.float32), -1, kernel)
            image = np.clip((1-alpha)*image[:, :, 0] + alpha*sharpened, 0, 255).astype(np.uint8)[:, :, None]
        if random.random() < .5:
            mode = random.choice((-1, 0, 1))
            if mode in (-1, 1):
                image, mask = image[:, ::-1], mask[:, :, ::-1]
            if mode in (-1, 0):
                image, mask = image[::-1], mask[:, ::-1]
        return original_tensor(image), torch.from_numpy(np.ascontiguousarray(mask)).float()


@torch.inference_mode()
def original_probability(model, bgr, device, on_fp32_retry=None):
    tensor = original_tensor(bgr).unsqueeze(0).to(device)
    device_type = torch.device(device).type
    amp = device_type == 'cuda'
    with torch.autocast(device_type=device_type, enabled=amp):
        logits = model(tensor)
    if not torch.isfinite(logits).all():
        if not amp:
            raise FloatingPointError('Nonfinite validation prediction in FP32')
        del logits
        # Recompute the whole image: casting already-overflowed logits cannot repair them.
        with torch.autocast(device_type=device_type, enabled=False):
            logits = model(tensor)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite validation prediction after FP32 retry')
        if on_fp32_retry is not None:
            on_fp32_retry()
        else:
            print(json.dumps({'phase': 'validation_fp32_retry'}), flush=True)
    probability = logits[0].float().sigmoid().cpu().numpy()
    if not np.isfinite(probability).all():
        raise FloatingPointError('Nonfinite validation prediction')
    return probability
