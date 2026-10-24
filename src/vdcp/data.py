"""Deterministic image manifests and mask-preserving training crops."""
import csv
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from vdcp.inference import image_tensor, read_rgb


def read_records(root, split):
    with (Path(root)/'data/splits'/f'{split}.csv').open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def read_masks(root, image_id):
    path = Path(root)/'data/raw/masks'/(Path(image_id).stem+'.png')
    packed = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    if packed is None:
        raise ValueError(f'Cannot read mask: {path}')
    return np.stack([((packed >> channel)&1) for channel in range(4)]).astype(np.uint8)


def augment_long_training(image, masks):
    """Training-only image effects and shared flips, without aliasing inputs."""
    if random.random() < 0.5:
        image, masks = image[:, ::-1], masks[:, :, ::-1]
    if random.random() < 0.5:
        image, masks = image[::-1], masks[:, ::-1]
    image, masks = image.copy(), masks.copy()
    effect = random.choice(('clahe', 'blur', 'sharpen', 'identity'))
    if effect == 'clahe':
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif effect == 'blur':
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.5)
    elif effect == 'sharpen':
        blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0.5)
        image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return image, masks


class SteelCrops(Dataset):
    def __init__(self, root, records, width=512, augment=True, positive_crop_probability=0.5,
                 hard_windows=None, hard_probability=0., augmentation=None):
        self.root, self.records, self.width = Path(root), records, width
        self.augment, self.positive_probability = augment, positive_crop_probability
        if augmentation not in (None, 'long_training'):
            raise ValueError('Unsupported training augmentation')
        self.augmentation = augmentation
        if not 0 <= hard_probability <= 1:
            raise ValueError('hard_probability must be in [0,1]')
        self.hard_windows, self.hard_probability = hard_windows or {}, hard_probability

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = read_rgb(self.root/'data/raw/train_images'/row['image_id'])
        masks = read_masks(self.root,row['image_id'])
        left = random.randint(0,image.shape[1]-self.width)
        active = [channel for channel in range(4) if masks[channel].any()]
        candidates = self.hard_windows.get(row['image_id'], [])
        hard = candidates and random.random() < self.hard_probability
        if hard:
            left = random.choice(candidates)
        elif active and random.random()<self.positive_probability:
            channel = random.choice(active)
            xs = np.flatnonzero(masks[channel].any(axis=0))
            center = int(random.choice(xs))
            left = min(max(center-random.randint(0,self.width-1),0),image.shape[1]-self.width)
        image, masks = image[:,left:left+self.width], masks[:,:,left:left+self.width]
        if self.augment and self.augmentation == 'long_training':
            image, masks = augment_long_training(image, masks)
        elif self.augment:
            if random.random()<0.5:
                image, masks = image[:,::-1], masks[:,:,::-1]
            if random.random()<0.5:
                image, masks = image[::-1], masks[:,::-1]
            if random.random()<0.5:
                image = np.clip(image.astype(np.float32)*random.uniform(0.85,1.15)+random.uniform(-12,12),0,255).astype(np.uint8)
        return image_tensor(image), torch.from_numpy(np.ascontiguousarray(masks)).float()


def sampling_weights(records, power=1., max_ratio=None):
    if not np.isfinite(power) or not 0 < power <= 1 or (max_ratio is not None and (not np.isfinite(max_ratio) or max_ratio < 1)):
        raise ValueError('Invalid sampling strength or maximum weight ratio')
    labels = np.array([[int(row[f'c{i}']) for i in range(1,5)] for row in records])
    counts = np.maximum(labels.sum(axis=0),1)
    negative_count = max(int((labels.sum(axis=1)==0).sum()),1)
    weights = np.maximum((labels/counts).max(axis=1),1/negative_count).astype(np.float64)**power
    return weights if max_ratio is None else np.minimum(weights, (1/negative_count)**power*max_ratio)
