"""Original-resolution probability fusion with complete edge coverage."""
from pathlib import Path
from numbers import Integral
import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def validate_inference_settings(settings):
    """Return prediction-affecting settings; batch size is execution metadata."""
    keys = ('window', 'stride', 'height_window', 'height_stride', 'amp')
    if not isinstance(settings, dict) or any(key not in settings for key in keys):
        raise ValueError('Inference settings must declare window, strides, height window and AMP')
    if set(settings) - set(keys) - {'batch_size'}:
        raise ValueError('Unknown inference setting')
    for key in keys[:-1]:
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError('Inference dimensions and strides must be positive integers')
    if (settings['stride'] > settings['window'] or settings['height_window'] != 256
            or settings['height_stride'] > 256 or not isinstance(settings['amp'], bool)):
        raise ValueError('Inference requires stride <= window, height window 256 and boolean AMP')
    if 'batch_size' in settings and (isinstance(settings['batch_size'], bool)
            or not isinstance(settings['batch_size'], Integral) or settings['batch_size'] <= 0):
        raise ValueError('Inference batch size must be a positive integer')
    return {key: settings[key] for key in keys}


def inference_settings(config=None):
    """Resolve checkpoint geometry, retaining the historical 512/384 defaults."""
    config = {} if config is None else config
    if not isinstance(config, dict):
        raise ValueError('Checkpoint configuration must be a dictionary')
    settings = dict(window=config.get('inference_window', config.get('crop_width', 512)),
                    stride=config.get('inference_stride', config.get('validation_stride', 384)),
                    height_window=256, height_stride=192, amp=config.get('amp', True))
    if 'inference' in config:
        if not isinstance(config['inference'], dict):
            raise ValueError('Explicit inference settings must be a dictionary')
        settings.update(config['inference'])
    return validate_inference_settings(settings)


def prediction_kwargs(settings):
    """Convert validated fixed-height metadata to predict_image arguments."""
    return {key: value for key, value in validate_inference_settings(settings).items()
            if key != 'height_window'}


def read_rgb(path):
    data = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError(f'Cannot decode image: {path}')
    return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


def image_tensor(image):
    normalized = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


def window_starts(length, window, stride):
    if length <= 0 or window <= 0 or stride <= 0 or stride > window:
        raise ValueError('Positive dimensions and stride <= window are required')
    if length <= window:
        return [0]
    positions = list(range(0, length-window+1, stride))
    if positions[-1] != length-window:
        positions.append(length-window)
    return positions


@torch.inference_mode()
def predict_image(model, image, device='cuda', window=512, stride=384, batch_size=2, amp=True,
                  height_stride=192):
    height, width = image.shape[:2]
    if batch_size <= 0:
        raise ValueError('Batch size must be positive')
    padded_height = max(256, height)
    padded_width = max(window, width)
    padded = np.pad(image, ((0,padded_height-height),(0,padded_width-width),(0,0)), mode='edge')
    positions = [(top, left) for top in window_starts(padded_height, 256, height_stride)
                 for left in window_starts(padded_width, window, stride)]
    total = np.zeros((4, padded_height, padded_width), dtype=np.float32)
    coverage = np.zeros((padded_height, padded_width), dtype=np.float32)
    model.eval()
    for offset in range(0, len(positions), batch_size):
        selected = positions[offset:offset+batch_size]
        batch = torch.stack([image_tensor(padded[top:top+256, left:left+window])
                             for top, left in selected]).to(device)
        with torch.autocast(device_type=torch.device(device).type, enabled=amp and torch.device(device).type=='cuda'):
            logits = model(batch)
        probabilities = logits.float().sigmoid().cpu().numpy()
        for probability, (top, left) in zip(probabilities, selected):
            total[:, top:top+256, left:left+window] += probability
            coverage[top:top+256, left:left+window] += 1
    if np.any(coverage == 0):
        raise RuntimeError('Sliding windows left uncovered pixels')
    return (total / coverage[None])[:, :height, :width]


def probability_masks(probabilities, thresholds=(0.5,0.5,0.5,0.5)):
    return (probabilities >= np.asarray(thresholds)[:,None,None]).astype(np.uint8)
