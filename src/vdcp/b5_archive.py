"""B5-only canonical FP32 inference and group evaluation."""
from pathlib import Path
import hashlib
import cv2
import numpy as np
import torch
from vdcp.kaggle_b5 import HistoricalFPN, original_tensor
from vdcp.regions import extract_regions
from vdcp.group_evaluation import score_final_masks, summarize_counts, COUNT_KEYS

MODEL_SHA256 = '7e97d666c9164f4c34dfb8ef88cbc120df5862b5d7dab5fd20e510fce230ce56'
THRESHOLDS = (.75, .65, .65, .85)
MIN_AREAS = (256, 1024, 256, 1024)

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def configure_runtime():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(0)

def load_model(checkpoint, device='cpu', expected_sha=MODEL_SHA256):
    configure_runtime()
    actual = sha256(checkpoint)
    if actual != expected_sha:
        raise ValueError(f'Checkpoint SHA256 differs: {actual}')
    # Original trusted checkpoint includes Python/NumPy RNG and optimizer state.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = HistoricalFPN()
    model.load_state_dict(payload['model'], strict=True)
    return model.float().to(device).eval()

def read_bgr(path):
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if value is None or value.shape != (256, 1600, 3):
        raise ValueError(f'Expected a readable 1600x256 image: {path}')
    return value

def postprocess(probabilities):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (4, 256, 1600) or not np.isfinite(probabilities).all():
        raise ValueError('Expected finite (4,256,1600) probabilities')
    raw = probabilities >= np.asarray(THRESHOLDS)[:, None, None]
    final = np.zeros_like(raw, dtype=np.uint8)
    for region in extract_regions(raw, MIN_AREAS):
        left, top, right, bottom = region.box
        final[region.class_id - 1, top:bottom, left:right] |= region.mask.astype(np.uint8)
    return final

@torch.inference_mode()
def predict(model, bgr, device='cpu'):
    if bgr.shape != (256,1600,3):
        raise ValueError('Expected native image shape (256,1600,3)')
    # One image, one forward, no autocast, no augmentation or resizing.
    logits = model(original_tensor(bgr).unsqueeze(0).to(device=device, dtype=torch.float32))
    if logits.dtype != torch.float32 or logits.shape != (1,4,256,1600):
        raise ValueError('Unexpected model output shape or precision')
    probability = logits.sigmoid()[0].cpu().numpy()
    return probability, postprocess(probability)

def packed_masks(path):
    packed = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    if packed is None or packed.shape != (256,1600):
        raise ValueError(f'Expected native packed mask: {path}')
    return np.stack([((packed >> c)&1) for c in range(4)]).astype(np.uint8)

