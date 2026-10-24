"""FP32 whole-group BCE/Dice updates for the separately identified cloud run.

Physical groups that fit use a single forward. Smaller physical batches replay
their RNG once to obtain the exact gradient of the whole-group Dice quotient.
The frozen engine and historical model/source remain byte-for-byte unchanged.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
ENGINE_SHA = 'd061e49271210ede89233139f00a85d7588f5c7be52bee510f88157a9ca64979'
_engine_path = ROOT / 'scripts/continue_b5_local_768.py'
if hashlib.sha256(_engine_path.read_bytes()).hexdigest() != ENGINE_SHA:
    raise ValueError('Frozen 768 computation engine changed')
_spec = importlib.util.spec_from_file_location('cloud5090_frozen_engine', _engine_path)
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)

SUPPORTED_MICRO_BATCHES = (18, 9, 6, 3)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensors(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError('Nonfinite model or optimizer state after update')


def _all_finite(tensors, device):
    """Return a device scalar; never convert each parameter's flag to Python."""
    buckets = {}
    for tensor in tensors:
        if tensor.is_floating_point() or tensor.is_complex():
            buckets.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    flags = [torch.stack(values).all().to(device=device, non_blocking=True)
             for values in buckets.values()]
    return torch.stack(flags).all() if flags else torch.ones((), device=device, dtype=torch.bool)


def _bn_buffers(model):
    return {name: value for name, value in model.named_buffers()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def _bn_equal(model, before, device):
    actual = _bn_buffers(model)
    if actual.keys() != before.keys():
        return torch.zeros((), device=device, dtype=torch.bool)
    flags = [(actual[name] == value).all().to(device=device, non_blocking=True)
             for name, value in before.items()]
    return torch.stack(flags).all() if flags else torch.ones((), device=device, dtype=torch.bool)


def _validate(model, opt, x, y, micro, device):
    if (not 0 < len(x) <= 18 or len(x) != len(y) or
            type(micro) is not int or micro not in SUPPORTED_MICRO_BATCHES):
        raise ValueError('Invalid accumulation group or physical micro batch')
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise ValueError('FP32 data required')
    parameters = list(model.parameters())
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in parameters):
        raise ValueError('FP32 model parameters required')
    if any(p.device != device for p in parameters):
        raise ValueError('Model parameters must be on the requested device')
    optimized = [p for group in opt.param_groups for p in group['params']]
    if len(optimized) != len(parameters) or {id(p) for p in optimized} != {id(p) for p in parameters}:
        raise ValueError('Optimizer coverage must contain every model parameter exactly once')


def _statistics(logits, target):
    if logits.shape != target.shape or logits.dtype != torch.float32:
        raise ValueError('FP32 logits must match target shape')
    probability = logits.sigmoid()
    return probability, torch.stack(((probability * target).sum(), probability.sum(),
        target.sum(), F.binary_cross_entropy_with_logits(logits, target, reduction='sum')))


def update_group(model, opt, scaler, x, y, micro, device):
    """Return whole effective-group ``.6 BCE + .4 Dice`` after one update.

    ``x``/``y`` are already augmented FP32 tensors for 1..18 images. Supported
    physical batches are 18, 9, 6 and 3. AMP is always disabled, while the saved
    GradScaler continues to advance normally. Frozen BN statistics and affine
    trainability are preserved. Gradients remain available after the update.

    Stochastic equivalence is to concatenated forwards at the chosen physical
    batch size: changing physical packing may change dropout masks. All finite,
    replay and BN checks reduce on-device before a bounded scalar transfer.
    """
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    _validate(model, opt, x, y, micro, device)
    engine.freeze_batchnorm(model)
    before_bn = {name: value.detach().clone() for name, value in _bn_buffers(model).items()}
    opt.zero_grad(set_to_none=True)
    replay_ok = torch.ones((), device=device, dtype=torch.bool)
    forward_ok = replay_ok.clone()
    if len(x) <= micro:
        with torch.autocast(device.type, enabled=False):
            xx = x.to(device, non_blocking=True); yy = y.to(device, non_blocking=True)
            logits = model(xx)
            if logits.shape != yy.shape or logits.dtype != torch.float32:
                raise ValueError('FP32 logits must match target shape')
            probability = logits.sigmoid()
            loss = .6 * F.binary_cross_entropy_with_logits(logits, yy)
            loss = loss + .4 * (1 - 2 * (probability * yy).sum() /
                (probability.sum() + yy.sum() + 1e-7))
            forward_ok = torch.isfinite(logits).all() & torch.isfinite(loss)
        loss_value = loss.detach()
        scaler.scale(loss).backward()
        del xx, yy, logits, probability, loss
    else:
        batches = list(zip(x.split(micro), y.split(micro)))
        rng = []; statistics = []
        total = torch.zeros(4, dtype=torch.float32, device=device)
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            for xx, yy in batches:
                rng.append(engine.capture_rng(device))
                xx = xx.to(device, non_blocking=True); yy = yy.to(device, non_blocking=True)
                logits = model(xx)
                probability, values = _statistics(logits, yy)
                forward_ok = forward_ok & torch.isfinite(logits).all() & torch.isfinite(values).all()
                statistics.append(values)
                total += values
                del xx, yy, logits, probability, values
        after = engine.capture_rng(device)
        intersection, probability_sum, target_sum, bce = total.unbind()
        denominator = probability_sum + target_sum + 1e-7
        loss_value = .6 * bce / y.numel() + .4 * (1 - 2 * intersection / denominator)
        if not bool(forward_ok & torch.isfinite(loss_value)):
            raise FloatingPointError('Nonfinite FP32 logits or whole-group loss')
        try:
            for (xx, yy), state, expected in zip(batches, rng, statistics):
                engine.restore_rng(state, device)
                with torch.autocast(device.type, enabled=False):
                    xx = xx.to(device, non_blocking=True); yy = yy.to(device, non_blocking=True)
                    logits = model(xx)
                    probability, actual = _statistics(logits, yy)
                    replay_ok = replay_ok & (actual.detach() == expected).all()
                    proxy = .6 * actual[3] / y.numel()
                    proxy = proxy + .4 * (-2 * actual[0] / denominator +
                        2 * intersection * actual[1] / denominator.square())
                scaler.scale(proxy).backward()
                del xx, yy, logits, probability, actual, proxy
        finally:
            engine.restore_rng(after, device)
    scaler.unscale_(opt)
    checks = torch.stack((forward_ok, replay_ok, _bn_equal(model, before_bn, device),
        _all_finite((p.grad for p in model.parameters() if p.grad is not None), device)))
    # The four diagnostic flags and reported loss cross the device boundary
    # together, rather than synchronizing each layer/parameter independently.
    forward_valid, replay_valid, bn_valid, gradients_valid, reported_loss = torch.cat(
        (checks.to(dtype=torch.float32), loss_value.reshape(1))).tolist()
    if not forward_valid:
        raise FloatingPointError('Nonfinite FP32 logits or whole-group loss')
    if not replay_valid:
        raise RuntimeError('RNG replay changed first/second pass predictions')
    if not bn_valid:
        raise RuntimeError('Frozen BN buffers changed')
    if not gradients_valid:
        raise FloatingPointError('Nonfinite FP32 gradients')
    scaler.step(opt); scaler.update()
    if not bool(_all_finite(_tensors((model.state_dict(), opt.state_dict())), device)):
        raise FloatingPointError('Nonfinite model or optimizer state after update')
    return reported_loss
