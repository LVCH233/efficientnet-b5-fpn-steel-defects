"""Single-T4 historical B5-FPN sessions; no local long training by default."""
import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from vdcp.data import read_records, read_masks
from vdcp.evaluate import aggregate_counts, postprocess_masks, save_probabilities
from vdcp.kaggle_b5 import (HistoricalFPN, OriginalCrops, original_loss,
    original_probability, experiment_identity, verify_resume_identity,
    can_start_epoch, check_records, file_sha)
from vdcp.regions import region_counts
from vdcp.train import seed_everything, worker_init, rng_state


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def optimizer_for(model, config):
    return torch.optim.RAdam([
        {'params': model.encoder.parameters(), 'lr': config['encoder_lr'],
         'weight_decay': config['encoder_weight_decay']},
        {'params': model.decoder.parameters(), 'lr': config['decoder_lr'],
         'weight_decay': config['decoder_weight_decay']}], decoupled_weight_decay=True)


def lr_factor(epoch, config):
    if epoch < config['flat_epochs']:
        return 1.
    # Upstream names this CosineAnnealingScheduler but implements logistic.sf.
    length = config['anneal_epochs']
    index = min(epoch-config['flat_epochs'], length-1)
    return 1 / (1 + math.exp((index-length*.5)/(length*.1)))


def save_full(path, model, optimizer, scheduler, scaler, epoch, identity, batch_size, best, config=None, row=None):
    temporary = Path(path).with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'rng': rng_state(), 'epoch': epoch, 'identity': identity,
                'batch_size': batch_size, 'best': best, 'config': config,
                'last_epoch_record': row}, temporary)
    temporary.replace(path)


def restore(path, model, optimizer, scheduler, scaler, identity):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    verify_resume_identity(payload, identity)
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['scaler'])
    import random
    random.setstate(payload['rng']['python'])
    np.random.set_state(payload['rng']['numpy'])
    torch.set_rng_state(payload['rng']['torch'])
    if payload['rng']['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload['rng']['cuda'])
    return payload


def validate(model, root, records, config, device, cache=None, checkpoint=None):
    reference = json.loads((root/config['validation_reference_protocol']).read_text(encoding='utf-8'))
    counts, dice_sum, precision_retries = [], 0., []
    started = time.monotonic()
    model.eval()
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        write_json(cache/'metadata.json', dict(checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=file_sha(checkpoint), split='val',
            split_csv_sha256=file_sha(root/'data/splits/val.csv'), probability_dtype='float16',
            inference=dict(window=1600, stride=1600, height_window=256, height_stride=192,
                           amp=True, nonfinite_policy='retry_fp32_then_raise', batch_size=1),
            images=len(records), input_convention=config['input_convention']))
    for index, row in enumerate(records, 1):
        image = cv2.imread(str(root/'data/raw/train_images'/row['image_id']))
        if image is None:
            raise ValueError('Cannot read validation image')
        def record_retry():
            precision_retries.append(row['image_id'])
            print(json.dumps(dict(phase='validation_fp32_retry', image_id=row['image_id'])), flush=True)
        try:
            probability = original_probability(model, image, device, on_fp32_retry=record_retry)
        except FloatingPointError as error:
            raise FloatingPointError(f'{error}; validation image {row["image_id"]}') from error
        target = read_masks(root, row['image_id'])
        prediction = probability >= .5
        intersection = (prediction & target.astype(bool)).sum(axis=(1,2))
        denominator = prediction.sum(axis=(1,2)) + target.sum(axis=(1,2))
        dice_sum += np.divide(2*intersection, denominator, out=np.ones(4), where=denominator>0).sum()
        filtered = postprocess_masks(probability, reference['thresholds'], reference['min_areas'])
        counts.append(region_counts(target, filtered))
        if cache:
            save_probabilities(cache/(Path(row['image_id']).stem+'.npz'), probability, target.shape)
        if index%200 == 0 or index == len(records):
            print(json.dumps(dict(phase='validation', images=index, total_images=len(records),
                                  seconds=time.monotonic()-started)), flush=True)
    result = aggregate_counts(counts)
    result['region_f2'] = 5*result['tp'] / max(5*result['tp']+4*result['fn']+result['fp'], 1)
    result.update(images=len(records), kaggle_style_mean_dice_at_05=dice_sum/(4*len(records)),
                  seconds=time.monotonic()-started, reference_protocol_sha256=file_sha(root/config['validation_reference_protocol']),
                  fp32_retry_count=len(precision_retries), fp32_retry_images=precision_retries,
                  inference_precision_policy='amp_with_fp32_retry_on_nonfinite_logits')
    if cache:
        metadata = json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
        metadata.update(fp32_retry_count=len(precision_retries), fp32_retry_images=precision_retries)
        write_json(cache/'metadata.json', metadata)
    return result


def preflight(root, train, val, config, device, output):
    """Probe fresh scratch models, discard them; choose actual batch without accumulation."""
    dataset = OriginalCrops(root, train, config['crop_width'])
    if device == 'cpu':
        raise RuntimeError('Production preflight requires a CUDA GPU')
    errors = []
    for batch_size in (config['batch_size'], config['fallback_batch_size']):
        model = optimizer = scaler = loader = loss = images = masks = None
        timings = []
        try:
            model = HistoricalFPN(root/config['pretrained_file']).to(device)
            optimizer = optimizer_for(model, config)
            scaler = torch.amp.GradScaler('cuda')
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
            torch.cuda.reset_peak_memory_stats()
            for index, (images, masks) in enumerate(loader):
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda'):
                    loss = original_loss(model(images.to(device)), masks.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite preflight loss')
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                torch.cuda.synchronize()
                timings.append(time.monotonic()-started)
                if index == 2:
                    break
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite parameters after preflight update')
            diagnostic_validation = validate(model, root, val[:20], config, device)
            report = dict(status='passed', diagnostic_only=True, gpu=torch.cuda.get_device_name(0),
                          actual_batch_size=batch_size, accumulation_steps=1,
                          mean_step_seconds=float(np.mean(timings[1:])),
                          estimated_epoch_seconds=float(np.mean(timings[1:]))*math.ceil(len(dataset)/batch_size),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          estimated_validation_seconds=diagnostic_validation['seconds'] * len(val)/min(len(val),20)*1.3,
                          failures=errors, utc=now())
            write_json(output/'preflight.json', report)
            print(json.dumps(report), flush=True)
            return report
        except torch.cuda.OutOfMemoryError as error:
            errors.append(dict(batch_size=batch_size, error=str(error)))
        finally:
            del loss, images, masks, model, optimizer, scaler, loader
            gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError('Actual batches 18 and 16 both failed; no small-batch fallback authorized')


def run(args):
    session_started = time.monotonic()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root/'configs/fpn_v9_kaggle_original.json').read_text(encoding='utf-8'))
    train, val = read_records(root, 'train'), read_records(root, 'val')
    check_records(train, val)
    if not args.smoke and (len(train), len(val)) != (8797, 1885):
        raise ValueError('Expected immutable full train/validation split')
    if args.smoke:
        train = train[:2]
        # Include a labelled defect in the full-image smoke evaluation.
        val = [next(row for row in val if any(int(row[f'c{i}']) for i in range(1,5)))]
    identity = experiment_identity(root, config)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Enable Kaggle T4 before training')
    if not args.smoke and not args.allow_local and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE') and not Path('/kaggle').exists():
        raise RuntimeError('Long training is limited to Kaggle; local requires explicit --allow-local')
    seed_everything(config['seed'])
    torch.set_num_threads(4)
    resumed = args.resume is not None
    if resumed:
        preview = torch.load(args.resume, map_location='cpu', weights_only=False)
        verify_resume_identity(preview, identity)
        batch_size = preview['batch_size']
        if batch_size not in ((1,) if args.smoke else (18,16)):
            raise ValueError('Unsupported checkpoint actual batch')
        del preview
        probe = json.loads((output/'preflight.json').read_text()) if (output/'preflight.json').exists() else None
    elif args.smoke:
        batch_size, probe = 1, None
    else:
        probe = preflight(root, train, val, config, args.device, output)
        batch_size = probe['actual_batch_size']
    seed_everything(config['seed'])
    model = HistoricalFPN(None if resumed else root/config['pretrained_file']).to(args.device)
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_factor(epoch, config))
    scaler = torch.amp.GradScaler('cuda', enabled=args.device=='cuda')
    best, completed = {'region_f2': -1., 'dice': -1.}, 0
    if resumed:
        payload = restore(args.resume, model, optimizer, scheduler, scaler, identity)
        completed, best = payload['epoch'], payload['best']
        del payload
    if args.evaluate_only:
        if not resumed:
            raise ValueError('--evaluate-only requires a verified --resume checkpoint')
        checkpoint = args.resume
        cache = output/'validation_cache'/file_sha(checkpoint)[:12]
        fixed = validate(model, root, val, config, args.device, cache, checkpoint)
        from vdcp.calibrate import calibrate
        from vdcp.evaluate import score_cache
        protocol = calibrate(root, cache, output/'validation')
        result = score_cache(root, cache, protocol, output/'validation')
        write_json(output/'final_evaluation.json', dict(fixed=fixed, calibrated=result))
        print(json.dumps(result), flush=True)
        return
    loader_generator = torch.Generator()
    loader = DataLoader(OriginalCrops(root, train, config['crop_width']), batch_size=batch_size,
                        shuffle=True, num_workers=0 if args.smoke else config['num_workers'],
                        worker_init_fn=worker_init, pin_memory=args.device=='cuda', drop_last=False,
                        generator=loader_generator)
    deadline = session_started + (args.session_hours or config['session_hours'])*3600
    estimate = probe['estimated_epoch_seconds'] if probe else 600.
    validation_estimate = probe.get('estimated_validation_seconds', 1800.) if probe else 1800.
    target = 1 if args.smoke else min(args.stop_at, config['epochs'])
    write_json(output/'run_identity.json', dict(identity=identity, config=config, batch_size=batch_size,
                                               source_adaptation=True, started_utc=now()))
    if not (output/'last.pt').exists():
        save_full(output/'last.pt', model, optimizer, scheduler, scaler, completed, identity, batch_size, best, config)
    for epoch in range(completed+1, target+1):
        due = epoch % config['validate_every'] == 0 or epoch == target
        if not args.smoke and not can_start_epoch(deadline-time.monotonic(), estimate,
                                                  validation_estimate if due else 0):
            break
        if shutil.disk_usage(output).free < 3*1024**3:
            raise RuntimeError('Less than 3GiB free disk space')
        write_json(output/'status.json', dict(status='training', epoch=epoch, completed_epochs=completed,
                                             target_epoch=target, actual_batch=batch_size, utc=now()))
        model.train()
        loader_generator.manual_seed(config['seed'] + epoch)
        epoch_start, total, samples = time.monotonic(), 0., 0
        for step, (images, masks) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device, enabled=args.device=='cuda'):
                loss = original_loss(model(images.to(args.device, non_blocking=True)), masks.to(args.device, non_blocking=True))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, step {step}')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total += loss.item()*len(images); samples += len(images)
            if step == 1 or step%50 == 0:
                print(json.dumps(dict(phase='train', epoch=epoch, step=step, total_steps=len(loader),
                                      loss=total/samples, elapsed_seconds=time.monotonic()-epoch_start)), flush=True)
        estimate = time.monotonic()-epoch_start
        metrics = validate(model, root, val, config, args.device) if due or args.smoke else None
        improved = []
        if metrics:
            validation_estimate = metrics['seconds']
            for key, value in [('region_f2', metrics['region_f2']), ('dice', metrics['kaggle_style_mean_dice_at_05'])]:
                if value > best[key]:
                    best[key] = value; improved.append(key)
        scheduler.step()
        row = dict(epoch=epoch, loss=total/samples, train_seconds=estimate, validation=metrics,
                   best=best.copy(), lr=[group['lr'] for group in optimizer.param_groups], utc=now())
        save_full(output/'last.pt', model, optimizer, scheduler, scaler, epoch, identity, batch_size, best, config, row)
        for key in improved:
            shutil.copyfile(output/'last.pt', output/f'best_{key}.pt')
        with (output/'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        completed = epoch
        print(json.dumps(dict(phase='epoch_complete', **row)), flush=True)
    state = dict(status='milestone_complete' if completed==target else 'session_budget_complete',
                 completed_epochs=completed, target_epoch=target, maximum_epochs=config['epochs'],
                 actual_batch=batch_size, best=best, utc=now(), resume_checkpoint=str(output/'last.pt'))
    write_json(output/'status.json', state)
    print(json.dumps(state), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/kaggle_b5')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-at', type=int, default=60)
    parser.add_argument('--session-hours', type=float)
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    parser.add_argument('--allow-local', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.stop_at <= 200 or (args.session_hours is not None and not 0 < args.session_hours <= 10.5):
        parser.error('stop-at must be 1..200 and session-hours within 0..10.5')
    run(args)


if __name__ == '__main__':
    main()
