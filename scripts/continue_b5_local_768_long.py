"""Independent epoch72-to-122 FP32 continuation; metric regressions are observations."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE_SHA = 'd061e49271210ede89233139f00a85d7588f5c7be52bee510f88157a9ca64979'
_engine_path = ROOT / 'scripts/continue_b5_local_768.py'
if hashlib.sha256(_engine_path.read_bytes()).hexdigest() != ENGINE_SHA:
    raise ValueError('Frozen 768 computation engine changed')
_spec = importlib.util.spec_from_file_location('long50_frozen_engine', _engine_path)
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)

FIXED = dict(schema='local_b5_768_long50_v1', base_config='configs/fpn_v9_kaggle_original.json',
    start_epoch=72, target_epoch=122, new_epochs=50, train_encoder=True, train_decoder=True,
    crop_width=768, crop_height=256, effective_batch_size=18, micro_batch_size=1,
    precision='fp32', validation_precision='fp32_direct',
    batchnorm='frozen_parent_running_buffers_trainable_affine',
    loss='0.6_BCE_0.4_global_Dice_two_pass_rng_replay', dice_epsilon=1e-7,
    foreground_crop_probability=.5, workers=0, train_images=8797, validation_images=1885,
    validate_every=1, updates_per_epoch=489, tail_group_size=13,
    optimizer='preserve_parent_RAdam', scheduler='preserve_parent_logistic_LambdaLR',
    parent_optimizer_step=35208, parent_optimizer_parameters=532,
    quality_stop=False, quality_policy='record_original_regression_thresholds_only',
    checkpoint_policy='last_best_and_every_completed_epoch', minimum_free_gib=5)


def write_status(output, **value):
    """Progress is telemetry: a Windows reader lock must not abort an update.

    Checkpoint/integrity writes retain their original strict error behavior.
    Only sharing/access errors on the replaceable progress JSON are retried and
    reported without raising; actual training faults are never swallowed here.
    """
    record = dict(at=engine.datetime.now().astimezone().isoformat(),pid=os.getpid(),**value)
    for attempt in range(10):
        try:
            engine.save_json(Path(output)/'status.json',record)
            return True
        except PermissionError as error:
            if attempt < 9:
                time.sleep(.1)
            else:
                engine.emit(phase='status_write_warning',at=record['at'],
                    current_epoch=value.get('current_epoch'),step=value.get('step'),
                    error=repr(error),training_aborted=False)
    return False


def validate_policy(policy):
    if policy != FIXED or any(type(policy.get(k)) is not type(v) for k, v in FIXED.items()):
        raise ValueError('Unsupported long continuation execution policy')
    return policy


def load_policy(path):
    return validate_policy(json.loads(Path(path).read_text(encoding='utf-8')))


def read_long_baseline(path):
    metrics = engine.read_baseline_gate(path)
    gate = json.loads(Path(path).read_text(encoding='utf-8'))
    # The immutable report stores precision at the gate level. Copy its verified
    # metadata into the epoch row without changing any measured metric or count.
    return {**metrics, 'canonical_precision': gate['canonical_precision'],
        'probability_dtype': 'float32', 'derived_protocol': True,
        'threshold_source': 'frozen reference; precision override FP32',
        'inference': dict(window=1600, stride=1600, height=256, amp=False, batch_size=1),
        'test_accessed': gate['test_accessed']}


def quality_observation(metrics, baseline):
    keys = ('region_f2', 'recall', 'false_alarm_rate')
    for item in (metrics, baseline):
        if any(not isinstance(item.get(k), Real) or isinstance(item[k], bool) or not math.isfinite(item[k]) for k in keys):
            raise ValueError('Finite quality metrics required')
    return dict(regressed=bool(engine.regression(metrics, baseline, engine.FIXED)), stop=False,
        policy='observation_only', reference_epoch=72,
        thresholds={k: engine.FIXED[k] for k in
            ('max_f2_drop', 'max_recall_drop', 'max_false_alarm_increase')})


def _production(identity):
    return identity.get('policy') == FIXED


def _metrics(metrics, complete=False):
    required = ('region_f2', 'recall', 'false_alarm_rate', 'classification_accuracy',
                'kaggle_style_mean_dice_at_05')
    if (not isinstance(metrics, dict) or metrics.get('images') != 1885 or
        metrics.get('n_gt') != 3029 or metrics.get('canonical_precision') != 'fp32_direct' or
        metrics.get('reference_protocol_sha256') != engine.PROTOCOL_SHA or
        not engine.finite_state(metrics) or any(not isinstance(metrics.get(k), Real) or
            isinstance(metrics[k], bool) or not math.isfinite(metrics[k]) for k in required)):
        raise ValueError('Complete finite canonical validation required')
    if complete and (len(metrics.get('by_class', [])) != 4 or
                     [m.get('class_id') for m in metrics['by_class']] != [1, 2, 3, 4]):
        raise ValueError('Four-class validation details required')


def normalize_validation_metrics(metrics):
    """Preserve numeric values while normalizing real validator NumPy scalars."""
    _metrics(metrics)
    result = metrics.copy()
    for key in ('region_f2', 'recall', 'false_alarm_rate', 'classification_accuracy',
                'kaggle_style_mean_dice_at_05'):
        result[key] = float(result[key])
    return result


def _check_payload(p, identity):
    required = {'model', 'optimizer', 'scheduler', 'scaler', 'rng', 'epoch', 'identity',
                'batch_size', 'best', 'config', 'last_epoch_record', 'pending_validation'}
    if (identity.get('schema') != FIXED['schema'] or not required.issubset(p) or
        p['identity'] != identity or p.get('diagnostic_only')):
        raise ValueError('Complete exact long-run identity required')
    if 'policy' in identity:
        validate_policy(identity['policy'])
    digest = hashlib.sha256(json.dumps(p['config'], sort_keys=True).encode()).hexdigest()
    if digest != identity['base_experiment']['config']:
        raise ValueError('Saved base config differs')
    epoch = p['epoch']
    if (type(epoch) is not int or not 72 <= epoch <= 122 or p['batch_size'] != 18 or
        p['scheduler'].get('last_epoch') != epoch or
        (p.get('last_epoch_record') or {}).get('epoch') != epoch or
        type(p['pending_validation']) is not bool):
        raise ValueError('Saved epoch/scheduler/record phase differs')
    if not all(engine.finite_state(p[k]) for k in ('model', 'optimizer', 'scheduler', 'scaler', 'best')):
        raise ValueError('Nonfinite saved training state')
    row = p['last_epoch_record']
    if p['pending_validation'] and (epoch == 72 or row.get('validation') is not None):
        raise ValueError('Invalid pending validation phase')
    if not p['pending_validation'] and not isinstance(row.get('validation'), dict):
        raise ValueError('Completed state needs validation')
    if _production(identity):
        validate_policy(identity['policy'])
        groups = p['optimizer'].get('param_groups', [])
        ids = [pid for group in groups for pid in group['params']]
        states = p['optimizer'].get('state', {})
        if len(groups) != 2 or [len(g['params']) for g in groups] != [501, 31] or len(set(ids)) != 532 or set(states) != set(ids):
            raise ValueError('Optimizer parameter groups or complete state differs')
        expected = 35208 + (epoch-72)*489
        if any(float(state.get('step', -1)) != expected for state in states.values()):
            raise ValueError('Optimizer step progress differs from completed epochs')
        config = p['config']
        n = min(epoch-config['flat_epochs'], config['anneal_epochs']-1)
        factor = 1. if epoch < config['flat_epochs'] else 1/(1+math.exp((n-config['anneal_epochs']*.5)/(config['anneal_epochs']*.1)))
        expected_lr = [config['encoder_lr']*factor, config['decoder_lr']*factor]
        if not engine.states_equal(p['scheduler'].get('_last_lr'), [g['lr'] for g in groups]) or any(
            not math.isclose(g['lr'], lr, rel_tol=1e-12, abs_tol=1e-15) for g, lr in zip(groups, expected_lr)):
            raise ValueError('Saved learning rate or original schedule differs')
        if not p['pending_validation']:
            _metrics(row['validation'], complete=True)
        if epoch > 72 and (row.get('samples') != 8797 or row.get('optimizer_updates') != 489 or
                           row.get('tail_group_size') != 13):
            raise ValueError('Incomplete epoch sample/update accounting')
    return p


def load_long_state(path, model, opt, scheduler, scaler, identity, device):
    p = _check_payload(torch.load(path, map_location='cpu', weights_only=False), identity)
    # The original generic loader restores all five state components. Long-schema
    # bounds and phase checks above are explicit, rather than relying on v1 guards.
    loaded = engine.load_state(path, model, opt, scheduler, scaler, identity, device)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Encoder and decoder parameters must all remain trainable')
    optimizer_ids = [id(v) for group in opt.param_groups for v in group['params']]
    if len(set(optimizer_ids)) != len(optimizer_ids) or set(optimizer_ids) != {id(v) for v in model.parameters()}:
        raise ValueError('Optimizer must update every trainable model parameter exactly once')
    del p
    return loaded


def verify_history(output, saved_epoch):
    path = Path(output)/'epochs.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []
    if [row.get('epoch') for row in rows] != list(range(72, saved_epoch+1)):
        raise ValueError('History must contain exactly continuous epochs72 through saved epoch')
    for row in rows:
        _metrics(row.get('validation'))
    return rows


def _publish_completed(output, p, baseline):
    """Idempotently finish publication after any interruption between atomic writes."""
    output = Path(output)
    if p['pending_validation']:
        raise ValueError('Cannot publish a pending epoch as complete')
    history_path = output/'epochs.jsonl'
    existing = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines()
                if line.strip()] if history_path.exists() else []
    epochs = [r.get('epoch') for r in existing]
    if len(epochs) != len(set(epochs)) or any(type(e) is not int or not 72 <= e <= p['epoch'] for e in epochs):
        raise ValueError('Duplicate or future history cannot be reconciled')
    row = p['last_epoch_record']
    _metrics(row['validation'], complete=_production(p['identity']))
    observation = quality_observation(row['validation'], baseline)
    if row.get('quality_observation') != observation:
        p = {**p, 'last_epoch_record': {**row, 'quality_observation': observation}}
        engine.atomic_torch(output/'last.pt', p)
    # Snapshot precedes the row: a row is never the only record of an epoch.
    snapshots = output/'checkpoints'
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots/f"epoch{p['epoch']:03d}.pt"
    if snapshot.exists():
        prior = torch.load(snapshot, map_location='cpu', weights_only=False)
        if not engine.states_equal(prior, p):
            raise ValueError('Existing immutable epoch snapshot differs')
        del prior
    else:
        engine.copy_atomic(output/'last.pt', snapshot)
    engine.history_row(output, p['last_epoch_record'])
    # A crash can occur after the engine selected best but before annotation.
    best_path = output/'best_region_f2.pt'
    best = torch.load(best_path, map_location='cpu', weights_only=False)
    if best.get('identity') != p['identity'] or best.get('pending_validation'):
        raise ValueError('Best checkpoint identity/phase differs')
    if best['epoch'] == p['epoch']:
        engine.copy_atomic(output/'last.pt', best_path)
    return p


def finish_long_validation(output, p, model, opt, scheduler, scaler, validator, args, device, baseline):
    def checked_validator(*values):
        result = normalize_validation_metrics(validator(*values))
        _metrics(result, complete=_production(p['identity']))
        quality_observation(result, baseline)
        return result
    engine.finish_validation(Path(output), p, model, opt, scheduler, scaler, checked_validator, args, device)
    # Read the actual persisted tensors rather than p, whose tensors may predate
    # this epoch. This also reconciles an engine publication interrupted earlier.
    saved = torch.load(Path(output)/'last.pt', map_location='cpu', weights_only=False)
    return _publish_completed(output, saved, baseline)


def _verify_snapshots(output, rows, identity):
    folder = output/'checkpoints'
    expected = {f"epoch{row['epoch']:03d}.pt" for row in rows}
    if {p.name for p in folder.glob('epoch*.pt')} != expected:
        raise ValueError('Every completed epoch requires exactly one immutable snapshot')
    for row in rows:
        p = _check_payload(torch.load(folder/f"epoch{row['epoch']:03d}.pt", map_location='cpu', weights_only=False), identity)
        if p['pending_validation'] or p['last_epoch_record'] != row:
            raise ValueError('History and immutable epoch snapshot differ')
        del p


def identity_for(base, runtime, adapter_sha, config_sha, baseline_sha, policy):
    if base.get('source') != engine.PARENT_SOURCE:
        raise ValueError('Frozen source identity differs')
    return dict(schema=FIXED['schema'], base_experiment=base, policy=validate_policy(policy),
        engine_sha256=ENGINE_SHA, adapter_sha256=adapter_sha, config_sha256=config_sha,
        parent_checkpoint_sha256=engine.PARENT_SHA, helper_sha256=engine.HELPERS.copy(),
        reference_protocol_sha256=engine.PROTOCOL_SHA, baseline_gate_sha256=baseline_sha,
        runtime=runtime, start_epoch=72, target_epoch=122, test_accessed=False,
        cuda_rng_mapping='saved device0 to local device0; cloud device1 ignored',
        derived_validation_protocol=dict(canonical_precision='fp32_direct',
            reference_protocol_sha256=engine.PROTOCOL_SHA,
            threshold_source='frozen reference thresholds and min_areas',
            precision_override='FP32 sigmoid and direct threshold; no FP16 round trip',
            connectivity=8, gt_min_area=1, region_iou_threshold=.5, exclusive=False))


def initialize_long_run(output, original, identity, baseline, resume=False):
    """Publish initial72 state, recovering only an exact interrupted migration."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'run_identity.json'
    if manifest.exists():
        if not resume or json.loads(manifest.read_text(encoding='utf-8')) != identity:
            raise ValueError('Explicit resume and identical complete run identity required')
        return
    artifacts = {'last.pt', 'best_region_f2.pt', 'baseline_validation.json', 'effective_policy.json'}
    incidental = {'writer.lock', 'latest_launch.json'}
    if resume:
        incidental.add('status.json')
        incidental.update({'last.pt.tmp', 'best_region_f2.pt.copying',
            'baseline_validation.json.tmp', 'effective_policy.json.tmp', 'run_identity.json.tmp'})
    allowed = artifacts | incidental if resume else incidental
    if any(path.name not in allowed and path.suffix != '.log' for path in output.iterdir()):
        raise ValueError('Initialization requires fresh output or an exact interrupted initial publication')
    last = output/'last.pt'
    if resume and not last.is_file():
        raise ValueError('Initial resume requires a complete saved epoch72 migration')
    expected = engine.migrate_parent(original, identity, baseline)
    _check_payload(expected, identity)
    # Validate every existing artifact before writing anything. A resume cannot
    # relabel an advanced, pending, altered, or unrelated checkpoint as epoch72.
    for name in ('last.pt', 'best_region_f2.pt'):
        path = output/name
        if path.exists():
            saved = _check_payload(torch.load(path, map_location='cpu', weights_only=False), identity)
            if not engine.states_equal(saved, expected):
                raise ValueError('Interrupted initial checkpoint differs from the exact parent72 migration')
            del saved
    metadata = {'baseline_validation.json': baseline,
                'effective_policy.json': identity.get('policy', FIXED)}
    for name, value in metadata.items():
        path = output/name
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Interrupted initial metadata differs from the bound experiment')
    if not last.exists():
        engine.atomic_torch(last, expected)
    if not (output/'best_region_f2.pt').exists():
        engine.copy_atomic(last, output/'best_region_f2.pt')
    for name, value in metadata.items():
        if not (output/name).exists():
            engine.save_json(output/name, value)
    # This is the final publication step. Earlier interruptions remain resumable
    # only while their complete last state is exactly the untouched migration.
    engine.save_json(manifest, identity)


def run(args):
    from vdcp.data import read_records
    from vdcp.kaggle_b5 import experiment_identity, OriginalCrops
    from torch.utils.data import DataLoader
    root = args.root.resolve()
    output = (args.probe_output if args.probe_only else args.output).resolve()
    policy = load_policy(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError('Local CUDA device0 required')
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    engine.cv2.setNumThreads(1)
    trainer, study = engine.frozen_helpers(root)
    base_config = json.loads((root/policy['base_config']).read_text(encoding='utf-8'))
    if engine.sha(args.parent) != engine.PARENT_SHA or engine.sha(root/base_config['validation_reference_protocol']) != engine.PROTOCOL_SHA:
        raise ValueError('Frozen parent/reference digest differs')
    engine.emit(phase='identity_hashing', parent=str(args.parent), output=str(output))
    baseline = read_long_baseline(args.baseline_gate)
    baseline_sha = engine.sha(args.baseline_gate)
    base = experiment_identity(root, base_config)
    original = torch.load(args.parent, map_location='cpu', weights_only=False)
    engine.verify_parent(original, base, base_config)
    runtime = engine.runtime_identity()
    engine.verify_baseline_binding(json.loads(args.baseline_gate.read_text(encoding='utf-8')), base, runtime)
    identity = identity_for(base, runtime, engine.sha(Path(__file__)), engine.sha(args.config), baseline_sha, policy)
    parent_bn = {name: value.clone() for name, value in original['model'].items()
                 if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    train, val = read_records(root, 'train'), read_records(root, 'val')
    if (len(train), len(val)) != (8797, 1885):
        raise ValueError('Immutable full development splits required')
    if args.probe_only:
        del original
        gc.collect()
        engine.scratch_probe(args, root, train, args.parent, base_config, identity, trainer, study, output)
        return
    manifest = output/'run_identity.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text(encoding='utf-8')) != identity:
            raise ValueError('Explicit resume and identical complete run identity required')
    else:
        probe = json.loads((args.probe_output/'preflight.json').read_text(encoding='utf-8'))
        if (probe.get('status') != 'passed' or probe.get('identity') != identity or
            set(probe.get('exact_save_load', {})) != {'model', 'optimizer', 'scheduler', 'scaler', 'rng'} or
            not all(probe['exact_save_load'].values()) or probe.get('groups', 0) < 3 or
            not probe.get('bn_unchanged') or not probe.get('finite_gradients')):
            raise ValueError('Matching successful real-model three-group probe required')
        initialize_long_run(output, original, identity, baseline, resume=args.resume)
    del original
    gc.collect()
    model, opt, scheduler, scaler = engine.make_components(trainer, study, base_config, 'cuda:0')
    p = load_long_state(output/'last.pt', model, opt, scheduler, scaler, identity, 'cuda:0')
    engine.assert_bn_unchanged(model, parent_bn)
    engine.save_json(output/'restoration_evidence.json', dict(epoch=p['epoch'], scheduler_last_epoch=scheduler.last_epoch,
        learning_rates=[g['lr'] for g in opt.param_groups], all_parameters_trainable=True,
        trainable_parameters=sum(1 for _ in model.parameters()), bn_unchanged=True,
        optimizer_steps=sorted({float(s['step']) for s in opt.state.values()}), identity=identity))
    engine.reconcile_best(output, identity)
    validator_args = (model, root, val, base_config, 'cuda:0')
    # Reconcile the last completed publication before requiring continuous history.
    # Pending epochs are validated first and never trained again.
    if p['pending_validation']:
        prior_rows = verify_history(output, p['epoch']-1)
        _verify_snapshots(output, prior_rows, identity)
        write_status(output, status='validation_recovery', current_epoch=p['epoch'], saved_epoch=p['epoch'], target_epoch=122)
        p = finish_long_validation(output, p, model, opt, scheduler, scaler, engine.validate, validator_args, 'cuda:0', baseline)
    else:
        p = _publish_completed(output, p, baseline)
    rows = verify_history(output, p['epoch'])
    _verify_snapshots(output, rows, identity)
    generator = torch.Generator()
    loader = DataLoader(OriginalCrops(root, train, 768), batch_size=18, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False, generator=generator)
    deadline = time.monotonic()+args.session_hours*3600
    for epoch in range(p['epoch']+1, 123):
        if (output/'STOP').exists() or time.monotonic() >= deadline:
            break
        if shutil.disk_usage(output).free < 5*1024**3:
            raise RuntimeError('Checkpoint reserve below 5GiB')
        engine.freeze_batchnorm(model)
        generator.manual_seed(base_config['seed']+epoch)
        total, samples, updates, tail = 0., 0, 0, 0
        started = time.monotonic()
        write_status(output, status='training', current_epoch=epoch, saved_epoch=p['epoch'], last_validated_epoch=p['epoch'], target_epoch=122)
        for step, (x, y) in enumerate(loader, 1):
            loss = engine.update_group(model, opt, scaler, x, y, 1, 'cuda:0')
            total += loss*len(x)
            samples += len(x)
            updates += 1
            tail = len(x)
            if step == 1 or step % 10 == 0:
                event = dict(phase='train', current_epoch=epoch, saved_epoch=p['epoch'], step=step,
                    total_steps=len(loader), loss=total/samples, seconds=time.monotonic()-started)
                write_status(output, status='training', target_epoch=122, **event)
                engine.emit(**event)
        if (samples, updates, tail) != (8797, 489, 13):
            raise ValueError('Full epoch including tail group required')
        engine.assert_bn_unchanged(model, parent_bn)
        scheduler.step()
        row = dict(epoch=epoch, loss=total/samples, train_seconds=time.monotonic()-started,
            validation=None, lr=[g['lr'] for g in opt.param_groups], samples=samples,
            optimizer_updates=updates, tail_group_size=tail, crop_width=768, micro_batch_size=1,
            effective_batch_size=18, train_encoder=True, train_decoder=True)
        engine.save_state(output/'last.pt', model, opt, scheduler, scaler, epoch, identity,
            p['best'], base_config, row, pending=True, device='cuda:0')
        # Keep only the current persisted state; do not retain stale CPU model copies.
        del p
        p = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        _check_payload(p, identity)
        write_status(output, status='validation', current_epoch=epoch, saved_epoch=epoch, last_validated_epoch=epoch-1, target_epoch=122)
        p = finish_long_validation(output, p, model, opt, scheduler, scaler, engine.validate, validator_args, 'cuda:0', baseline)
        engine.assert_bn_unchanged(model, parent_bn)
        rows = verify_history(output, epoch)
        window = rows[-5:]
        moving = dict(kind='auxiliary_five_epoch_mean', epochs=[r['epoch'] for r in window],
            **{key: sum(r['validation'][key] for r in window)/len(window)
               for key in ('region_f2', 'recall', 'false_alarm_rate', 'classification_accuracy')})
        engine.save_json(output/'latest_trend.json', moving)
        write_status(output, status='epoch_complete', current_epoch=epoch, saved_epoch=epoch,
            last_validated_epoch=epoch, target_epoch=122, best=p['best'],
            quality_observation=p['last_epoch_record']['quality_observation'], auxiliary_trend=moving)
        engine.emit(phase='epoch_complete', epoch=epoch, best=p['best'],
            quality_observation=p['last_epoch_record']['quality_observation'], validation=p['last_epoch_record']['validation'])
    if (experiment_identity(root, base_config) != base or engine.sha(args.parent) != engine.PARENT_SHA or
        engine.sha(Path(__file__)) != identity['adapter_sha256'] or engine.sha(_engine_path) != ENGINE_SHA or
        engine.sha(args.config) != identity['config_sha256'] or engine.sha(args.baseline_gate) != baseline_sha):
        raise ValueError('Bound inputs changed during continuation')
    engine.frozen_helpers(root)
    write_status(output, status='milestone_complete' if p['epoch'] == 122 else
        'stopped_at_epoch_boundary' if (output/'STOP').exists() else 'session_budget_complete',
        current_epoch=p['epoch'], saved_epoch=p['epoch'], last_validated_epoch=p['epoch'],
        target_epoch=122, best=p['best'], test_accessed=False)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/b5_local_768_long50_v1')
    parser.add_argument('--probe-output', type=Path, default=ROOT/'work/local_b5_768_long50_probe')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/local_b5_768_long50.json')
    parser.add_argument('--baseline-gate', type=Path, default=ROOT/'reports/v9/local_768_bnfix/baseline_precision.json')
    parser.add_argument('--parent', type=Path, default=ROOT/'artifacts/cloud_recovery/paired_epoch72/b5_paired_long/control/best_region_f2.pt')
    parser.add_argument('--target-epoch', type=int, choices=(122,), default=122)
    parser.add_argument('--session-hours', type=float, default=40.)
    parser.add_argument('--probe-groups', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args(argv)
    if args.resume and args.probe_only:
        parser.error('Resume and probe are mutually exclusive')
    if not math.isfinite(args.session_hours) or not 0 < args.session_hours <= 48 or not 3 <= args.probe_groups <= 10:
        parser.error('Invalid session or probe budget')
    output, scratch = args.output.resolve(), args.probe_output.resolve()
    if output == scratch or output in scratch.parents or scratch in output.parents:
        parser.error('Probe and production outputs must be separate directories')
    if (args.root/'work').resolve() not in scratch.parents:
        parser.error('Probe must use a dedicated directory beneath work/')
    for folder in ('outputs/b5_local_crop', 'outputs/b5_local_768_bnfix_v1', 'work/local_b5_768_bnfix_probe'):
        old = (args.root/folder).resolve()
        if any(path == old or old in path.parents for path in (output, scratch)):
            parser.error('Legacy output/probe directories cannot be reused')
    if args.probe_only and any((scratch/name).exists() for name in ('run_identity.json', 'last.pt', 'best_region_f2.pt')):
        parser.error('Probe cannot overwrite production state')
    return args


def main():
    args = arguments()
    output = (args.probe_output if args.probe_only else args.output).resolve()
    for name, folder in dict(TMP='tmp', TEMP='tmp', TMPDIR='tmp', TORCH_HOME='cache/torch',
        HF_HOME='cache/hf', PIP_CACHE_DIR='cache/pip', UV_CACHE_DIR='cache/uv').items():
        directory = args.root/'work'/folder
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    with engine.exclusive_writer(args.root/'work/local_b5_768_process_mutex'), engine.exclusive_writer(output), engine.keep_system_awake():
        try:
            run(args)
        except Exception as error:
            prior = json.loads((output/'status.json').read_text(encoding='utf-8')) if (output/'status.json').exists() else {}
            for key in ('status', 'at', 'pid'):
                prior.pop(key, None)
            write_status(output, **{**prior, 'status': 'failed', 'error': repr(error)})
            raise


if __name__ == '__main__':
    main()
