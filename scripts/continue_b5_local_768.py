"""Single-device FP32 epoch72 continuation with frozen BN and exact group Dice.

The frozen original trainer/helpers are imported only after digest checks. This
adapter never imports the deprecated local continuation. Two forwards reuse the
same already-augmented CPU tensors and replay each microbatch's RNG state.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
PARENT_SOURCE = '73cad53ba189a4df70950f29920897b287c0762c8b5cb4d134adcddd8e2bee66'
PARENT_SHA = '5b9f07774bbf464d872e6edf15ed2a07d238bc2a43ef1e581dfd8a7eb00860cb'
PROTOCOL_SHA = 'ba265b3e5c2188ef5e70d2c60f0a89adb61bbae632de32583c8eb4bf90a8ed35'
HELPERS = {
    'train_kaggle_b5.py': '3309271c47c5a926163b1f4392e8f77465a4e9799ab3e1e1590c6c59ef8c48e2',
    'study_kaggle_b5.py': 'db196856a8872db3626a6564cb08eaec388470d93db4d8940361565cfcab53f4',
}
FIXED = dict(schema='local_b5_768_bnfix_v1', base_config='configs/fpn_v9_kaggle_original.json',
    crop_width=768, crop_height=256, effective_batch_size=18, micro_batch_size=1,
    precision='fp32', validation_precision='fp32_direct',
    batchnorm='frozen_parent_running_buffers_trainable_affine',
    loss='0.6_BCE_0.4_global_Dice_two_pass_rng_replay', dice_epsilon=1e-7,
    foreground_crop_probability=.5, workers=0, validation_images=1885, validate_every=1,
    default_target_epoch=73, allowed_target_epochs=[73,76,80],
    optimizer='preserve_parent_RAdam', scheduler='preserve_parent_logistic_LambdaLR',
    max_f2_drop=.01, max_recall_drop=.02, max_false_alarm_increase=.1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def emit(**value):
    print(json.dumps(value, allow_nan=False), flush=True)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary); temporary.replace(path)


def copy_atomic(source, target):
    target = Path(target); temporary = target.with_suffix(target.suffix+'.copying')
    shutil.copyfile(source, temporary); temporary.replace(target)


def validate_config(config):
    if config != FIXED or any(type(config.get(k)) is not type(v) for k,v in FIXED.items()):
        raise ValueError('Unsupported local config; every fixed policy field must match')
    return config


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding='utf-8')))


def identity_for(base, runtime, adapter_sha, config_sha, baseline_sha, config):
    validate_config(config)
    if base.get('source') != PARENT_SOURCE: raise ValueError('Frozen source identity differs')
    return dict(schema='local_b5_768_bnfix_v1', base_experiment=base,
        parent_checkpoint_sha256=PARENT_SHA, adapter_sha256=adapter_sha, config_sha256=config_sha,
        helper_sha256=HELPERS.copy(), baseline_gate_sha256=baseline_sha, runtime=runtime,
        crop_width=768, effective_batch_size=18, start_epoch=72, policy=config,
        derived_validation_protocol=dict(canonical_precision='fp32_direct',
            reference_protocol_sha256=PROTOCOL_SHA, threshold_source='frozen reference thresholds and min_areas',
            precision_override='FP32 sigmoid and direct threshold; no FP16 round trip',
            connectivity=8, gt_min_area=1, region_iou_threshold=.5, exclusive=False),
        cuda_rng_mapping='saved device0 to local device0; cloud device1 ignored', test_accessed=False)


def capture_rng(device):
    cuda = None
    if torch.device(device).type == 'cuda': cuda = [torch.cuda.get_rng_state(0).clone()]
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone(), cuda=cuda)


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if torch.device(device).type == 'cuda':
        if torch.device(device).index not in (None,0) or not state.get('cuda'):
            raise ValueError('Saved device0 CUDA RNG is required')
        torch.cuda.set_rng_state(state['cuda'][0], device=0)


def freeze_batchnorm(model):
    model.train()
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            if not layer.track_running_stats: raise ValueError('BN requires saved running statistics')
            layer.eval()
            # Affine parameters retain their saved requires_grad flags (normally True).
    return model


def bn_snapshot(model):
    return {name: value.detach().cpu().clone() for name,value in model.named_buffers()
        if name.endswith(('running_mean','running_var','num_batches_tracked'))}


def assert_bn_unchanged(model, expected):
    actual = bn_snapshot(model)
    if not states_equal(actual, expected): raise RuntimeError('Frozen BN buffers changed')


def states_equal(first, second):
    if isinstance(first, torch.Tensor): return isinstance(second,torch.Tensor) and torch.equal(first.cpu(),second.cpu())
    if isinstance(first, np.ndarray): return isinstance(second,np.ndarray) and np.array_equal(first,second)
    if isinstance(first, dict): return isinstance(second,dict) and first.keys()==second.keys() and all(states_equal(v,second[k]) for k,v in first.items())
    if isinstance(first,(list,tuple)): return type(first) is type(second) and len(first)==len(second) and all(states_equal(a,b) for a,b in zip(first,second))
    return first == second


def finite_state(state):
    if isinstance(state, torch.Tensor): return bool(torch.isfinite(state).all())
    if isinstance(state, dict): return all(finite_state(v) for v in state.values())
    if isinstance(state,(list,tuple)): return all(finite_state(v) for v in state)
    if isinstance(state,float): return math.isfinite(state)
    return True


def update_group(model, opt, scaler, x, y, micro, device):
    """Gradient of original_loss over the whole group, using bounded GPU memory.

    I and D are detached whole-group statistics. The second pass differentiates
    -2 I_micro/D + 2 I_global P_micro/D², the exact quotient-rule gradient.
    BCE is normalized by whole-group element count, including partial groups.
    Stochastic references use concatenated microbatch forwards, because dropout
    masks for one full physical batch need not equal smaller-kernel masks.
    """
    if not 0 < len(x) <= 18 or micro not in (1,2,3,6,9) or len(x)!=len(y):
        raise ValueError('Invalid accumulation group')
    if x.dtype != torch.float32 or y.dtype != torch.float32: raise ValueError('FP32 data required')
    freeze_batchnorm(model); before_bn=bn_snapshot(model)
    opt.zero_grad(set_to_none=True)
    batches=list(zip(x.split(micro), y.split(micro)))
    rng=[]; statistics=[]
    intersection=torch.zeros((),device=device); probability_sum=intersection.clone()
    target_sum=intersection.clone(); bce=intersection.clone()
    with torch.no_grad(), torch.autocast(torch.device(device).type,enabled=False):
        for xx,yy in batches:
            rng.append(capture_rng(device)); xx=xx.to(device); yy=yy.to(device)
            logits=model(xx).float(); p=logits.sigmoid()
            if logits.shape!=yy.shape or not torch.isfinite(logits).all(): raise FloatingPointError('Invalid first-pass logits')
            values=torch.stack(((p*yy).sum(),p.sum(),yy.sum(),F.binary_cross_entropy_with_logits(logits,yy,reduction='sum')))
            if not torch.isfinite(values).all(): raise FloatingPointError('Nonfinite group statistics')
            statistics.append(values.detach().clone())
            intersection+=values[0]; probability_sum+=values[1];target_sum+=values[2];bce+=values[3]
            del xx,yy,logits,p,values
    after=capture_rng(device)
    denominator=probability_sum+target_sum+1e-7
    loss_value=(.6*bce/y.numel()+.4*(1-2*intersection/denominator)).item()
    if not math.isfinite(loss_value): raise FloatingPointError('Nonfinite global loss')
    for (xx,yy),state,expected in zip(batches,rng,statistics):
        restore_rng(state,device)
        with torch.autocast(torch.device(device).type,enabled=False):
            xx=xx.to(device); yy=yy.to(device);logits=model(xx).float(); p=logits.sigmoid()
            local_i=(p*yy).sum(); local_p=p.sum()
            actual=torch.stack((local_i.detach(),local_p.detach(),yy.sum().detach(),F.binary_cross_entropy_with_logits(logits.detach(),yy,reduction='sum')))
            if not torch.equal(actual,expected): raise RuntimeError('RNG replay changed first/second pass predictions')
            proxy=.6*F.binary_cross_entropy_with_logits(logits,yy,reduction='sum')/y.numel()
            proxy=proxy+.4*(-2*local_i/denominator+2*intersection*local_p/denominator.square())
        scaler.scale(proxy).backward()
        del xx,yy,logits,p,proxy,actual,local_i,local_p
    restore_rng(after,device)
    scaler.unscale_(opt)
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError('Nonfinite local gradients')
    scale=scaler.get_scale();scaler.step(opt);scaler.update()
    if scaler.get_scale()<scale or not finite_state(model.state_dict()) or not finite_state(opt.state_dict()):
        raise FloatingPointError('Invalid optimizer update')
    assert_bn_unchanged(model,before_bn)
    return loss_value


def save_state(path,model,opt,scheduler,scaler,epoch,identity,best,config,row,pending=False,device='cuda:0'):
    atomic_torch(path,dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(),rng=capture_rng(device),epoch=epoch,identity=identity,batch_size=18,
        best=best.copy(),config=config,last_epoch_record=row,pending_validation=pending))


def load_state(path,model,opt,scheduler,scaler,identity,device):
    p=torch.load(path,map_location='cpu',weights_only=False)
    required={'model','optimizer','scheduler','scaler','rng','epoch','identity','batch_size','best','config','last_epoch_record','pending_validation'}
    if not required.issubset(p) or p['identity']!=identity or p.get('diagnostic_only') or p['batch_size']!=18 or not finite_state(p['model']) or not finite_state(p['optimizer']):
        raise ValueError('Checkpoint identity or complete finite state differs')
    if identity.get('schema')=='local_b5_768_bnfix_v1':
        config_digest=hashlib.sha256(json.dumps(p['config'],sort_keys=True).encode()).hexdigest()
        if config_digest!=identity['base_experiment']['config']:raise ValueError('Saved base config differs')
        if not 72<=p['epoch']<=80 or p['scheduler'].get('last_epoch')!=p['epoch'] or (p.get('last_epoch_record') or {}).get('epoch')!=p['epoch']:
            raise ValueError('Saved epoch/scheduler phase differs')
    model.load_state_dict(p['model'],strict=True); opt.load_state_dict(p['optimizer'])
    scheduler.load_state_dict(p['scheduler']);scaler.load_state_dict(p['scaler']);restore_rng(p['rng'],device)
    freeze_batchnorm(model)
    return p


def read_baseline_gate(path):
    gate=json.loads(Path(path).read_text(encoding='utf-8'))
    if gate.get('status')!='complete' or gate.get('parent_sha256')!=PARENT_SHA or gate.get('canonical_precision')!='fp32_direct' or gate.get('reference_protocol_sha256')!=PROTOCOL_SHA:
        raise ValueError('Complete bound canonical FP32 baseline required')
    m=gate.get('fp32_direct',{})
    required=('region_f2','recall','false_alarm_rate','kaggle_style_mean_dice_at_05')
    if m.get('images')!=1885 or m.get('n_gt')!=3029 or any(not isinstance(m.get(k),(int,float)) or not math.isfinite(m[k]) for k in required):
        raise ValueError('Incomplete baseline validation')
    return m


def verify_baseline_binding(gate,base,runtime):
    if gate.get('base_experiment')!=base or gate.get('test_accessed') is not False:
        raise ValueError('Baseline data/source identity differs')
    mapping={key:key for key in ('python','torch','cuda','cudnn','numpy','opencv','scipy','gpu','cudnn_benchmark','cudnn_deterministic')}
    mapping.update(tf32_matmul='matmul_allow_tf32',tf32_cudnn='cudnn_allow_tf32')
    environment=gate.get('environment',{})
    if any(key not in environment or environment[key]!=runtime.get(target) for key,target in mapping.items()):
        raise ValueError('Baseline precision/library/device environment differs')


def regression(metrics,baseline,config):
    return (metrics['region_f2'] < baseline['region_f2']-config['max_f2_drop'] or
        metrics['recall'] < baseline['recall']-config['max_recall_drop'] or
        metrics['false_alarm_rate'] > baseline['false_alarm_rate']+config['max_false_alarm_increase'])


def next_action(payload,target):
    if not 72<=payload['epoch']<=80: raise ValueError('Invalid continuation epoch')
    if payload.get('pending_validation'): return 'validate'
    return 'complete' if payload['epoch']>=target else 'train'


def fp32_postprocess(probabilities,thresholds,min_areas):
    from vdcp.evaluate import validate_postprocessing
    from vdcp.regions import extract_regions
    validate_postprocessing(thresholds,min_areas)
    p=np.asarray(probabilities)
    if p.dtype!=np.float32 or p.ndim!=3 or p.shape[0]!=4 or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise ValueError('Direct FP32 probabilities required')
    binary=(p>=np.asarray(thresholds)[:,None,None]).astype(np.uint8)
    filtered=np.zeros_like(binary)
    for region in extract_regions(binary,min_areas):
        left,top,right,bottom=region.box
        filtered[region.class_id-1,top:bottom,left:right]|=region.mask.astype(np.uint8)
    return filtered


@torch.inference_mode()
def validate(model,root,records,base_config,device):
    from vdcp.data import read_masks
    from vdcp.evaluate import aggregate_counts
    from vdcp.kaggle_b5 import original_tensor
    from vdcp.regions import region_counts
    if len(records)!=1885: raise ValueError('Every epoch requires complete 1885-image validation')
    protocol_path=root/base_config['validation_reference_protocol']
    if sha(protocol_path)!=PROTOCOL_SHA:raise ValueError('Frozen threshold reference changed')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    model.eval();counts=[];dice_sum=0.;started=time.monotonic()
    for index,row in enumerate(records,1):
        image=cv2.imread(str(root/'data/raw/train_images'/row['image_id']))
        if image is None or image.shape[:2]!=(256,1600):raise ValueError('Full native validation image required')
        tensor=original_tensor(image).unsqueeze(0).to(device)
        with torch.autocast(torch.device(device).type,enabled=False):logits=model(tensor).float()
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite FP32 validation logits')
        probability=logits[0].sigmoid().cpu().numpy()
        target=read_masks(root,row['image_id'])
        masks=fp32_postprocess(probability,protocol['thresholds'],protocol['min_areas'])
        counts.append(region_counts(target,masks))
        binary=probability>=.5
        intersection=(binary&target.astype(bool)).sum(axis=(1,2))
        denominator=binary.sum(axis=(1,2))+target.sum(axis=(1,2))
        dice_sum+=np.divide(2*intersection,denominator,out=np.ones(4),where=denominator>0).sum()
        if index%200==0 or index==len(records):emit(phase='validation',images=index,total_images=len(records),seconds=time.monotonic()-started)
    result=aggregate_counts(counts)
    result.update(region_f2=5*result['tp']/max(5*result['tp']+4*result['fn']+result['fp'],1),
        images=len(records),kaggle_style_mean_dice_at_05=dice_sum/(4*len(records)),seconds=time.monotonic()-started,
        reference_protocol_sha256=PROTOCOL_SHA,canonical_precision='fp32_direct',probability_dtype='float32',
        inference_precision_policy='FP32_forward_sigmoid_direct_threshold_no_FP16_round_trip',
        derived_protocol=True,threshold_source='frozen reference; precision override FP32',
        inference=dict(window=1600,stride=1600,height=256,amp=False,batch_size=1),test_accessed=False)
    return result


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value);return value


def frozen_helpers(root):
    for name,digest in HELPERS.items():
        if sha(root/'scripts'/name)!=digest:raise ValueError('Frozen helper changed')
    return module(root/'scripts/train_kaggle_b5.py','bnfix_trainer'),module(root/'scripts/study_kaggle_b5.py','bnfix_study')


def make_components(trainer,study,config,device):
    model=study.full_precision_class(trainer.HistoricalFPN)().to(device)
    opt=trainer.optimizer_for(model,config)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda epoch:trainer.lr_factor(epoch,config))
    scaler=torch.amp.GradScaler('cuda',enabled=torch.device(device).type=='cuda')
    return model,opt,scheduler,scaler


def verify_parent(payload,base,base_config):
    needed={'model','optimizer','scheduler','scaler','rng','epoch','batch_size','best','config','identity','last_epoch_record','pending_validation'}
    if not needed.issubset(payload) or payload.get('diagnostic_only') or payload['epoch']!=72 or payload['batch_size']!=18 or payload['pending_validation']:
        raise ValueError('Reliable complete epoch72 parent required')
    if payload['identity'].get('base_experiment')!=base or payload['identity'].get('arm')!='control' or payload['config']!=base_config:
        raise ValueError('Parent base/config/control identity differs')
    if not finite_state(payload['model']) or not finite_state(payload['optimizer']):raise ValueError('Nonfinite parent state')
    if payload['scheduler'].get('last_epoch')!=72:raise ValueError('Parent LR phase differs')


def migrate_parent(payload,identity,baseline):
    row={**payload['last_epoch_record'],'epoch':72,'validation':baseline,'validation_origin':'new_local_fp32_baseline_gate'}
    return {**payload,'identity':identity,'best':dict(region_f2=baseline['region_f2'],dice=baseline['kaggle_style_mean_dice_at_05']),
        'last_epoch_record':row,'pending_validation':False,
        'migration':dict(parent_checkpoint_sha256=PARENT_SHA,parent_identity=payload['identity'],preserved=['model','optimizer','scheduler','scaler','rng','epoch','config'])}


def history_row(output,row):
    path=output/'epochs.jsonl'
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []
    rows=[r for r in rows if r['epoch']!=row['epoch']]+[row]
    temporary=path.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in sorted(rows,key=lambda r:r['epoch'])),encoding='utf-8');temporary.replace(path)


def reconcile_history(output,p):
    if not p['pending_validation'] and (p.get('last_epoch_record') or {}).get('validation'):
        history_row(output,p['last_epoch_record'])


def finish_validation(output,p,model,opt,scheduler,scaler,validator,args,device):
    if not p['pending_validation']:return p
    # Validation must not change the saved next-epoch random stream.
    state=capture_rng(device)
    try:metrics=validator(*args)
    finally:restore_rng(state,device)
    if metrics.get('images')!=1885 or metrics.get('n_gt')!=3029 or metrics.get('canonical_precision')!='fp32_direct' or metrics.get('reference_protocol_sha256')!=PROTOCOL_SHA or not finite_state(metrics):
        raise ValueError('Incomplete canonical epoch validation')
    best=p['best'].copy();improved=metrics['region_f2']>best['region_f2']
    best['region_f2']=max(best['region_f2'],metrics['region_f2']);best['dice']=max(best['dice'],metrics['kaggle_style_mean_dice_at_05'])
    row={**p['last_epoch_record'],'validation':metrics,'best':best.copy()}
    candidate=output/'validated.pt'
    save_state(candidate,model,opt,scheduler,scaler,p['epoch'],p['identity'],best,p['config'],row,device=device)
    if improved:copy_atomic(candidate,output/'best_region_f2.pt')
    candidate.replace(output/'last.pt');history_row(output,row)
    return {**p,'best':best,'last_epoch_record':row,'pending_validation':False}


def reconcile_best(output,identity):
    last=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    best=torch.load(output/'best_region_f2.pt',map_location='cpu',weights_only=False)
    if last['identity']!=identity or best['identity']!=identity or best.get('pending_validation'):raise ValueError('Selected best identity differs')
    metrics=last['last_epoch_record'].get('validation')
    if not last['pending_validation'] and metrics and metrics['region_f2']==last['best']['region_f2'] and metrics['region_f2']>best['last_epoch_record']['validation']['region_f2']:
        copy_atomic(output/'last.pt',output/'best_region_f2.pt')


def status(output,**value):
    save_json(output/'status.json',dict(at=datetime.now().astimezone().isoformat(),pid=os.getpid(),**value))


@contextmanager
def exclusive_writer(output):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'writer.lock').open('a+b') as stream:
        if os.name=='nt':
            import msvcrt
            stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
            try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as error:raise RuntimeError('Another writer holds this output') from error
            try:yield
            finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:yield
            finally:fcntl.flock(stream,fcntl.LOCK_UN)


@contextmanager
def keep_system_awake():
    if os.name!='nt':yield;return
    import ctypes
    if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):raise RuntimeError('Cannot request awake state')
    try:yield
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def scratch_probe(args,root,train,parent,base_config,identity,trainer,study,output):
    from vdcp.kaggle_b5 import OriginalCrops
    from torch.utils.data import DataLoader
    components=make_components(trainer,study,base_config,'cuda:0');model,opt,scheduler,scaler=components
    # Full parent restoration without production initialization or migrated files.
    p=load_state(parent,model,opt,scheduler,scaler,torch.load(parent,map_location='cpu',weights_only=False)['identity'],'cuda:0')
    del p
    snapshot=bn_snapshot(model);durations=[];losses=[]
    loader=DataLoader(OriginalCrops(root,train,768),batch_size=18,num_workers=0,shuffle=False)
    torch.cuda.reset_peak_memory_stats(0)
    for index,(x,y) in enumerate(loader,1):
        started=time.monotonic();loss=update_group(model,opt,scaler,x,y,1,'cuda:0');torch.cuda.synchronize(0)
        durations.append(time.monotonic()-started);losses.append(loss)
        emit(phase='scratch_probe_group',step=index,loss=loss,seconds=durations[-1])
        if index>=args.probe_groups:break
    assert_bn_unchanged(model,snapshot)
    checkpoint=output/'diagnostic_state.pt'
    save_state(checkpoint,model,opt,scheduler,scaler,72,identity,{'region_f2':-1.},base_config,{'epoch':72,'validation':None},device='cuda:0')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    saved['diagnostic_only']=True;atomic_torch(checkpoint,saved)
    # The diagnostic flag is deliberately rejected by production load_state.
    model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler']);restore_rng(saved['rng'],'cuda:0')
    comparisons=dict(model=states_equal(model.state_dict(),saved['model']),optimizer=states_equal(opt.state_dict(),saved['optimizer']),
        scheduler=states_equal(scheduler.state_dict(),saved['scheduler']),scaler=states_equal(scaler.state_dict(),saved['scaler']),rng=states_equal(capture_rng('cuda:0'),saved['rng']))
    if not all(comparisons.values()):raise RuntimeError('Real model save/load state differs')
    assert_bn_unchanged(model,snapshot)
    average=sum(durations[1:] or durations)/len(durations[1:] or durations)
    result=dict(status='passed',diagnostic_only=True,identity=identity,groups=len(losses),losses=losses,
        group_seconds=durations,mean_group_seconds=average,estimated_epoch_seconds=average*math.ceil(len(train)/18),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
        finite_gradients=True,bn_unchanged=True,bn_buffer_count=len(snapshot),exact_save_load=comparisons,
        quality_validated=False,production_mutated=False)
    save_json(output/'preflight.json',result);emit(phase='scratch_probe_complete',**result)
    del saved,components,model,opt,scheduler,scaler,loader,x,y;gc.collect();torch.cuda.empty_cache()
    return result


def runtime_identity():
    import scipy
    return dict(device='cuda:0',gpu=torch.cuda.get_device_name(0),torch=torch.__version__,python=sys.version,
        numpy=np.__version__,opencv=cv2.__version__,cuda=torch.version.cuda,platform=sys.platform,
        scipy=scipy.__version__,cudnn=torch.backends.cudnn.version(),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def run(args):
    from vdcp.data import read_records
    from vdcp.kaggle_b5 import experiment_identity,OriginalCrops
    from torch.utils.data import DataLoader
    root=args.root.resolve();output=(args.probe_output if args.probe_only else args.output).resolve()
    parent=args.parent.resolve();config_path=args.config.resolve();policy=load_config(config_path)
    if not torch.cuda.is_available():raise RuntimeError('Local CUDA device0 required')
    torch.cuda.set_device(0);torch.set_num_threads(2);cv2.setNumThreads(1)
    trainer,study=frozen_helpers(root)
    base_config=json.loads((root/policy['base_config']).read_text(encoding='utf-8'))
    if sha(parent)!=PARENT_SHA or sha(root/base_config['validation_reference_protocol'])!=PROTOCOL_SHA:raise ValueError('Frozen parent/reference digest differs')
    baseline=None if args.skip_baseline_for_probe else read_baseline_gate(args.baseline_gate)
    baseline_sha=None if baseline is None else sha(args.baseline_gate)
    emit(phase='identity_hashing',parent=str(parent),output=str(output))
    base=experiment_identity(root,base_config)
    original=torch.load(parent,map_location='cpu',weights_only=False);verify_parent(original,base,base_config)
    runtime=runtime_identity()
    if baseline is not None:verify_baseline_binding(json.loads(args.baseline_gate.read_text(encoding='utf-8')),base,runtime)
    identity=identity_for(base,runtime,sha(Path(__file__)),sha(config_path),baseline_sha,policy)
    train,val=read_records(root,'train'),read_records(root,'val')
    if (len(train),len(val))!=(8797,1885):raise ValueError('Immutable full development splits required')
    if args.probe_only:
        del original;gc.collect()
        scratch_probe(args,root,train,parent,base_config,identity,trainer,study,output);return
    manifest=output/'run_identity.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text(encoding='utf-8'))!=identity:raise ValueError('Explicit --resume and identical run identity required')
    else:
        if args.resume or (output/'last.pt').exists() or (output/'best_region_f2.pt').exists():raise ValueError('Partial or absent initialization; preserve it and use fresh output')
        # A scratch GPU probe is required before publishing any production state.
        probe_path=args.probe_output/'preflight.json'
        probe=json.loads(probe_path.read_text(encoding='utf-8'))
        if probe.get('status')!='passed' or probe.get('identity')!=identity or not all(probe.get('exact_save_load',{}).values()) or probe.get('groups',0)<2:
            raise ValueError('Matching successful real-model scratch probe required')
        migrated=migrate_parent(original,identity,baseline)
        atomic_torch(output/'last.pt',migrated);copy_atomic(output/'last.pt',output/'best_region_f2.pt')
        save_json(output/'baseline_validation.json',baseline);save_json(manifest,identity)
        del migrated
    del original;gc.collect()
    components=make_components(trainer,study,base_config,'cuda:0');model,opt,scheduler,scaler=components
    p=load_state(output/'last.pt',model,opt,scheduler,scaler,identity,'cuda:0');parent_bn=bn_snapshot(model)
    reconcile_best(output,identity)
    validator_args=(model,root,val,base_config,'cuda:0')
    if p['pending_validation']:
        status(output,status='validation_recovery',current_epoch=p['epoch'],saved_epoch=p['epoch'],target_epoch=args.target_epoch)
        p=finish_validation(output,p,model,opt,scheduler,scaler,validate,validator_args,'cuda:0')
    reconcile_history(output,p)
    if p['epoch']>72 and regression(p['last_epoch_record']['validation'],baseline,policy):
        status(output,status='quality_regression_stop',current_epoch=p['epoch'],saved_epoch=p['epoch'],target_epoch=args.target_epoch);return
    if args.target_epoch>73:
        rows=[json.loads(line) for line in (output/'epochs.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        gate73=next((r for r in rows if r['epoch']==73),None)
        if gate73 is None or not gate73.get('validation') or regression(gate73['validation'],baseline,policy):raise ValueError('Validated passing epoch73 required before target76/80')
    generator=torch.Generator()
    loader=DataLoader(OriginalCrops(root,train,768),batch_size=18,shuffle=True,num_workers=0,pin_memory=True,drop_last=False,generator=generator)
    started=time.monotonic();deadline=started+args.session_hours*3600
    probe=json.loads((args.probe_output/'preflight.json').read_text(encoding='utf-8'))
    estimate=probe['estimated_epoch_seconds'];validation_estimate=baseline.get('seconds',1800.)
    for epoch in range(p['epoch']+1,args.target_epoch+1):
        if (output/'STOP').exists():break
        if deadline-time.monotonic()<1.25*(estimate+validation_estimate)+180:break
        if shutil.disk_usage(output).free<5*1024**3:raise RuntimeError('Checkpoint reserve below 5GiB')
        freeze_batchnorm(model);generator.manual_seed(base_config['seed']+epoch)
        total=0.;samples=0;epoch_start=time.monotonic()
        status(output,status='training',current_epoch=epoch,saved_epoch=p['epoch'],last_validated_epoch=p['epoch'],target_epoch=args.target_epoch)
        for step,(x,y) in enumerate(loader,1):
            loss=update_group(model,opt,scaler,x,y,1,'cuda:0');total+=loss*len(x);samples+=len(x)
            if step==1 or step%10==0:
                event=dict(phase='train',current_epoch=epoch,saved_epoch=p['epoch'],step=step,total_steps=len(loader),loss=total/samples,seconds=time.monotonic()-epoch_start)
                status(output,status='training',target_epoch=args.target_epoch,**event);emit(**event)
        assert_bn_unchanged(model,parent_bn);scheduler.step();estimate=time.monotonic()-epoch_start
        row=dict(epoch=epoch,loss=total/samples,train_seconds=estimate,validation=None,lr=[group['lr'] for group in opt.param_groups],samples=samples,crop_width=768,micro_batch_size=1,effective_batch_size=18)
        save_state(output/'last.pt',model,opt,scheduler,scaler,epoch,identity,p['best'],base_config,row,pending=True,device='cuda:0')
        p={**p,'epoch':epoch,'last_epoch_record':row,'pending_validation':True}
        status(output,status='validation',current_epoch=epoch,saved_epoch=epoch,last_validated_epoch=epoch-1,target_epoch=args.target_epoch)
        p=finish_validation(output,p,model,opt,scheduler,scaler,validate,validator_args,'cuda:0')
        assert_bn_unchanged(model,parent_bn)
        validation_estimate=p['last_epoch_record']['validation']['seconds']
        stopped=regression(p['last_epoch_record']['validation'],baseline,policy)
        status(output,status='quality_regression_stop' if stopped else 'epoch_complete',current_epoch=epoch,saved_epoch=epoch,last_validated_epoch=epoch,target_epoch=args.target_epoch,best=p['best'])
        emit(phase='epoch_complete',epoch=epoch,best=p['best'],quality_gate_passed=not stopped,validation=p['last_epoch_record']['validation'])
        if stopped:return
    # Recheck every immutable input before claiming a completed milestone.
    if experiment_identity(root,base_config)!=base or sha(parent)!=PARENT_SHA or sha(Path(__file__))!=identity['adapter_sha256'] or sha(config_path)!=identity['config_sha256'] or sha(args.baseline_gate)!=baseline_sha:
        raise ValueError('Bound inputs changed during training')
    frozen_helpers(root)
    status(output,status='milestone_complete' if p['epoch']>=args.target_epoch else 'stopped_at_epoch_boundary' if (output/'STOP').exists() else 'session_budget_complete',
        current_epoch=p['epoch'],saved_epoch=p['epoch'],last_validated_epoch=p['epoch'],target_epoch=args.target_epoch,best=p['best'],test_accessed=False)


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/b5_local_768_bnfix_v1')
    parser.add_argument('--probe-output',type=Path,default=ROOT/'work/local_b5_768_bnfix_probe')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/local_b5_768_bnfix.json')
    parser.add_argument('--baseline-gate',type=Path,default=ROOT/'reports/v9/local_768_bnfix/baseline_precision.json')
    parser.add_argument('--parent',type=Path,default=ROOT/'artifacts/cloud_recovery/paired_epoch72/b5_paired_long/control/best_region_f2.pt')
    parser.add_argument('--target-epoch',type=int,choices=(73,76,80),default=73)
    parser.add_argument('--session-hours',type=float,default=10.5)
    parser.add_argument('--probe-groups',type=int,default=3)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--skip-baseline-for-probe',action='store_true')
    args=parser.parse_args(argv)
    if args.resume and args.probe_only:parser.error('--resume cannot be combined with --probe-only')
    if args.target_epoch>73 and not args.resume:parser.error('Targets76/80 require explicit --resume after passing epoch73')
    if args.skip_baseline_for_probe and not args.probe_only:parser.error('Baseline bypass is diagnostic-only')
    if not 0<args.session_hours<=10.5 or not 2<=args.probe_groups<=10:parser.error('Invalid session or scratch group budget')
    if args.output.resolve()==args.probe_output.resolve():parser.error('Probe and production outputs must differ')
    scratch_root=(args.root/'work').resolve()
    if scratch_root not in args.probe_output.resolve().parents:parser.error('Probe output must be a dedicated directory beneath work/')
    if args.probe_only and any((args.probe_output/name).exists() for name in ('run_identity.json','last.pt','best_region_f2.pt')):
        parser.error('Probe cannot overwrite a production checkpoint directory')
    old=(args.root/'outputs/b5_local_crop').resolve()
    if args.output.resolve()==old or old in args.output.resolve().parents:parser.error('Deprecated output cannot be reused')
    return args


def main():
    args=arguments();output=(args.probe_output if args.probe_only else args.output).resolve()
    # All temporary/cache variables affect this process only, and remain on D:.
    for name,folder in dict(TMP='tmp',TEMP='tmp',TMPDIR='tmp',TORCH_HOME='cache/torch',HF_HOME='cache/hf',PIP_CACHE_DIR='cache/pip',UV_CACHE_DIR='cache/uv').items():
        directory=args.root/'work'/folder;directory.mkdir(parents=True,exist_ok=True);os.environ[name]=str(directory)
    # Shared process lock prevents competing probe/production writers even when
    # different output directories are selected; each output also owns a lock.
    with exclusive_writer(args.root/'work/local_b5_768_process_mutex'),exclusive_writer(output),keep_system_awake():
        try:run(args)
        except Exception as error:
            prior=json.loads((output/'status.json').read_text(encoding='utf-8')) if (output/'status.json').exists() else {}
            prior.pop('status',None);prior.pop('at',None);prior.pop('pid',None)
            status(output,**{**prior,'status':'failed','error':repr(error)});raise


if __name__=='__main__':main()
