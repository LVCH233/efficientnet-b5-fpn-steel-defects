"""Audited cross-runtime continuation: native768 FP32 whole-group training to200."""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
import cloud_b5_compute as compute
import cloud_b5_runtime as runtime

engine=compute.engine
SCHEMA='cloud_b5_768_fp32_epoch200_v1'

def policy_for(source_epoch,micro):
    return dict(schema=SCHEMA,source_epoch=source_epoch,target_epoch=200,crop_width=768,crop_height=256,
        effective_batch_size=18,micro_batch_size=micro,workers=4,validation_batch_size=8,validation_workers=4,
        precision='fp32',validation_precision='fp32_direct',train_encoder=True,train_decoder=True,
        batchnorm='frozen_parent_running_buffers_trainable_affine',loss='0.6_BCE_0.4_global_Dice',
        train_images=8797,validation_images=1885,updates_per_epoch=489,tail_group_size=13,
        quality_stop=False,total_time_limit_hours=None,validate_every=1,
        snapshot_folders=['/root/vdcp-cloud-snapshots'],minimum_free_gib=3,
        cudnn_benchmark=False,cudnn_allow_tf32=True,matmul_allow_tf32=False)

def verify_policy(policy):
    if (not isinstance(policy,dict) or type(policy.get('source_epoch')) is not int or
        not 72<=policy['source_epoch']<200 or policy.get('micro_batch_size') not in compute.SUPPORTED_MICRO_BATCHES or
        policy != policy_for(policy['source_epoch'],policy['micro_batch_size'])):
        raise ValueError('Cloud execution policy differs from audited mathematical configuration')
    return policy

def validate_checkpoint(p,identity=None):
    required={'model','optimizer','scheduler','scaler','rng','epoch','identity','batch_size','best','config',
              'last_epoch_record','pending_validation'}
    if not required.issubset(p) or p.get('diagnostic_only') or (identity is not None and p['identity']!=identity):
        raise ValueError('Complete matching saved state required')
    epoch=p['epoch'];row=p['last_epoch_record'];groups=p['optimizer'].get('param_groups',[])
    if type(epoch) is not int or not 72<=epoch<=200 or p['batch_size']!=18 or p['scheduler'].get('last_epoch')!=epoch or row.get('epoch')!=epoch:
        raise ValueError('Saved epoch/scheduler/record differs')
    ids=[key for group in groups for key in group['params']]
    states=p['optimizer'].get('state',{})
    if len(groups)!=2 or [len(group['params']) for group in groups]!=[501,31] or len(set(ids))!=532 or set(states)!=set(ids):
        raise ValueError('All532 original optimizer states required')
    if any(float(state.get('step',-1))!=35208+(epoch-72)*489 for state in states.values()):
        raise ValueError('Optimizer steps differ from completed epochs')
    cfg=p['config'];index=min(epoch-cfg['flat_epochs'],cfg['anneal_epochs']-1)
    factor=1. if epoch<cfg['flat_epochs'] else 1/(1+math.exp((index-cfg['anneal_epochs']*.5)/(cfg['anneal_epochs']*.1)))
    lr=[cfg['encoder_lr']*factor,cfg['decoder_lr']*factor]
    if any(not math.isclose(g['lr'],v,rel_tol=1e-12,abs_tol=1e-15) for g,v in zip(groups,lr)) or p['scheduler'].get('_last_lr')!=[g['lr'] for g in groups]:
        raise ValueError('Original learning-rate schedule was reset or changed')
    if not all(engine.finite_state(p[key]) for key in ('model','optimizer','scheduler','scaler','best')):
        raise FloatingPointError('Nonfinite checkpoint state')
    if type(p['pending_validation']) is not bool:
        raise ValueError('Explicit validation phase required')
    if p['pending_validation']:
        if row.get('validation') is not None:raise ValueError('Pending checkpoint cannot claim validated metrics')
    elif not isinstance(row.get('validation'),dict) or row['validation'].get('images')!=1885:
        raise ValueError('Complete1885-image validation required')
    if epoch>72 and (row.get('samples')!=8797 or row.get('optimizer_updates')!=489 or row.get('tail_group_size')!=13):
        raise ValueError('Incomplete epoch sample/update accounting')
    return p

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def initial_artifact(path,value,checkpoint=False):
    path=Path(path)
    if path.exists():
        prior=torch.load(path,map_location='cpu',weights_only=False) if checkpoint else json.loads(path.read_text())
        if not engine.states_equal(prior,value):raise ValueError('Existing initialization artifact differs: '+str(path))
        return
    if checkpoint:runtime.atomic_checkpoint(path,value)
    else:runtime.atomic_json(path,value,strict=True)

def ensure_output_reserve(output,checkpoint_bytes,reserve_bytes=3*1024**3):
    if shutil.disk_usage(output).free<2*checkpoint_bytes+reserve_bytes:
        raise RuntimeError('Output filesystem lacks atomic full-checkpoint reserve')

def updated_best(prior,metrics):
    selected=metrics['region_f2']>prior.get('region_f2',-1)
    return dict(region_f2=max(prior.get('region_f2',-1),metrics['region_f2']),
        dice=max(prior.get('dice',-1),metrics['kaggle_style_mean_dice_at_05'])),selected

def report(output,**values):
    value=dict(at=engine.datetime.now().astimezone().isoformat(),pid=os.getpid(),target_epoch=200,**values)
    runtime.atomic_json(Path(output)/'status.json',value)
    engine.emit(**value)

def worker_init(index):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed);cv2.setNumThreads(1);torch.set_num_threads(1)

def configure():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_device(0)

def components(root,config):
    trainer,study=engine.frozen_helpers(root)
    return engine.make_components(trainer,study,config,'cuda:0')

def load_components(root,p):
    model,opt,scheduler,scaler=components(root,p['config'])
    model.load_state_dict(p['model'],strict=True);opt.load_state_dict(copy.deepcopy(p['optimizer']))
    scheduler.load_state_dict(p['scheduler']);scaler.load_state_dict(p['scaler'])
    engine.restore_rng(p['rng'],'cuda:0');engine.freeze_batchnorm(model)
    if not all(v.requires_grad for v in model.parameters()):raise ValueError('Every encoder/decoder parameter must train')
    return model,opt,scheduler,scaler

def saved_payload(model,opt,scheduler,scaler,epoch,identity,best,config,row,pending):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(),rng=engine.capture_rng('cuda:0'),epoch=epoch,identity=identity,
        batch_size=18,best=best.copy(),config=config,last_epoch_record=row,pending_validation=pending)

def verify_inputs(root):
    manifest=json.loads((root/'deployment/inputs_manifest.json').read_text())
    for index,item in enumerate(manifest['files'],1):
        path=root/item['path']
        if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Transferred input hash differs: '+item['path'])
        if index%4000==0:engine.emit(phase='input_verification',files=index,total=len(manifest['files']))
    return dict(files=len(manifest['files']),test_images_accessed=False)

def source_payload(root):
    folder=root/'deployment';manifest=json.loads((folder/'source_manifest.json').read_text())
    for item in manifest['files']:
        if sha(folder/item['file'])!=item['sha256']:raise ValueError('Transferred saved state differs')
    p=torch.load(folder/'source_last.pt',map_location='cpu',weights_only=False)
    import continue_b5_local_768_long as local
    local._check_payload(p,json.loads((folder/'source_identity.json').read_text()))
    validate_checkpoint(p)
    if p['pending_validation']:raise ValueError('Migration must use a fully saved/validated local epoch')
    bn=torch.load(folder/'parent_bn.pt',map_location='cpu',weights_only=False)
    actual={name:value for name,value in p['model'].items() if name in bn}
    if not engine.states_equal(actual,bn):raise ValueError('Original72 BN statistics differ')
    return p,manifest

def probe_loader(root,records,seed):
    from vdcp.kaggle_b5 import OriginalCrops
    generator=torch.Generator().manual_seed(seed)
    indices=torch.randperm(len(records),generator=generator)[:54].tolist()
    return DataLoader(Subset(OriginalCrops(root,records,768),indices),batch_size=18,shuffle=False,num_workers=4,
        worker_init_fn=worker_init,pin_memory=True,prefetch_factor=2,generator=generator,multiprocessing_context='spawn')

def probe(root,output):
    from vdcp.data import read_records
    from vdcp.kaggle_b5 import OriginalCrops
    output.mkdir(parents=True,exist_ok=True)
    inputs=verify_inputs(root);p,manifest=source_payload(root)
    source_epoch=p['epoch'];records=read_records(root,'train')
    loader=probe_loader(root,records,p['config']['seed']+source_epoch+1)
    batches=list(loader)
    del loader
    results=[]
    for micro in compute.SUPPORTED_MICRO_BATCHES:
        model=opt=scheduler=scaler=None
        try:
            model,opt,scheduler,scaler=load_components(root,p)
            before_bn=engine.bn_snapshot(model);before={name:v.detach().cpu().clone() for name,v in model.named_parameters()}
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
            losses=[compute.update_group(model,opt,scaler,x,y,micro,'cuda:0') for x,y in batches]
            torch.cuda.synchronize();seconds=time.monotonic()-started
            engine.assert_bn_unchanged(model,before_bn)
            changed={name:not torch.equal(before[name],v.detach().cpu()) for name,v in model.named_parameters()}
            expected=35208+(source_epoch-72)*489+3
            if not all(changed.values()) or {float(s['step']) for s in opt.state.values()}!={float(expected)}:
                raise ValueError('Real GPU probe did not update every original parameter/state')
            record=dict(micro=micro,status='passed',groups=3,seconds=seconds,seconds_per_group=seconds/3,
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                total_memory=torch.cuda.get_device_properties(0).total_memory,losses=losses,
                changed_parameters=sum(changed.values()),bn_unchanged=True)
            if record['peak_reserved']>record['total_memory']*.94:record['status']='insufficient_memory_margin'
            results.append(record);engine.emit(phase='gpu_probe',**record)
        except torch.cuda.OutOfMemoryError:
            results.append(dict(micro=micro,status='oom'));engine.emit(phase='gpu_probe',micro=micro,status='oom')
        finally:
            del model,opt,scheduler,scaler
            gc.collect();torch.cuda.empty_cache()
    passed=[r for r in results if r['status']=='passed']
    if not passed:raise RuntimeError('No verified physical batch fits with memory reserve')
    selected=min(passed,key=lambda r:r['seconds_per_group'])['micro']
    # The final probe save/load includes all source state components and the next RNG stream.
    model,opt,scheduler,scaler=load_components(root,p)
    x,y=batches[0];compute.update_group(model,opt,scaler,x,y,selected,'cuda:0')
    state=saved_payload(model,opt,scheduler,scaler,p['epoch'],p['identity'],p['best'],p['config'],p['last_epoch_record'],False)
    scratch=root/'work/cloud5090_probe_state.pt';runtime.atomic_checkpoint(scratch,state)
    loaded=torch.load(scratch,map_location='cpu',weights_only=False)
    equality={key:engine.states_equal(state[key],loaded[key]) for key in ('model','optimizer','scheduler','scaler','rng')}
    if not all(equality.values()):raise ValueError('Probe checkpoint roundtrip differs')
    tail_x,tail_y=batches[1][0][:13],batches[1][1][:13]
    tail_loss=compute.update_group(model,opt,scaler,tail_x,tail_y,selected,'cuda:0')
    next_model={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    next_rng=engine.capture_rng('cuda:0')
    del model,opt,scheduler,scaler,state;gc.collect();torch.cuda.empty_cache()
    model,opt,scheduler,scaler=load_components(root,loaded)
    restored_tail_loss=compute.update_group(model,opt,scaler,tail_x,tail_y,selected,'cuda:0')
    next_update_exact=all(torch.equal(value,model.state_dict()[name].detach().cpu()) for name,value in next_model.items())
    for name,value in next_model.items():
        torch.testing.assert_close(value,model.state_dict()[name].detach().cpu(),rtol=1e-5,atol=1e-6)
    if not math.isclose(tail_loss,restored_tail_loss,rel_tol=1e-5,abs_tol=1e-6) or not engine.states_equal(next_rng,engine.capture_rng('cuda:0')):
        raise ValueError('Real GPU saved/restored next tail13 update or RNG differs')
    runtime.atomic_json(output/'preflight.json',dict(status='passed',source_epoch=source_epoch,
        source_checkpoint_sha256=sha(root/'deployment/source_last.pt'),code_sha256=code_hashes(root),
        runtime=engine.runtime_identity(),inputs_manifest_sha256=sha(root/'deployment/inputs_manifest.json'),
        selected_micro=selected,benchmarks=results,exact_save_load=equality,inputs=inputs,
        numerical_probe_updates_applied_to_production=0,real_tail13=True,next_update_exact=next_update_exact,
        next_update_close=True,next_rng_exact=True),strict=True)
    engine.emit(phase='preflight_complete',selected_micro=selected,source_epoch=source_epoch)

def code_hashes(root):
    return {name:sha(root/'scripts'/name) for name in ('continue_b5_cloud5090.py','cloud_b5_compute.py',
        'cloud_b5_runtime.py','cloud_b5_validation.py','continue_b5_local_768.py','train_kaggle_b5.py','study_kaggle_b5.py')}

def initialize(root,output):
    from vdcp.kaggle_b5 import experiment_identity
    preflight=json.loads((output/'preflight.json').read_text())
    if preflight.get('status')!='passed' or not all(preflight['exact_save_load'].values()):raise ValueError('GPU preflight required')
    p,manifest=source_payload(root)
    if (preflight['source_epoch']!=p['epoch'] or preflight['source_checkpoint_sha256']!=sha(root/'deployment/source_last.pt') or
        preflight['code_sha256']!=code_hashes(root) or preflight['runtime']!=engine.runtime_identity() or
        preflight['inputs_manifest_sha256']!=sha(root/'deployment/inputs_manifest.json') or
        not preflight.get('real_tail13') or not preflight.get('next_rng_exact')):
        raise ValueError('GPU acceptance must match exact source checkpoint and cloud implementation')
    policy=verify_policy(policy_for(p['epoch'],preflight['selected_micro']))
    base=experiment_identity(root,p['config'])
    if base!=p['identity']['base_experiment']:raise ValueError('Cloud data/source/config differs from saved experiment')
    identity=dict(schema=SCHEMA,base_experiment=base,policy=policy,runtime=engine.runtime_identity(),
        code_sha256=code_hashes(root),source_checkpoint_sha256=sha(root/'deployment/source_last.pt'),
        source_identity=p['identity'],source_epoch=p['epoch'],target_epoch=200,test_accessed=False,
        stochastic_execution_change='physical packing and worker streams explicitly changed; same FP32 whole-group objective')
    intent_path=output/'initialization_intent.json'
    migration=(json.loads(intent_path.read_text()) if intent_path.exists() else dict(at=engine.datetime.now().astimezone().isoformat(),source_epoch=p['epoch'],
        source_checkpoint_sha256=identity['source_checkpoint_sha256'],training_updates_added=0,
        runtime_changed=True,source_state_preserved=True))
    if migration.get('source_checkpoint_sha256')!=identity['source_checkpoint_sha256'] or migration.get('source_epoch')!=p['epoch']:
        raise ValueError('Initialization intent differs from source checkpoint')
    initial_artifact(intent_path,migration)
    new=dict(p,identity=identity,cloud_migration=migration)
    best=torch.load(root/'deployment/source_best.pt',map_location='cpu',weights_only=False)
    validate_checkpoint(best)
    if best['identity']!=p['identity']:raise ValueError('Source best identity differs')
    best_new=dict(best,identity=identity,cloud_migration=migration)
    unchanged={key:engine.states_equal(p[key],new[key]) for key in p if key!='identity'}
    if not all(unchanged.values()):raise ValueError('Migration changed saved training state')
    initial_artifact(output/'best_region_f2.pt',best_new,checkpoint=True)
    initial_artifact(output/'last.pt',new,checkpoint=True)
    history=(root/'deployment/epochs.jsonl').read_bytes()
    if (output/'epochs.jsonl').exists() and (output/'epochs.jsonl').read_bytes()!=history:raise ValueError('Initialization history differs')
    (output/'epochs.jsonl').write_bytes(history)
    initial_artifact(output/'migration_evidence.json',dict(**migration,state_unchanged=unchanged,
        optimizer_steps=sorted({float(s['step']) for s in p['optimizer']['state'].values()}),
        scheduler_last_epoch=p['scheduler']['last_epoch'],learning_rates=[g['lr'] for g in p['optimizer']['param_groups']],
        best_epoch=best['epoch'],source_files=manifest))
    initial_artifact(output/'policy.json',policy)
    initial_artifact(output/'run_identity.json',identity)
    report(output,status='ready',current_epoch=p['epoch']+1,saved_epoch=p['epoch'],last_validated_epoch=p['epoch'])

def normalized_metrics(value):
    import continue_b5_local_768_long as local
    result=local.normalize_validation_metrics(value);local._metrics(result,complete=True)
    return result

def train(root,output):
    from vdcp.data import read_records
    from vdcp.kaggle_b5 import OriginalCrops
    from cloud_b5_validation import validate
    identity=json.loads((output/'run_identity.json').read_text());policy=verify_policy(identity['policy'])
    if identity['code_sha256']!=code_hashes(root) or identity['runtime']!=engine.runtime_identity():
        raise ValueError('Pinned cloud implementation or runtime differs')
    cfg=json.loads((root/'configs/fpn_v9_kaggle_original.json').read_text())
    # Byte hashes were checked at deployment; source/config/ref stay checked on each restart.
    if sha(root/'reference/locked_protocol.json')!=engine.PROTOCOL_SHA:raise ValueError('Frozen validation protocol differs')
    p=validate_checkpoint(torch.load(output/'last.pt',map_location='cpu',weights_only=False),identity)
    if p['config']!=cfg:raise ValueError('Saved base config differs')
    model,opt,scheduler,scaler=load_components(root,p)
    expected_bn=torch.load(root/'deployment/parent_bn.pt',map_location='cpu',weights_only=False)
    engine.assert_bn_unchanged(model,expected_bn)
    runtime.atomic_json(output/'restoration_evidence.json',dict(epoch=p['epoch'],scheduler_last_epoch=scheduler.last_epoch,
        optimizer_steps=sorted({float(s['step']) for s in opt.state.values()}),bn_unchanged=True,
        trainable_parameters=sum(1 for v in model.parameters() if v.requires_grad),all_parameters_trainable=True,
        learning_rates=[g['lr'] for g in opt.param_groups],micro_batch_size=policy['micro_batch_size']),strict=True)
    train_records=read_records(root,'train');val_records=read_records(root,'val')
    if (len(train_records),len(val_records))!=(8797,1885):raise ValueError('Full development splits required')
    folders=[Path(v) for v in policy['snapshot_folders']]+[output/'checkpoints']
    def validation(epoch):
        report(output,status='validation',current_epoch=epoch,saved_epoch=epoch,last_validated_epoch=epoch-1)
        def progress(event):
            report(output,status='validation',current_epoch=epoch,saved_epoch=epoch,last_validated_epoch=epoch-1,
                validation_done=event['images'],validation_total=1885,seconds=event['seconds'])
        state=engine.capture_rng('cuda:0')
        try:return normalized_metrics(validate(model,root,val_records,cfg,'cuda:0',
            batch_size=policy['validation_batch_size'],workers=policy['validation_workers'],progress=progress))
        finally:engine.restore_rng(state,'cuda:0')
    def finalize(saved):
        row=saved['last_epoch_record'];row['validation']=validation(saved['epoch'])
        best,selected=updated_best(saved['best'],row['validation'])
        row['best']=best.copy()
        engine.assert_bn_unchanged(model,expected_bn)
        result=saved_payload(model,opt,scheduler,scaler,saved['epoch'],identity,best,cfg,row,False)
        validate_checkpoint(result,identity)
        runtime.commit_epoch(output,result,row,is_best=selected,snapshot_folders=folders,reserve_bytes=3*1024**3)
        return result
    if not (output/'cloud_rebaseline.json').exists() and p['epoch']==identity['source_epoch']:
        metrics=validation(p['epoch'])
        runtime.atomic_json(output/'cloud_rebaseline.json',dict(epoch=p['epoch'],cloud=metrics,
            source=p['last_epoch_record']['validation'],training_state_unchanged=True,
            note='same saved weights; Linux/Torch2.8/Blackwell and validation packing changed'),strict=True)
    if p['pending_validation']:p=finalize(p)
    else:runtime.recover_publication(output,snapshot_folders=folders,reserve_bytes=3*1024**3)
    for epoch in range(p['epoch']+1,201):
        if (output/'STOP').exists():break
        ensure_output_reserve(output,(output/'last.pt').stat().st_size)
        generator=torch.Generator().manual_seed(cfg['seed']+epoch)
        loader=DataLoader(OriginalCrops(root,train_records,768),batch_size=18,shuffle=True,
            num_workers=policy['workers'],worker_init_fn=worker_init,pin_memory=True,prefetch_factor=2,generator=generator,
            multiprocessing_context='spawn')
        total=0.;samples=0;started=time.monotonic();last=0.
        report(output,status='training',current_epoch=epoch,saved_epoch=p['epoch'],last_validated_epoch=p['epoch'],step=0,total_steps=489)
        for step,(x,y) in enumerate(loader,1):
            loss=compute.update_group(model,opt,scaler,x,y,policy['micro_batch_size'],'cuda:0')
            total+=loss*len(x);samples+=len(x);tail=len(x)
            if step==1 or step%10==0 or time.monotonic()-last>20:
                report(output,status='training',current_epoch=epoch,saved_epoch=p['epoch'],last_validated_epoch=p['epoch'],
                    step=step,total_steps=489,loss=total/samples,seconds=time.monotonic()-started,
                    samples=samples,micro_batch_size=policy['micro_batch_size']);last=time.monotonic()
        if (samples,step,tail)!=(8797,489,13):raise ValueError('Incomplete training epoch')
        scheduler.step();engine.assert_bn_unchanged(model,expected_bn)
        row=dict(epoch=epoch,loss=total/samples,samples=samples,optimizer_updates=step,tail_group_size=tail,
            train_seconds=time.monotonic()-started,lr=[g['lr'] for g in opt.param_groups],validation=None,
            execution=dict(micro_batch_size=policy['micro_batch_size'],workers=policy['workers'],precision='fp32'))
        p=saved_payload(model,opt,scheduler,scaler,epoch,identity,p['best'],cfg,row,True)
        validate_checkpoint(p,identity);runtime.atomic_checkpoint(output/'last.pt',p)
        p=finalize(p)
        report(output,status='epoch_complete',current_epoch=epoch,saved_epoch=epoch,last_validated_epoch=epoch,
            validation_done=1885,validation_total=1885,metrics=row['validation'],best=p['best'])
        del loader;gc.collect()
    report(output,status='milestone_complete' if p['epoch']==200 else 'stopped_at_epoch_boundary',
        current_epoch=p['epoch'],saved_epoch=p['epoch'],last_validated_epoch=p['epoch'],validation_done=1885,
        validation_total=1885,best=p['best'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe',action='store_true');parser.add_argument('--initialize',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    args.root=args.root.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    configure()
    with runtime.ProcessLock(args.output/'writer.lock'):
        try:
            if args.probe:probe(args.root,args.output)
            elif args.initialize:initialize(args.root,args.output)
            elif args.resume:train(args.root,args.output)
            else:parser.error('Explicit --probe, --initialize or --resume required')
        except Exception as error:
            prior={}
            try:prior=json.loads((args.output/'status.json').read_text())
            except (OSError,ValueError):pass
            for key in ('pid','at','target_epoch','status'):prior.pop(key,None)
            report(args.output,**{**prior,'status':'failed','error':repr(error)})
            raise

if __name__=='__main__':main()
