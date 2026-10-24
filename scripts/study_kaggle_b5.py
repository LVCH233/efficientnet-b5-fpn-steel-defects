"""Validation-only B5 epoch45/60 preservation, calibration and geometric diagnosis."""
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import torch

PARENT_SOURCE='73cad53ba189a4df70950f29920897b287c0762c8b5cb4d134adcddd8e2bee66'
TRAINING_ADAPTER='3ccb91b5b0a1e6252f9b8b2c304b817900ef754e11dca7e33cf90d6f731b42b6'
POLICY='full_model_fp32_with_preserved_grad_scaler'
THRESHOLDS=(.15,.25,.35,.45,.5,.55,.65,.75,.85,.95)
AREAS=(1,16,64,256,1024,4096)
KEYS=('n_gt','n_pred','tp','fp','fn','correct_class')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def emit(**value):
    print(json.dumps(value,allow_nan=False),flush=True)


def copy_verified(source,target,expected):
    target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(target)!=expected: raise ValueError('Preserved checkpoint identity differs')
        return
    temporary=target.with_suffix(target.suffix+'.copying')
    shutil.copyfile(source,temporary)
    if sha(temporary)!=expected: raise ValueError('Checkpoint preservation failed')
    temporary.replace(target)


def restore_partial(previous,output,identity):
    previous,output=Path(previous),Path(output)
    if json.loads((previous/'study_identity.json').read_text())!=identity:
        raise ValueError('Attached study identity differs')
    if output.exists():
        if not (output/'study_identity.json').exists() or json.loads((output/'study_identity.json').read_text())!=identity:
            raise ValueError('Current working study identity differs')
        return
    staging=output.with_name(output.name+'_resume_staging')
    if (staging/'study_identity.json').exists() and json.loads((staging/'study_identity.json').read_text())!=identity:
        raise ValueError('Resume staging identity differs')
    files=[p for p in previous.rglob('*') if p.is_file()]
    missing=sum(max(0,p.stat().st_size-((staging/p.relative_to(previous)).stat().st_size
                if (staging/p.relative_to(previous)).exists() else 0)) for p in files)
    if shutil.disk_usage(output.parent).free<missing+512*1024**2:
        raise RuntimeError('Insufficient disk space to restore previous study output')
    shutil.copytree(previous,staging,dirs_exist_ok=True)
    staging.replace(output)
    emit(phase='previous_study_restored',files=len(files))


def bound_protocol(params,metadata):
    # A generic locked protocol would be accepted by the legacy scorer, which
    # does not implement argmax. Make accidental use there fail explicitly.
    return dict(**{**params,'locked':False,'study_locked':True,'schema':'vdcp_b5_study_protocol_v1',
        'checkpoint_sha256':metadata['checkpoint_sha256'],'validation_csv_sha256':metadata['split_csv_sha256'],
        'inference':metadata['inference'],'input_convention':metadata['input_convention'],
        'precision_policy':POLICY,'probability_dtype':'float16','source_sha256':PARENT_SOURCE,
        'adapter_sha256':sha(Path(__file__)),'calibration_split':'val','region_iou_threshold':.5,
        'connectivity':8,'gt_min_area':1,'postprocessing_adapter':'study_kaggle_b5.prediction',
        'postprocessing_order':'float16 quantization; optional argmax before threshold; component-area filter',
        'deployment_requirement':'Use matching study adapter; generic locked-protocol pipelines must reject this schema'})


def budget(deadline):
    if deadline is not None and time.monotonic()>deadline-120:
        raise TimeoutError('Study session reserve reached; complete caches and partial grid are preserved')


def record_failure(output,error,started):
    # Never publish an output directory merely to log a failed restoration.
    save(Path(output).parent/'b5_study_session_status.json',dict(
        status='budget_exhausted' if isinstance(error,TimeoutError) else 'failed',
        error=repr(error),complete=False,elapsed_seconds=time.monotonic()-started,
        partial_output=str(output)))


def verify_payload(value,base,epoch):
    required={'model','optimizer','scheduler','scaler','rng','epoch','identity','batch_size','best','config','last_epoch_record'}
    if not required.issubset(value) or value.get('diagnostic_only') or value['epoch']!=epoch:
        raise ValueError('Require complete original training checkpoint at expected epoch')
    identity=value['identity']
    expected=dict(base_experiment=base,adapter_sha256=TRAINING_ADAPTER,precision_policy=POLICY,
                  actual_batch_size=value['batch_size'])
    if identity!=expected or base.get('source')!=PARENT_SOURCE or value['batch_size'] not in (18,16):
        raise ValueError('Checkpoint runtime identity differs')
    if not value['model'] or not all(torch.isfinite(v).all().item() for v in value['model'].values()):
        raise ValueError('Checkpoint state must be finite')
    row=value['last_epoch_record'] or {}
    if row.get('epoch')!=epoch or row.get('validation',{}).get('images')!=1885 or row['validation'].get('n_gt')!=3029:
        raise ValueError('Require saved full validation reference at this epoch')


def probability_view(probability,exclusive):
    p=np.asarray(probability).astype(np.float16).astype(np.float32)
    if p.ndim!=3 or p.shape[0]!=4 or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise ValueError('Invalid probability tensor')
    if exclusive:
        winner=p.argmax(axis=0)
        p=np.where(np.arange(4)[:,None,None]==winner[None],p,0.)
    return p


def prediction(probability,thresholds,min_areas,exclusive=False):
    from vdcp.evaluate import postprocess_masks
    return postprocess_masks(probability_view(probability,exclusive),thresholds,min_areas)


def check_reference(actual,expected):
    if any(actual[key]!=expected[key] for key in KEYS):
        raise ValueError(f'Fixed-reference count mismatch: {actual} versus {expected}')


def full_precision_class(base):
    class FP32(base):
        def forward(self,images):
            with torch.autocast(device_type=images.device.type,enabled=False):
                return super().forward(images.float())
    return FP32


def cache_model(root,checkpoint,records,output,device='cuda',deadline=None):
    from vdcp.kaggle_b5 import HistoricalFPN,original_probability
    from vdcp.evaluate import load_probabilities,save_probabilities,validate_cache_metadata
    metadata=dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),split='val',
        split_csv_sha256=sha(root/'data/splits/val.csv'),images=len(records),probability_dtype='float16',
        inference=dict(window=1600,stride=1600,height_window=256,height_stride=192,amp=False,batch_size=1),
        input_convention='OpenCV BGR channel0; historical mean .3439/std .0383',precision_policy=POLICY)
    output.mkdir(parents=True,exist_ok=True)
    meta_path=output/'metadata.json'
    if meta_path.exists() and json.loads(meta_path.read_text())!=metadata:
        raise ValueError('Cache metadata identity differs')
    save(meta_path,metadata); validate_cache_metadata(root,metadata)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=full_precision_class(HistoricalFPN)().to(device).eval()
    model.load_state_dict(payload['model'],strict=True)
    del payload
    started=time.monotonic()
    manifest_path=output/'cache_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    names={Path(r['image_id']).stem+'.npz' for r in records}
    if set(manifest)-names: raise ValueError('Cache manifest identity differs from validation records')
    try:
        for index,row in enumerate(records,1):
            budget(deadline)
            target=output/(Path(row['image_id']).stem+'.npz')
            shape=(4,int(row['height']),int(row['width']))
            if target.exists() and target.name in manifest:
                # Existing files must validate; corruption is never silently reused.
                if sha(target)!=manifest[target.name]: raise ValueError('Cache file identity differs')
                load_probabilities(target,shape)
            else:
                image=cv2.imread(str(root/'data/raw/train_images'/row['image_id']))
                if image is None: raise ValueError('Cannot read validation image '+row['image_id'])
                save_probabilities(target,original_probability(model,image,device),shape)
            manifest[target.name]=sha(target)
            save(manifest_path,manifest)
            if index%100==0 or index==len(records):
                emit(phase='validation_cache',checkpoint=checkpoint.name,images=index,total=len(records),seconds=time.monotonic()-started)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if sha(checkpoint)!=metadata['checkpoint_sha256']: raise ValueError('Checkpoint changed during inference')
    return metadata


def cache_identity(root,cache,records):
    from vdcp.evaluate import validate_cache_metadata
    metadata=json.loads((cache/'metadata.json').read_text())
    validate_cache_metadata(root,metadata)
    if metadata['split']!='val' or metadata['images']!=len(records):
        raise ValueError('Calibration requires the exact validation records')
    manifest={Path(row['image_id']).stem+'.npz':sha(cache/(Path(row['image_id']).stem+'.npz')) for row in records}
    stored=cache/'cache_manifest.json'
    if stored.exists() and json.loads(stored.read_text())!=manifest:
        raise ValueError('Prediction cache content identity differs')
    return dict(metadata=metadata,files=manifest)


def calibrate_grid(root,cache,records,output,exclusive=False,thresholds=THRESHOLDS,areas=AREAS,
                   stop_after=None,deadline=None):
    from vdcp.calibrate import select_candidate
    from vdcp.data import read_masks
    from vdcp.evaluate import load_probabilities,validate_postprocessing
    from vdcp.regions import extract_regions,region_iou,match_iou_matrix
    for t in thresholds:
        for a in areas: validate_postprocessing([t]*4,[a]*4)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    info=dict(cache=cache_identity(root,cache,records),exclusive=exclusive,thresholds=list(thresholds),areas=list(areas),
              image_ids=[r['image_id'] for r in records],study_sha256=sha(Path(__file__)))
    identity=hashlib.sha256(json.dumps(info,sort_keys=True).encode()).hexdigest()
    state_path=output/'grid_progress.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['identity']!=identity: raise ValueError('Calibration progress identity differs')
        grid,start=state['grid'],state['next_index']
        if not isinstance(start,int) or not 0<=start<=len(records): raise ValueError('Invalid calibration progress index')
    else:
        grid=[[dict(class_id=c+1,threshold=float(t),min_area=int(a),n_gt=0,n_pred=0,tp=0,fp=0,fn=0)
               for t in thresholds for a in areas] for c in range(4)]
        start=0
    started=time.monotonic()
    for number in range(start,len(records)):
        budget(deadline)
        row=records[number]; truth=read_masks(root,row['image_id'])
        p=probability_view(load_probabilities(cache/(Path(row['image_id']).stem+'.npz'),truth.shape),exclusive)
        actual_regions=extract_regions(truth)
        binary=np.zeros_like(truth)
        for c in range(4):
            actual=[region for region in actual_regions if region.class_id==c+1]
            for ti,t in enumerate(thresholds):
                binary.fill(0); binary[c]=p[c]>=t
                predicted=extract_regions(binary)
                ious=np.array([[region_iou(a,b) for b in predicted] for a in actual],dtype=float).reshape(len(actual),len(predicted))
                for ai,area in enumerate(areas):
                    kept=[j for j,region in enumerate(predicted) if region.area>=area]
                    matches=match_iou_matrix(ious[:,kept],.5)
                    counts=grid[c][ti*len(areas)+ai]
                    for key,value in dict(n_gt=len(actual),n_pred=len(kept),tp=len(matches),
                        fp=len(kept)-len(matches),fn=len(actual)-len(matches)).items(): counts[key]+=value
        # Atomic durable state after each image, so budget interruption loses none.
        save(state_path,dict(identity=identity,next_index=number+1,total=len(records),grid=grid))
        if (number+1)%20==0 or number+1==len(records):
            elapsed=time.monotonic()-started
            emit(phase='calibration_grid',exclusive=exclusive,images=number+1,total=len(records),seconds=elapsed,
                 estimated_remaining_seconds=elapsed/max(number+1-start,1)*(len(records)-number-1))
        if stop_after is not None and number+1-start>=stop_after and number+1<len(records): return None
    chosen=[select_candidate(rows) for rows in grid]
    metadata=info['cache']['metadata']
    protocol=dict(calibration_split='val',checkpoint_sha256=metadata['checkpoint_sha256'],
        validation_csv_sha256=metadata['split_csv_sha256'],inference=metadata['inference'],probability_dtype='float16',
        region_iou_threshold=.5,connectivity=8,gt_min_area=1,thresholds=[r['threshold'] for r in chosen],
        min_areas=[r['min_area'] for r in chosen],exclusive=exclusive,
        postprocessing_adapter='study_kaggle_b5.prediction',selected_classwise_proxy_results=chosen,
        selection_method='Same-class F2 proxy; final selection uses unchanged global matching',search_identity=identity)
    protocol=bound_protocol(protocol,metadata)
    with (output/'calibration_grid.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(grid[0][0])); writer.writeheader()
        writer.writerows(r for rows in grid for r in rows)
    save(output/'candidate_protocol.json',protocol)
    return protocol


def score_candidates(root,cache,records,candidates,output,deadline=None):
    from vdcp.data import read_masks
    from vdcp.evaluate import load_probabilities,aggregate_counts
    from vdcp.regions import region_counts
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    values={name:[] for name in candidates}; streams={}
    try:
        for name in candidates: streams[name]=(output/(name+'_per_image.jsonl')).open('w')
        for index,row in enumerate(records,1):
            budget(deadline)
            truth=read_masks(root,row['image_id'])
            p=load_probabilities(cache/(Path(row['image_id']).stem+'.npz'),truth.shape)
            for name,params in candidates.items():
                counts=region_counts(truth,prediction(p,params['thresholds'],params['min_areas'],params.get('exclusive',False)))
                values[name].append(counts)
                streams[name].write(json.dumps(dict(image_id=row['image_id'],**counts))+'\n')
            if index%100==0: emit(phase='global_scoring',images=index,total=len(records),candidates=len(candidates))
    finally:
        for stream in streams.values(): stream.close()
    result={}
    for name,counts in values.items():
        item=aggregate_counts(counts)
        item.update(region_f2=5*item['tp']/max(5*item['tp']+4*item['fn']+item['fp'],1),
                    images=len(records),split='val',protocol=candidates[name])
        result[name]=item
        save(output/(name+'_summary.json'),item)
    return result


def select_global(results):
    def rank(item):
        feasible=bool(item.get('recall') is not None and item['recall']>=.98 and item['false_alarm_rate']<=.10
                      and item.get('classification_accuracy') is not None and item['classification_accuracy']>=.90)
        return feasible,item['region_f2'],item.get('classification_accuracy') or 0.,-item['fp']
    return max(results,key=lambda name:rank(results[name]))


def diagnose(root,cache,records,params,expected,output,deadline=None):
    from vdcp.data import read_masks
    from vdcp.evaluate import load_probabilities,draw_prediction
    from vdcp.improvements import diagnose_regions
    from vdcp.regions import extract_regions,region_iou,match_iou_matrix
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    result=dict(counts={k:0 for k in KEYS},fp_reasons={},fn_reasons={},fragmentation_candidates=0,merge_candidates=0,
        fp_on_class_absent_images=[0]*4,image_confusion=dict(tp=0,fp=0,fn=0,tn=0))
    worst=[]
    with (output/'per_image.jsonl').open('w') as stream:
        for index,row in enumerate(records,1):
            budget(deadline)
            truth=read_masks(root,row['image_id'])
            p=load_probabilities(cache/(Path(row['image_id']).stem+'.npz'),truth.shape)
            pred=prediction(p,params['thresholds'],params['min_areas'],params.get('exclusive',False))
            raw=prediction(p,params['thresholds'],[1]*4,params.get('exclusive',False))
            item=diagnose_regions(truth,pred,raw)
            for group in ('counts','fp_reasons','fn_reasons'):
                for key,value in item[group].items():
                    if isinstance(value,int): result[group][key]=result[group].get(key,0)+value
            for key in ('fragmentation_candidates','merge_candidates'): result[key]+=item[key]
            actual,estimated=extract_regions(truth),extract_regions(pred)
            ious=np.array([[region_iou(a,b) for b in estimated] for a in actual],dtype=float).reshape(len(actual),len(estimated))
            matched={j for _,j in match_iou_matrix(ious,.5)}
            for j,region in enumerate(estimated):
                if j not in matched and not np.any(truth[region.class_id-1]): result['fp_on_class_absent_images'][region.class_id-1]+=1
            key=('tp' if pred.any() else 'fn') if truth.any() else ('fp' if pred.any() else 'tn')
            result['image_confusion'][key]+=1
            stream.write(json.dumps(dict(image_id=row['image_id'],**item))+'\n')
            worst.append((item['counts']['fp']+item['counts']['fn'],row['image_id']))
            if index%100==0: emit(phase='error_diagnosis',images=index,total=len(records))
    check_reference(result['counts'],expected)
    if sum(result['fp_reasons'].values())!=expected['fp'] or sum(result['fn_reasons'].values())!=expected['fn']:
        raise ValueError('Diagnosis error partitions do not match official counts')
    for _,image_id in sorted(worst,reverse=True)[:12]:
        budget(deadline)
        image=cv2.imread(str(root/'data/raw/train_images'/image_id))
        truth=read_masks(root,image_id)
        p=load_probabilities(cache/(Path(image_id).stem+'.npz'),truth.shape)
        pred=prediction(p,params['thresholds'],params['min_areas'],params.get('exclusive',False))
        # draw_prediction expects RGB; encode BGR only at the end.
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        panels=[]
        for label,panel in [('Input',rgb),('Ground truth',draw_prediction(rgb,truth)),('Prediction',draw_prediction(rgb,pred))]:
            panel=panel.copy(); cv2.putText(panel,label,(12,22),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            panels.append(panel)
        cv2.imwrite(str(output/(Path(image_id).stem+'.jpg')),cv2.cvtColor(np.concatenate(panels,axis=0),cv2.COLOR_RGB2BGR))
    result.update(images=len(records),split='val',protocol=params,
        interpretation='Geometric error categories; merge/fragmentation flags overlap and are not physical-instance labels')
    save(output/'summary.json',result)
    return result


def source_digest(root):
    digest=hashlib.sha256()
    for file in sorted((root/'src').rglob('*.py'))+[root/'scripts/train_kaggle_b5.py']:
        digest.update(file.relative_to(root).as_posix().encode()); digest.update(file.read_bytes())
    return digest.hexdigest()


def study_models(root,models,references,records,output,fixed,deadline=None,device='cuda'):
    comparison={}; fixed_by_model={}
    # Preserve both full prediction sets before spending the session on CPU grids.
    for name,checkpoint in models.items():
        budget(deadline)
        directory=output/name; directory.mkdir(parents=True,exist_ok=True)
        cache=directory/'cache'
        missing=sum(not (cache/(Path(r['image_id']).stem+'.npz')).exists() for r in records)
        required=sum(4*int(r['height'])*int(r['width'])*2 for r in records
                     if not (cache/(Path(r['image_id']).stem+'.npz')).exists())*1.05+512*1024**2
        if missing and shutil.disk_usage(output).free<required: raise RuntimeError('Insufficient disk reserve for missing prediction caches')
        metadata=cache_model(root,checkpoint,records,cache,device=device,deadline=deadline)
        fixed_by_model[name]=bound_protocol(fixed,metadata)
        original=score_candidates(root,cache,records,{'fixed':fixed_by_model[name]},directory/'fixed',deadline)
        check_reference(original['fixed'],references[name])
        emit(phase='fixed_reference_reproduced',checkpoint=name,**{k:original['fixed'][k] for k in KEYS})
    for name,checkpoint in models.items():
        directory=output/name; cache=directory/'cache'
        candidates={'fixed':fixed_by_model[name]}
        for exclusive in (False,True):
            label='exclusive' if exclusive else 'independent'
            protocol=calibrate_grid(root,cache,records,directory/label,exclusive=exclusive,deadline=deadline)
            for shift in (-.1,0.,.1):
                params={**protocol,'thresholds':[float(np.clip(t+shift,.05,.99)) for t in protocol['thresholds']],
                        'uniform_threshold_shift':shift}
                candidates[f'{label}_{shift:+.1f}']=params
        summaries=score_candidates(root,cache,records,candidates,directory/'global_scores',deadline)
        check_reference(summaries['fixed'],references[name])
        selected=select_global(summaries)
        for candidate in dict.fromkeys(('fixed',selected)):
            diagnose(root,cache,records,candidates[candidate],summaries[candidate],directory/'diagnosis'/candidate,deadline)
        cache_identity(root,cache,records)
        save(directory/'selection.json',dict(selected=selected,summary=summaries[selected],baseline=summaries['fixed'],
            interpretation='Validation development selection; no independent test or leaderboard claim'))
        comparison[name]=dict(selected=selected,summary=summaries[selected],baseline=summaries['fixed'])
        save(output/'comparison_progress.json',comparison)
    return comparison


def run_study(root,training,output,deadline=None,previous=None):
    from vdcp.kaggle_b5 import experiment_identity
    from vdcp.data import read_records
    from vdcp.train import seed_everything
    root,training,output=map(Path,(root,training,output))
    config=json.loads((root/'configs/fpn_v9_kaggle_original.json').read_text())
    base=experiment_identity(root,config)
    seed_everything(config['seed'])
    if base['source']!=PARENT_SOURCE: raise ValueError('Frozen source differs')
    status=json.loads((training/'status.json').read_text())
    if status.get('status')!='milestone_complete' or status.get('completed_epochs')!=60:
        raise ValueError('Require the successful complete60 output')
    records=read_records(root,'val')
    if len(records)!=1885: raise ValueError('Require all1885 validation images')
    inputs={name:sha(training/name) for name in ('best_region_f2.pt','last.pt')}
    identity=dict(base=base,input_hashes=inputs,study_sha256=sha(Path(__file__)),thresholds=THRESHOLDS,areas=AREAS)
    identity=json.loads(json.dumps(identity))
    if previous is not None: restore_partial(previous,output,identity)
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'study_identity.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity: raise ValueError('Study output identity differs')
    save(manifest,identity)
    save(output/'status.json',dict(status='running',phase='input_verification'))
    preserved=output/'preserved'; preserved.mkdir(exist_ok=True)
    for path in training.iterdir():
        if path.is_file() and path.suffix!='.pt': shutil.copyfile(path,preserved/path.name)
    references={}; models={}
    for name,epoch in (('best_region_f2.pt',45),('last.pt',60)):
        source=training/name; target=preserved/name
        payload=torch.load(source,map_location='cpu',weights_only=False)
        verify_payload(payload,base,epoch)
        if hashlib.sha256(json.dumps(payload['config'],sort_keys=True).encode()).hexdigest()!=base['config']:
            raise ValueError('Checkpoint config contents differ')
        references[f'epoch{epoch}']=payload['last_epoch_record']['validation']
        del payload
        copy_verified(source,target,inputs[name])
        models[f'epoch{epoch}']=target
    save(output/'preserved_manifest.json',dict(inputs=inputs,preserved={f.name:sha(f) for f in preserved.iterdir() if f.is_file()}))
    emit(phase='checkpoint_preservation_verified',models=list(models),images=len(records))
    reference=json.loads((root/config['validation_reference_protocol']).read_text())
    fixed=dict(thresholds=reference['thresholds'],min_areas=reference['min_areas'],exclusive=False)
    comparison=study_models(root,models,references,records,output,fixed,deadline)
    if inputs!={name:sha(training/name) for name in ('best_region_f2.pt','last.pt')}:
        raise ValueError('Input checkpoint bytes changed')
    if experiment_identity(root,config)!=base: raise ValueError('Source/data/config changed during study')
    save(output/'comparison.json',dict(status='complete',images=1885,results=comparison,
        test_accessed=False,training_performed=False,selection_scope='development validation only'))
    save(output/'status.json',dict(status='complete',phase='complete',images=1885,models=list(models)))
    emit(phase='study_complete',results=comparison)


def main():
    if not Path('/kaggle/input').exists() or not torch.cuda.is_available():
        raise RuntimeError('Use free Kaggle T4 for this validation-only study')
    started=float(os.environ.get('VDCP_SESSION_STARTED_MONOTONIC',time.monotonic()))
    deadline=started+10.5*3600
    torch.set_num_threads(4); cv2.setNumThreads(1)
    training=list(Path('/kaggle/input').rglob('kaggle_b5_fp32/last.pt'))
    roots=[p.parents[1] for p in Path('/kaggle/input').rglob('configs/fpn_v9_kaggle_original.json')
           if not any(part.endswith('_staging') for part in p.parts)]
    roots=[r for r in roots if source_digest(r)==PARENT_SOURCE]
    if len(training)!=1 or len(roots)!=1:
        raise ValueError('Attach only successful FP32 Training Recovery version1 output with training_source')
    root=roots[0]; sys.path.insert(0,str(root/'src'))
    output=Path('/kaggle/working/b5_posttraining_study')
    previous=[p.parent for p in Path('/kaggle/input').rglob('b5_posttraining_study/study_identity.json')]
    if len(previous)>1: raise ValueError('Attach at most one newest partial study output')
    emit(phase='study_starting',input=str(training[0].parent),root=str(root),training_performed=False)
    try:
        run_study(root,training[0].parent,output,deadline,previous[0] if previous else None)
    except Exception as error:
        record_failure(output,error,started)
        raise


if __name__=='__main__':
    main()
