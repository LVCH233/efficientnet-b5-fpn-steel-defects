"""Explicit new-lineage 768 continuation using the byte-pinned two-pass engine.

This does not claim the historical parent identity. It records the parent digest
and a new source/data identity while preserving available optimizer and RNG state.
"""
import argparse,csv,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.utils.data import DataLoader
import cloud_b5_compute as compute
from vdcp.kaggle_b5 import OriginalCrops,file_sha
from vdcp.b5_archive import predict,read_bgr,packed_masks
from vdcp.group_evaluation import score_final_masks,summarize_counts,COUNT_KEYS
engine=compute.engine

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--parent-sha256',required=True)
    p.add_argument('--data-root',type=Path,required=True,help='Contains data/splits and data/raw/{train_images,masks}')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-epoch',type=int,required=True)
    p.add_argument('--micro-batch',type=int,default=1,choices=[1,3,6,9,18])
    p.add_argument('--device',default='cuda:0',choices=['cuda:0'],help='Frozen RNG mapping requires CUDA device0')
    args=p.parse_args()
    if file_sha(args.parent)!=args.parent_sha256:raise ValueError('Parent SHA256 differs')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use an empty output; pass its last.pt as a new explicit parent to resume')
    payload=torch.load(args.parent,map_location='cpu',weights_only=False)
    required={'model','optimizer','scheduler','scaler','rng','config','epoch'}
    if not required.issubset(payload):raise ValueError('Complete training state required; model-only epoch200 cannot resume')
    if payload.get('pending_validation'):raise ValueError('Parent has pending validation; finish it in its original lineage first')
    if not int(payload['epoch'])<args.target_epoch<=200:p.error('target must exceed parent epoch and be at most 200')
    trainer,study=engine.frozen_helpers(ROOT)
    model,opt,scheduler,scaler=engine.make_components(trainer,study,payload['config'],args.device)
    model.load_state_dict(payload['model'],strict=True);opt.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler']);scaler.load_state_dict(payload['scaler'])
    for state in opt.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor):state[key]=value.to(args.device)
    engine.restore_rng(payload['rng'],args.device)
    root=args.data_root.resolve()
    def records(split):
        with (root/'data/splits'/f'{split}.csv').open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
    train,val=records('train'),records('val')
    from vdcp.kaggle_b5 import check_records
    check_records(train,val)
    if len(train)!=8797 or len(val)!=1885:raise ValueError('Expected archived train/val counts')
    for split in ['train','val','test']:
        if file_sha(root/'data/splits'/f'{split}.csv')!=file_sha(ROOT/'data/splits'/f'{split}.csv'):raise ValueError('Training split differs from archive')
    assets=[]
    for row in sorted(train+val,key=lambda r:r['image_id']):
        name=row['image_id'];image_sha=file_sha(root/'data/raw/train_images'/name)
        if image_sha!=row['sha256']:raise ValueError('Training image SHA256 differs')
        mask_sha=file_sha(root/'data/raw/masks'/(Path(name).stem+'.png'))
        assets.append(dict(image_id=name,image_sha256=image_sha,mask_sha256=mask_sha))
    assets_sha=hashlib.sha256(json.dumps(assets,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    identity=dict(schema='b5_archive_new_lineage_v1',training_assets_sha256=assets_sha,parent_sha256=args.parent_sha256,parent_identity=payload.get('identity'),source={str(f.relative_to(ROOT)):file_sha(f) for d in ['src','scripts'] for f in sorted((ROOT/d).rglob('*.py'))},splits={n:file_sha(root/'data/splits'/f'{n}.csv') for n in ['train','val','test']},crop_width=768,effective_batch=18,micro_batch=args.micro_batch,validation='canonical group evaluator; FP32 sigmoid batch1 no TTA')
    args.output.mkdir(parents=True,exist_ok=True);engine.save_json(args.output/'lineage.json',identity);engine.save_json(args.output/'data_assets.json',assets)
    bn=engine.bn_snapshot(model)
    generator=torch.Generator()
    loader=DataLoader(OriginalCrops(root,train,768),batch_size=18,shuffle=True,num_workers=0,pin_memory=True,drop_last=False,generator=generator)
    for epoch in range(int(payload['epoch'])+1,args.target_epoch+1):
        engine.freeze_batchnorm(model);generator.manual_seed(payload['config']['seed']+epoch);total=0.;samples=0
        for x,y in loader:
            # Same whole-group quotient and tail normalization as frozen code.
            loss=engine.update_group(model,opt,scaler,x,y,1,args.device) if args.micro_batch==1 else compute.update_group(model,opt,scaler,x,y,args.micro_batch,args.device)
            total+=float(loss)*len(x);samples+=len(x)
        engine.assert_bn_unchanged(model,bn);scheduler.step()
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=engine.capture_rng(args.device),config=payload['config'],epoch=epoch,identity=identity,pending_validation=True)
        engine.atomic_torch(args.output/'last.pt',state)
        model.eval();counts={k:0 for k in COUNT_KEYS}
        for row in val:
            name=row['image_id'];_,masks=predict(model,read_bgr(root/'data/raw/train_images'/name),args.device)
            metrics=score_final_masks(packed_masks(root/'data/raw/masks'/(Path(name).stem+'.png')),masks)
            for k in COUNT_KEYS:counts[k]+=metrics[k]
        engine.restore_rng(state['rng'],args.device)
        row=dict(epoch=epoch,loss=total/samples,samples=samples,validation=summarize_counts(counts))
        state.update(pending_validation=False,last_epoch_record=row)
        engine.atomic_torch(args.output/'last.pt',state);engine.atomic_torch(args.output/f'epoch{epoch}.pt',state)
        engine.save_json(args.output/f'epoch{epoch}.json',row);print(json.dumps(row),flush=True)
if __name__=='__main__':main()
