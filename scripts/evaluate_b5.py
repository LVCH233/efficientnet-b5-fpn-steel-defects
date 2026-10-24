"""Evaluate B5 with the pinned group protocol; external image/mask paths required."""
import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vdcp.b5_archive import load_model,read_bgr,predict,packed_masks,MODEL_SHA256,sha256
from vdcp.group_evaluation import score_final_masks,summarize_counts,COUNT_KEYS,EVALUATOR_SHA256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--masks',type=Path,required=True)
    p.add_argument('--split',type=Path,default=ROOT/'data/splits/val.csv')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--limit',type=int,help='Diagnostic subset; omitted for full evaluation')
    args=p.parse_args()
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    with args.split.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    if args.limit:rows=rows[:args.limit]
    names=[r['image_id'] for r in rows]
    if len(names)!=len(set(names)) or any(Path(n).name!=n or '/' in n or '\\' in n for n in names):raise ValueError('Duplicate or unsafe image identifiers')
    model=load_model(args.checkpoint,args.device);counts={k:0 for k in COUNT_KEYS};started=time.monotonic()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.with_suffix('.images.jsonl').open('w',encoding='utf-8') as stream:
        for i,row in enumerate(rows,1):
            name=row['image_id'];image=args.images/name
            if row.get('sha256') and sha256(image)!=row['sha256']:raise ValueError(f'Image SHA256 differs: {name}')
            _,pred=predict(model,read_bgr(image),args.device)
            result=score_final_masks(packed_masks(args.masks/(Path(name).stem+'.png')),pred)
            for k in COUNT_KEYS:counts[k]+=result[k]
            stream.write(json.dumps(dict(image_id=name,**result))+'\n');stream.flush()
            if i%25==0:print(json.dumps(dict(images=i,total=len(rows),seconds=time.monotonic()-started)),flush=True)
    result=summarize_counts(counts)
    result.update(n_images=len(rows),diagnostic_subset=args.limit is not None,split_sha256=sha256(args.split),checkpoint_sha256=MODEL_SHA256,evaluator_sha256=EVALUATOR_SHA256,seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=True)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
