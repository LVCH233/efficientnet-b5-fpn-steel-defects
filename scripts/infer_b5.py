"""Infer a native-resolution image using the formal epoch135 model."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from vdcp.b5_archive import load_model, read_bgr, predict, MODEL_SHA256, THRESHOLDS, MIN_AREAS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    model=load_model(args.checkpoint,args.device)
    probabilities,masks=predict(model,read_bgr(args.image),args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,probabilities=probabilities,masks=masks)
    result=dict(image=args.image.name,shape=list(probabilities.shape),dtype=str(probabilities.dtype),mask_pixels=masks.sum(axis=(1,2)).tolist(),checkpoint_sha256=MODEL_SHA256,thresholds=THRESHOLDS,min_areas=MIN_AREAS,fp32=True,batch_size=1,tta=False)
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
