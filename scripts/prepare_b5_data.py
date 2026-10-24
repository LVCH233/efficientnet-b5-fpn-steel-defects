"""Decode Kaggle RLE annotations into original bit-packed PNG masks without resplitting."""
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
from vdcp.rle import decode_rle
from vdcp.b5_archive import sha256,read_bgr

def load_annotations(path):
    labels=defaultdict(lambda:['']*4);seen=set()
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if 'ImageId_ClassId' in row:name,c=row['ImageId_ClassId'].rsplit('_',1)
            else:name,c=row['ImageId'],row['ClassId']
            if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Unsafe image id')
            c=int(c)
            if c not in range(1,5):raise ValueError('Invalid class')
            if (name,c) in seen:raise ValueError('Duplicate image/class annotation')
            seen.add((name,c));labels[name][c-1]=row['EncodedPixels'] or ''
    return labels

def decode_packed_for_row(row, labels):
    packed=np.zeros((256,1600),np.uint8)
    # Some Kaggle tables omit empty classes and wholly negative images.
    # The archived split flags independently determine whether absence is valid.
    texts=labels.get(row['image_id'], ['']*4)
    for c,text in enumerate(texts):
        mask=decode_rle(text,(256,1600)).astype(np.uint8)
        if int(mask.any())!=int(row[f'c{c+1}']):
            raise ValueError(f"Annotation class presence differs from split: {row['image_id']} class {c+1}")
        packed |= mask<<c
    return packed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--masks',type=Path,required=True)
    p.add_argument('--splits',type=Path,default=ROOT/'data/splits')
    args=p.parse_args();labels=load_annotations(args.annotations)
    rows=[]
    for split in ['train','val','test']:
        with (args.splits/(split+'.csv')).open(encoding='utf-8-sig',newline='') as stream:rows.extend(csv.DictReader(stream))
    if len({r['image_id'] for r in rows})!=len(rows):raise ValueError('Overlapping or duplicate split names')
    args.masks.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows,1):
        name=row['image_id'];path=args.images/name
        read_bgr(path)
        if sha256(path)!=row['sha256']:raise ValueError(f'Image hash differs: {name}')
        packed=decode_packed_for_row(row,labels)
        target=args.masks/(Path(name).stem+'.png')
        if target.exists():
            old=cv2.imdecode(np.fromfile(target,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
            if not np.array_equal(old,packed):raise ValueError(f'Existing mask differs: {target}')
        else:
            ok,encoded=cv2.imencode('.png',packed,[cv2.IMWRITE_PNG_COMPRESSION,3])
            if not ok:raise ValueError(f'Encoding failed: {name}')
            encoded.tofile(target)
        if i%1000==0:print(json.dumps(dict(prepared=i,total=len(rows))),flush=True)
    print(json.dumps(dict(prepared=len(rows),split_policy='preserved archived CSVs')))
if __name__=='__main__':main()
