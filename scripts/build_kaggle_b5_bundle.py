"""Build the private Kaggle input, excluding internal/unlabelled test images."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vdcp.data import read_records
from vdcp.kaggle_b5 import check_records, file_sha, experiment_identity


def bundle_files(root, weights):
    train, val = read_records(root, 'train'), read_records(root, 'val')
    check_records(train, val)
    files = [(p, p.relative_to(root).as_posix()) for p in sorted((root/'src').rglob('*.py'))]
    files += [(root/'scripts/train_kaggle_b5.py', 'scripts/train_kaggle_b5.py'),
              (root/'configs/fpn_v9_kaggle_original.json', 'configs/fpn_v9_kaggle_original.json'),
              (root/'src/vdcp/vendor/kh_fpn/LICENSE', 'src/vdcp/vendor/kh_fpn/LICENSE'),
              (root/'src/vdcp/vendor/kh_fpn/upstream_manifest.json', 'src/vdcp/vendor/kh_fpn/upstream_manifest.json'),
              (weights, 'pretrained/efficientnet-b5-b6417697.pth'),
              (root/'reference/locked_protocol.json', 'reference/locked_protocol.json'),
              (root/'reference/group_unified_v2.py', 'reference/group_unified_v2.py')]
    for split in ('train','val'):
        files.append((root/f'data/splits/{split}.csv', f'data/splits/{split}.csv'))
    for row in train+val:
        name = row['image_id']
        for relative in (f'data/raw/train_images/{name}', f'data/raw/masks/{Path(name).stem}.png'):
            files.append((root/relative, relative))
    if len({name for _,name in files}) != len(files):
        raise ValueError('Duplicate archive entry')
    return files


def build(root, weights, output):
    files = bundle_files(root, weights)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'format':1, 'train_images':len(read_records(root,'train')),
                'validation_images':len(read_records(root,'val')),
                'test_images_included':0, 'files':{},
                'internal_test_csv_sha256':file_sha(root/'data/splits/test.csv')}
    if (manifest['train_images'],manifest['validation_images']) != (8797,1885):
        raise ValueError('Unexpected split counts')
    temporary = output.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for index,(path,name) in enumerate(files,1):
            manifest['files'][name] = {'bytes':path.stat().st_size,'sha256':file_sha(path)}
            archive.write(path, 'vdcp_b5/'+name)
            if index%4000==0:
                print(f'Packed {index}/{len(files)} files',flush=True)
        archive.writestr('vdcp_b5/bundle_manifest.json',json.dumps(manifest,indent=2)+'\n')
    temporary.replace(output)
    report = {'archive':str(output.resolve()),'bytes':output.stat().st_size,
              'sha256':file_sha(output), 'train_images':8797,'validation_images':1885,
              'test_images_included':0,'file_count':len(files),
              'weights_sha256':file_sha(weights)}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--weights',type=Path,default=ROOT/'work/kaggle_upstream/efficientnet-b5-b6417697.pth')
    parser.add_argument('--output',type=Path,default=ROOT/'work/kaggle_b5/vdcp_b5_private.zip')
    args=parser.parse_args()
    build(args.root,args.weights,args.output)
