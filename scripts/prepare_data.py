"""Prepare the exact user-supplied dataset; no test labels are invented."""
import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vdcp.preparation import safe_extract
from vdcp.rle import decode_rle


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def extract(source):
    raw = ROOT / 'data/raw'
    scratch = ROOT / 'work/archives'
    raw.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    with ZipFile(source) as outer:
        for name in ['train.csv', 'sample_submission.csv']:
            data = outer.read(name)
            target = raw / name
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Existing metadata differs: {target}')
            target.write_bytes(data)
        for name in ['train_images.zip', 'test_images.zip']:
            local = scratch / name
            if not local.exists():
                print(f'Extract nested archive {name}', flush=True)
                with outer.open(name) as origin, local.open('wb') as target:
                    shutil.copyfileobj(origin, target, length=1024*1024)
            if local.stat().st_size != outer.getinfo(name).file_size:
                raise ValueError(f'Incomplete nested archive: {local}')
            print(f'Extract and verify {name}', flush=True)
            safe_extract(local, raw / name.replace('.zip', ''))


def audit(source):
    raw = ROOT / 'data/raw'
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    records = defaultdict(lambda: [''] * 4)
    with (raw / 'train.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            image_id, class_id = row['ImageId_ClassId'].rsplit('_', 1)
            records[image_id][int(class_id)-1] = row['EncodedPixels']
    files = {path.name for path in (raw / 'train_images').glob('*.jpg')}
    if files != set(records):
        raise ValueError(f'Image/label mismatch: missing={len(set(records)-files)}, extra={len(files-set(records))}')
    mask_dir = raw / 'masks'
    mask_dir.mkdir(exist_ok=True)
    manifest, sizes, class_counts, pixel_counts = [], Counter(), np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64)
    positive = 0
    for number, image_id in enumerate(sorted(records), start=1):
        path = raw / 'train_images' / image_id
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if (width, height) != (1600, 256):
                raise ValueError(f'Unexpected image size {image_id}: {(width, height)}')
        sizes[f'{width}x{height}'] += 1
        encoded = np.zeros((height, width), dtype=np.uint8)
        labels = []
        for channel, text in enumerate(records[image_id]):
            mask = decode_rle(text, (height, width))
            active = int(mask.any())
            labels.append(active)
            class_counts[channel] += active
            pixel_counts[channel] += int(mask.sum())
            encoded |= mask << channel
        positive += int(any(labels))
        Image.fromarray(encoded).save(mask_dir / (Path(image_id).stem + '.png'), compress_level=3)
        manifest.append(dict(image_id=image_id, width=width, height=height, sha256=file_hash(path),
            **{f'c{i+1}': labels[i] for i in range(4)}))
        if number % 1000 == 0:
            print(f'Audited {number}/{len(records)} labeled images', flush=True)
    test_files = sorted((raw / 'test_images').glob('*.jpg'))
    for path in test_files:
        with Image.open(path) as image:
            image.load()
            if image.size != (1600, 256):
                raise ValueError(f'Unexpected test image size: {path}')
    with (raw / 'manifest.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    groups = Counter(item['sha256'] for item in manifest)
    result = dict(source=str(source), source_sha256=file_hash(source), train_csv_sha256=file_hash(raw/'train.csv'),
        labeled_images=len(manifest), unlabeled_test_images=len(test_files), positive_images=positive,
        negative_images=len(manifest)-positive, class_image_counts=class_counts.tolist(),
        class_pixel_counts=pixel_counts.tolist(), dimensions=dict(sizes),
        duplicate_sha256_groups=sum(count>1 for count in groups.values()),
        duplicate_extra_images=sum(count-1 for count in groups.values()),
        invalid_images=0, invalid_rle=0, test_labels_available=False,
        class_semantics='Numeric IDs 1-4; physical defect names are not supplied')
    (reports / 'data_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


def split():
    from vdcp.splits import create_splits
    with (ROOT/'data/raw/manifest.csv').open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    assignments = create_splits(records)
    summary = {}
    target = ROOT / 'data/splits'
    target.mkdir(parents=True, exist_ok=True)
    for name in ['train', 'val', 'test']:
        selected = [dict(row, split=name) for row in records if assignments[row['image_id']] == name]
        with (target / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        summary[name] = dict(images=len(selected),
            class_image_counts=[sum(int(row[f'c{i}']) for row in selected) for i in range(1,5)],
            negative_images=sum(not any(int(row[f'c{i}']) for i in range(1,5)) for row in selected),
            csv_sha256=file_hash(target/f'{name}.csv'))
    summary['seed'] = 20261008
    summary['method'] = 'SHA256-grouped iterative multi-label stratification; 70/15/15 at group level'
    (ROOT/'reports/split_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['extract','audit','split','all'], default='all')
    parser.add_argument('--source', type=Path, default=ROOT/'机器视觉检测数据集/steel-de.zip')
    args = parser.parse_args()
    if args.stage in ['extract','all']:
        extract(args.source)
    if args.stage in ['audit','all']:
        audit(args.source)
    if args.stage in ['split','all']:
        split()
