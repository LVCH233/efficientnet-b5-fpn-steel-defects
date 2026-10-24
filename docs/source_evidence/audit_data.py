"""Read-only data/split/GT audit; writes only data_audit.json beside this script.

No model import, prediction, inference, tuning, or training. Mask reads use one
OpenCV thread. Duplicate checks use existing manifest SHA256 values, not newly
hashed JPEG payloads. Run with the project Python interpreter and -B.
"""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import importlib.util
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
cv2.setNumThreads(1)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def identity(path):
    stat = path.stat()
    return {"path": str(path), "sha256": sha256(path), "bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    started = time.monotonic()
    evaluator = load_module("audit_evaluate_v2", OUT / "evaluate_v2.py")
    splits_mod = load_module("audit_original_splits", ROOT / "src/vdcp/splits.py")
    manifest = read_csv(ROOT / "data/raw/manifest.csv")
    names = ["train", "val", "test"]
    rows = {name: read_csv(ROOT / f"data/splits/{name}.csv") for name in names}
    ids = {name: {r["image_id"] for r in rows[name]} for name in names}
    groups = defaultdict(list)
    for r in manifest:
        groups[r["sha256"]].append(r["image_id"])
    actual_assignment = {r["image_id"]: name for name in names for r in rows[name]}
    expected_assignment = splits_mod.create_splits(manifest, seed=20261008)
    original_summary = json.loads((ROOT / "reports/split_summary.json").read_text(encoding="utf-8"))
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read masks and manifests only; no predictions/inference/tuning or original-file writes",
        "gt_definition": "evaluate_v2._gt_labels: independent per-class 8-connected components, no GT size filtering",
        "small_area_definition": "strict area < threshold in original 1600x256 pixels; cumulative thresholds, not disjoint bins",
        "seed": 20261008, "second_split_seed": 20261009,
        "split_method": "SHA256 groups; iterative multi-label shuffle split 70%/30%, then 50%/50% of held groups",
        "manifest_images": len(manifest), "manifest_unique_ids": len({r["image_id"] for r in manifest}),
        "split_unique_ids": {n: len(ids[n]) for n in names},
        "pairwise_image_id_intersection": {f"{a}/{b}": len(ids[a] & ids[b]) for i, a in enumerate(names) for b in names[i+1:]},
        "split_union_equals_manifest": set().union(*ids.values()) == {r["image_id"] for r in manifest},
        "duplicate_sha256_groups": sum(len(v) > 1 for v in groups.values()),
        "duplicate_extra_images": sum(len(v)-1 for v in groups.values()),
        "sha256_groups_crossing_splits": sum(len({actual_assignment[i] for i in members}) > 1 for members in groups.values()),
        "duplicate_check_boundary": "Manifest-declared SHA256 hashes; JPEG contents were not re-read/rehashed in this audit",
        "split_reproduction_matches_original_seed": actual_assignment == expected_assignment,
        "splits": {},
    }
    cutoffs = [16, 64, 100, 256, 1024]
    for name in names:
        counts = Counter()
        per_class = np.zeros(4, dtype=np.int64)
        pixels = np.zeros(4, dtype=np.int64)
        small = {str(k): np.zeros(4, dtype=np.int64) for k in cutoffs}
        histogram = Counter()
        class_images = np.zeros(4, dtype=np.int64)
        overlap_images = 0
        mismatch_ids = []
        max_bits = 0
        for index, row in enumerate(rows[name], 1):
            path = ROOT / "data/raw/masks" / (Path(row["image_id"]).stem + ".png")
            encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if encoded is None or encoded.shape != (int(row["height"]), int(row["width"])):
                raise ValueError(f"Missing or wrong shape mask: {path}")
            if encoded.ndim != 2 or encoded.dtype != np.uint8 or encoded.max() > 15:
                raise ValueError(f"Invalid encoded 4-bit mask: {path}")
            gt4 = ((encoded[None, :, :] >> np.arange(4)[:, None, None]) & 1).astype(np.uint8)
            active = gt4.any(axis=(1, 2)).astype(int)
            if active.tolist() != [int(row[f"c{c}"]) for c in range(1, 5)]:
                mismatch_ids.append(row["image_id"])
            class_images += active
            counts["positive_images" if active.any() else "negative_images"] += 1
            overlap_images += int((gt4.sum(axis=0) > 1).any())
            _, gcls, garea = evaluator._gt_labels(gt4)
            per_class += np.bincount(gcls[1:], minlength=5)[1:5]
            pixels += gt4.sum(axis=(1, 2)).astype(np.int64)
            histogram[len(garea)-1] += 1
            for limit in cutoffs:
                small[str(limit)] += np.bincount(gcls[1:][garea[1:] < limit], minlength=5)[1:5]
            if index % 1000 == 0:
                print(f"{name}: audited {index}/{len(rows[name])} masks", flush=True)
        file_identity = identity(ROOT / f"data/splits/{name}.csv")
        result["splits"][name] = {
            "images": len(rows[name]), "proportion": len(rows[name]) / len(manifest),
            **counts, "class_image_counts": class_images.tolist(),
            "gt_components": int(per_class.sum()), "gt_components_per_class": per_class.tolist(),
            "foreground_pixels_per_class": pixels.tolist(),
            "small_gt_components": {k: {"total": int(v.sum()), "per_class": v.tolist()} for k, v in small.items()},
            "gt_component_count_image_histogram": dict(sorted(histogram.items())),
            "cross_class_overlap_images": overlap_images,
            "mask_class_presence_mismatch_ids": mismatch_ids,
            "csv_identity": file_identity,
            "csv_sha256_matches_original_summary": file_identity["sha256"] == original_summary[name]["csv_sha256"],
            "class_image_counts_match_original_summary": class_images.tolist() == original_summary[name]["class_image_counts"],
        }
        print(f"{name}: images={len(rows[name])}, GT={per_class.tolist()}, total={int(per_class.sum())}", flush=True)
    result["totals"] = {
        "images": sum(s["images"] for s in result["splits"].values()),
        "gt_components": sum(s["gt_components"] for s in result["splits"].values()),
        "gt_components_per_class": np.sum([s["gt_components_per_class"] for s in result["splits"].values()], axis=0).tolist(),
    }
    result["historical_test_summary_files"] = []
    for path in sorted((ROOT / "reports/experiments").glob("*/test/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        result["historical_test_summary_files"].append({**identity(path), "split": data.get("split"),
            "images": data.get("images"), "n_gt": data.get("n_gt"),
            "created_utc": data.get("created_utc"), "calibration_split": data.get("protocol", {}).get("calibration_split")})
    evidence = ["README.md", "scripts/prepare_data.py", "src/vdcp/splits.py", "src/vdcp/selection.py",
        "data/raw/manifest.csv", "reports/data_audit.json", "reports/split_summary.json",
        "reports/experiments/selection.json", "reports/experiments/baseline_selection.json",
        "reports/improvements/selection.json", "reports/improvements/改进实验报告.md",
        "docs/superpowers/specs/2026-10-09-targeted-improvement-design.md",
        "reports/v3/provisional_selection.json", "PROJECT_HANDOFF_2026-10-16.md",
        "PROJECT_HANDOFF_2026-10-17_768_ONLY.md"]
    result["evidence_file_identities"] = [identity(ROOT / p) for p in evidence] + [identity(OUT / "evaluate_v2.py")]
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    output = OUT / "data_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}; {result['elapsed_seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
