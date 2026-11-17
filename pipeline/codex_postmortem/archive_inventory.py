"""Audit submission lineage without inspecting individual test predictions."""

import hashlib
import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cloud_hashes(files):
    return sorted(digest(data) for data in files)


def inspect(archive, sources):
    with ZipFile(archive) as z:
        paths = sorted(p for p in z.namelist() if re.fullmatch(r'synthetic/[^/]+\.bin', p))
        clouds = [z.read(p) for p in paths]
        raw = z.read('predictions.json')
        detections = dict.fromkeys(('vehicle', 'pedestrian', 'truck'), 0)
        predictions = json.loads(raw)
        for frame in predictions.values():
            for box in frame:
                key = box['class']
                detections[key] = detections.get(key, 0) + 1
        hashes = cloud_hashes(clouds)
        return {
            'archive': archive.name,
            'source': [name for name, expected in sources.items() if hashes == expected],
            'synthetic_frames': len(clouds),
            'raw_xyz_rows': sum(map(len, clouds)) // 12,
            'uncompressed_bytes': sum(i.file_size for i in z.infolist()),
            'max_bin_bytes': max(map(len, clouds), default=0),
            'frames_with_prediction_entries': len(predictions),
            'detections': detections,
            'predictions_sha256': digest(raw),
            'synthetic_sha256': digest(''.join(hashes).encode()),
            'ordered_synthetic_sha256': digest(''.join(digest(data) for data in clouds).encode()),
        }


def main():
    twin = Path(__file__).resolve().parents[1] / 'twin'
    sources = {
        name: cloud_hashes(p.read_bytes() for p in (twin / name / 'synthetic').glob('*.bin'))
        for name in ('final_synthetic_v3', 'final_synthetic_v4')
    }
    for path in sorted(Path(sys.argv[1]).glob('bdc2026_trackB_submission*.zip')):
        print(json.dumps(inspect(path, sources)), flush=True)


if __name__ == '__main__':
    main()
