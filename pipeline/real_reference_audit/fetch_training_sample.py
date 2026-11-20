"""Retrieve 50 explicitly allowed infrastructure scan/label pairs via ZIP byte ranges."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile


ARCHIVES = [
    ('train1.zip', 'hwpc7uepoxm4wne7b3aspy13jie6t24d', 1668580274387, 11924436997, 13),
    ('train2.zip', 'cju53lgfqauucmbo0l7wn4umqv0mrvt5', 1668583086234, 10818096799, 13),
    ('train3.zip', 'kpblm90tbgu0okx00tj9rydozky6q6ag', 1692718880602, 10403114237, 12),
    ('train4.zip', '75o3cjdceupby9o52jrl5bhl7e6kk8ov', 1692719408915, 8180373064, 12),
]
HERE = Path(__file__).resolve().parent


class RangeReader:
    def __init__(self, url, size):
        self.url = url
        self.redirect = url
        self.size = size
        self.offset = 0

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        self.offset = (0, self.offset, self.size)[whence] + offset
        if self.offset < 0:
            raise ValueError('Negative ZIP offset')
        return self.offset

    def read(self, count=-1):
        start = self.offset
        stop = self.size if count < 0 else min(self.size, start + count)
        if stop <= start:
            return b''
        if stop - start > 16 * 2**20:
            raise ValueError('Unexpectedly large ZIP range')
        end = stop - 1
        for attempt in range(4):
            request = Request(self.redirect, headers={'Range': f'bytes={start}-{end}',
                                                      'Accept-Encoding': 'identity'})
            try:
                with urlopen(request, timeout=25) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                        raise ValueError('Publisher did not return the exact requested ZIP range')
                    data = response.read()
                    if len(data) != stop - start:
                        raise OSError('Short ZIP range')
                    self.redirect = response.geturl()
                self.offset = stop
                return data
            except OSError:
                self.redirect = self.url
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError('Unreachable retry state')


def exclusions():
    kit = HERE.parents[1] / 'starting_kit/track_b/forbidden_frames.txt'
    ids = {line.strip() for line in kit.read_text().splitlines() if re.fullmatch(r'\d+_\d{6}', line.strip())}
    with (HERE / 'v2x_real_forbidden_frames_mapping.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(ids) != 100 or len(rows) != 100 or ids != {row['forbidden_id'] for row in rows}:
        raise ValueError('The official kit and organizer table disagree')
    keys = set()
    for row in rows:
        index, frame = row['forbidden_id'].split('_')
        key = f"{row['scenario_folder']}/-1/{frame}"
        if (row['scenario_index'] != index or row['frame'] != frame or
                row['point_cloud_file'] != key + '.bin' or row['label_file'] != key + '.yaml'):
            raise ValueError('Inconsistent organizer path mapping')
        keys.add(key)
    if len(keys) != 100:
        raise ValueError('Expected 100 unique forbidden paths')
    return keys


def download_subset(spec, excluded, output):
    name, share, file_id, size, count = spec
    url = ('https://ucla.box.com/index.php?rm=box_download_shared_file'
           f'&shared_name={share}&file_id=f_{file_id}')
    with ZipFile(RangeReader(url, size)) as archive:
        paths = {}
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if len(path.parts) < 3 or path.suffix not in ('.bin', '.yaml') or path.parent.name != '-1':
                continue
            scenario, agent, stem = path.parts[-3], path.parent.name, path.stem
            if not re.fullmatch(r'20\d{2}(?:-\d{2}){5}_\d+_\d+', scenario) or not re.fullmatch(r'\d{6}', stem):
                continue
            key = f'{scenario}/{agent}/{stem}'
            if key not in excluded:
                entry = paths.setdefault(key, {})
                if path.suffix in entry:
                    raise ValueError(f'Duplicate training member: {key}{path.suffix}')
                entry[path.suffix] = info
        scenes = defaultdict(list)
        for key, pair in sorted(paths.items()):
            if len(pair) == 2:
                scenes[key.split('/')[0]].append(key)
        if not scenes:
            raise ValueError(f'No eligible paired roadside frames in {name}')
        # Spread across scenes first and within each scene by taking evenly spaced frames.
        selected = []
        ordered = sorted(scenes)
        quotas = {scene: 0 for scene in ordered}
        while sum(quotas.values()) < count:
            previous = sum(quotas.values())
            for scene in ordered:
                if quotas[scene] < len(scenes[scene]) and sum(quotas.values()) < count:
                    quotas[scene] += 1
            if sum(quotas.values()) == previous:
                raise ValueError(f'Insufficient eligible frames in {name}')
        for scene, quota in quotas.items():
            candidates = scenes[scene]
            selected.extend(candidates[(2 * i + 1) * len(candidates) // (2 * quota)] for i in range(quota))
        manifest = []
        for key in selected:
            record = {'archive': name, 'key': key, 'files': {}}
            for suffix, info in sorted(paths[key].items()):
                # ZipFile validates CRC when reading each selected member.
                data = archive.read(info)
                relative = Path(name.removesuffix('.zip')) / (key + suffix)
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                record['files'][suffix] = {'local': relative.as_posix(), 'source': info.filename,
                                           'sha256': hashlib.sha256(data).hexdigest()}
            manifest.append(record)
            print(f'{name} verified {len(manifest)}/{count}', flush=True)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    excluded = exclusions()
    args.output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda spec: download_subset(spec, excluded, args.output), ARCHIVES)
        records = [frame for subset in results for frame in subset]
    keys = {frame['key'] for frame in records}
    if len(keys) != 50 or not keys.isdisjoint(excluded):
        raise ValueError('Expected 50 unique non-forbidden frames')
    (args.output / 'manifest.json').write_text(json.dumps({
        'source': 'UCLA V2X-Real-LiDAR-128 public training split',
        'usage': 'realism reference and evaluation only', 'verified_exclusions': len(excluded),
        'frames': records,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
