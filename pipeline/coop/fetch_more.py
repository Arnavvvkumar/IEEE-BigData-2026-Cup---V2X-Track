"""Fetch ~150 MORE non-forbidden public V2X-Real training frames (roadside sensor -1 scan + labels of all agents) via ZIP byte
ranges, for VALIDATION / unlabelled-cloud calibration only (Nov 7 organizer notice). They never enter detector training.
Excludes the 100 forbidden frames and the 50 dev frames already held. Spread across scenes, evenly spaced within a scene.
usage: fetch_more.py <out dir>"""
import os
import hashlib
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/real_reference_audit'))
from fetch_training_sample import ARCHIVES, RangeReader, exclusions  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
excluded = exclusions()
have = {r['key'] for r in json.loads((W / 'experiments/real_reference_audit/train_reference50_verified/manifest.json').read_text())['frames']}
assert len(excluded) == 100 and len(have) == 50 and not (excluded & have)


def run(spec):
    name, share, file_id, size, base = spec
    count = base * 3
    url = f'https://ucla.box.com/index.php?rm=box_download_shared_file&shared_name={share}&file_id=f_{file_id}'
    with ZipFile(RangeReader(url, size)) as archive:
        members = {}
        for info in archive.infolist():
            p = PurePosixPath(info.filename)
            if len(p.parts) < 3 or p.suffix not in ('.bin', '.yaml'):
                continue
            scenario, agent, stem = p.parts[-3], p.parent.name, p.stem
            if re.fullmatch(r'20\d{2}(?:-\d{2}){5}_\d+_\d+', scenario) and re.fullmatch(r'\d{6}', stem):
                members[(scenario, agent, stem, p.suffix)] = info
        scenes = defaultdict(list)
        for (scenario, agent, stem, suf) in members:
            key = f'{scenario}/-1/{stem}'
            if agent == '-1' and suf == '.bin' and (scenario, '-1', stem, '.yaml') in members and key not in excluded and key not in have:
                scenes[scenario].append(stem)
        ordered = sorted(scenes)
        quotas = {s: 0 for s in ordered}
        while sum(quotas.values()) < count:
            before = sum(quotas.values())
            for s in ordered:
                if quotas[s] < len(scenes[s]) and sum(quotas.values()) < count:
                    quotas[s] += 1
            if sum(quotas.values()) == before:
                break
        rec = []
        for s, q in quotas.items():
            cand = sorted(scenes[s])
            for i in range(q):
                stem = cand[(2 * i + 1) * len(cand) // (2 * q)]
                row = {'archive': name, 'key': f'{s}/-1/{stem}', 'files': {}}
                assert row['key'] not in excluded and row['key'] not in have
                for (sc, agent, st, suf), info in members.items():
                    if sc == s and st == stem and (suf == '.yaml' or agent == '-1'):
                        data = archive.read(info)
                        rel = Path(name.removesuffix('.zip')) / sc / agent / f'{st}{suf}'
                        (out / rel).parent.mkdir(parents=True, exist_ok=True)
                        (out / rel).write_bytes(data)
                        row['files'][f'{agent}{suf}'] = {'local': rel.as_posix(), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                rec.append(row)
                print(name, len(rec), '/', count, flush=True)
        return rec


with ThreadPoolExecutor(max_workers=4) as ex:
    recs = [r for sub in ex.map(run, ARCHIVES) for r in sub]
keys = {r['key'] for r in recs}
assert keys.isdisjoint(excluded) and keys.isdisjoint(have) and len(keys) == len(recs)
(out / 'manifest.json').write_text(json.dumps({'source': 'UCLA V2X-Real-LiDAR-128 public training split', 'usage': 'validation / unlabelled calibration only; never detector training',
                                               'excluded_forbidden': len(excluded), 'excluded_existing_dev': len(have), 'frames': recs}, indent=1))
print('done frames', len(recs), '| files', sum(len(r['files']) for r in recs))
