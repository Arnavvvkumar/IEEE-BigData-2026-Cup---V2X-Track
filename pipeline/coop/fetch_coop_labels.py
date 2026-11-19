"""Fetch the OTHER agents' label files (.yaml only) for the 50 non-forbidden dev frames, via ZIP byte ranges from the public
V2X-Real training archives. Purpose: test whether the organizers' test labels are the cooperative (multi-agent) labels.
Labels are used for validation only (allowed by the Nov 7 notice). No forbidden frame is touched.
usage: fetch_coop_labels.py <out dir>"""
import os
import json
import re
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

W = Path(os.environ['BDC_ROOT'])
sys.path.insert(0, str(W / 'experiments/real_reference_audit'))
from fetch_training_sample import ARCHIVES, RangeReader, exclusions  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((W / 'experiments/real_reference_audit/train_reference50_verified/manifest.json').read_text())
excluded = exclusions()
want = {}
for row in manifest['frames']:
    assert row['key'] not in excluded
    scenario, agent, stem = row['key'].split('/')
    want.setdefault(row['archive'], set()).add((scenario, stem))
log = {}
for name, share, file_id, size, _ in ARCHIVES:
    if name not in want:
        continue
    url = f'https://ucla.box.com/index.php?rm=box_download_shared_file&shared_name={share}&file_id=f_{file_id}'
    with ZipFile(RangeReader(url, size)) as archive:
        agents = {}
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if len(path.parts) < 3 or path.suffix != '.yaml':
                continue
            scenario, agent, stem = path.parts[-3], path.parent.name, path.stem
            if not re.fullmatch(r'20\d{2}(?:-\d{2}){5}_\d+_\d+', scenario):
                continue
            agents.setdefault(scenario, set()).add(agent)
            if (scenario, stem) in want[name] and agent != '-1' and f'{scenario}/-1/{stem}' not in excluded:
                target = out / scenario / agent / f'{stem}.yaml'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
                log.setdefault(f'{scenario}/{stem}', []).append(agent)
        for scenario in sorted({s for s, _ in want[name]}):
            print(name, scenario, 'agents', sorted(agents.get(scenario, [])), flush=True)
(out / 'fetched.json').write_text(json.dumps(log, indent=1))
print('frames with other-agent labels', len(log), 'of', sum(len(v) for v in want.values()), '| files', sum(len(v) for v in log.values()))
