"""Independent audit of the eligible real validation inputs, and a scenario-disjoint confirmation split proposal.

Inputs (read-only): starting_kit/track_b/forbidden_frames.txt, the organizer mapping CSV, the 50-frame verified public
TRAINING reference (train_reference50_verified/manifest.json + files) and its staged detector kit (real_dev_cw1).
No network access, no new real data, nothing written outside this directory. Real frames are evaluation-only.
"""
import argparse
import csv
import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'V2X-Real'))
from opencood.data_utils import SUPER_CLASS_MAP  # noqa: E402
from opencood.utils.transformation_utils import x1_to_x2  # noqa: E402

REF = ROOT / 'experiments/real_reference_audit/train_reference50_verified'
KIT = ROOT / 'experiments/real_reference_audit/real_dev_cw1'
PCR = np.array([-40, -40, -8, 40, 40, 2], dtype=float)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def forbidden():
    ids = [ln.strip() for ln in (ROOT / 'starting_kit/track_b/forbidden_frames.txt').read_text().splitlines()
           if re.fullmatch(r'\d+_\d{6}', ln.strip())]
    with (ROOT / 'experiments/real_reference_audit/v2x_real_forbidden_frames_mapping.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(ids) == len(set(ids)) == 100 and {r['forbidden_id'] for r in rows} == set(ids) and len(rows) == 100
    keys, by_scen = set(), defaultdict(list)
    for r in rows:
        idx, fr = r['forbidden_id'].split('_')
        assert r['scenario_index'] == idx and r['frame'] == fr
        key = f"{r['scenario_folder']}/-1/{fr}"
        assert r['point_cloud_file'] == key + '.bin' and r['label_file'] == key + '.yaml'
        keys.add(key)
        by_scen[r['scenario_folder']].append(int(fr))
    assert len(keys) == 100
    return keys, by_scen


def truth(label):
    cls = {s: g for g, names in SUPER_CLASS_MAP.items() for s in names}
    out, raw = [], Counter()
    for obj in label['vehicles'].values():
        raw[obj['obj_type']] += 1
        c = cls.get(obj['obj_type'])
        if c is None:
            continue
        t = x1_to_x2([*(np.asarray(obj['location']) + obj['center']), *obj['angle']], label['lidar_pose'])
        ctr = t[:3, 3]
        if (ctr < PCR[:3]).any() or (ctr > PCR[3:]).any():
            continue
        out.append({'class': c, 'subtype': obj['obj_type'], 'range': float(np.hypot(*ctr[:2]))})
    return out, raw


def session(scenario):
    return scenario.split('_')[0]  # recording timestamp; _a_b suffixes are parts of the same drive


def propose_split(frames, seed=2027, trials=50000):
    sessions = sorted({f['session'] for f in frames})
    per = {s: Counter() for s in sessions}
    for f in frames:
        per[f['session']]['frames'] += 1
        for k in ('vehicle', 'pedestrian', 'truck'):
            per[f['session']][k] += f['boxes'][k]
        per[f['session']]['archive:' + f['archive']] += 1
    tot = sum(per.values(), Counter())
    keys = ['frames', 'vehicle', 'pedestrian', 'truck'] + [k for k in tot if k.startswith('archive:')]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        mask = rng.random(len(sessions)) < 0.5
        a = sum((per[s] for s, m in zip(sessions, mask) if m), Counter())
        score = sum(abs(a[k] / max(tot[k], 1) - 0.5) for k in keys)
        if best is None or score < best[0]:
            best = (score, mask.copy(), a)
    score, mask, a = best
    A = {s for s, m in zip(sessions, mask) if m}
    return {'seed': seed, 'trials': trials, 'imbalance_score': round(float(score), 4),
            'selection_sessions': sorted(A), 'confirmation_sessions': sorted(set(sessions) - A),
            'selection_totals': {k: a[k] for k in keys}, 'confirmation_totals': {k: tot[k] - a[k] for k in keys}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    keys, fscen = forbidden()
    man = json.loads((REF / 'manifest.json').read_text())
    idx = json.loads((KIT / 'source_index.json').read_text())
    kit_ids = json.loads((KIT / 'detection_test_frame_ids.json').read_text())
    checks, frames, subtypes = {}, [], Counter()
    mkeys = [r['key'] for r in man['frames']]
    checks['manifest_50_unique'] = len(mkeys) == len(set(mkeys)) == 50
    checks['no_forbidden_path'] = not set(mkeys) & keys
    official_ids = {ln.strip() for ln in (ROOT / 'starting_kit/track_b/forbidden_frames.txt').read_text().splitlines() if ln.strip()}
    checks['kit_ids_are_local_aliases_not_official_ids'] = not (set(json.loads((KIT / 'detection_test_frame_ids.json').read_text())) & official_ids)
    checks['source_is_public_training_split'] = man['source'] == 'UCLA V2X-Real-LiDAR-128 public training split'
    hash_bad, stage_bad = [], []
    for r in man['frames']:
        for suf, info in r['files'].items():
            p = REF / info['local']
            if sha(p) != info['sha256'] or info['source'] != r['key'] + suf:
                hash_bad.append(r['key'] + suf)
    checks['reference_files_match_manifest_sha256'] = not hash_bad
    by_key = {r['key']: r for r in man['frames']}
    checks['kit_ids_match_index'] = kit_ids == list(idx) and len(kit_ids) == 50
    for name, row in idx.items():
        r = by_key[row['source_key']]
        src = np.fromfile(REF / r['files']['.bin']['local'], dtype='<f4').reshape(-1, 4)
        xyz = src[np.isfinite(src[:, :3]).all(1), :3]
        staged = np.fromfile(KIT / 'detection_test_frames' / f'{name}.bin', dtype='<f4')
        if staged.size != xyz.size or not np.array_equal(staged.reshape(-1, 3), xyz) or row['raw_xyz_count'] != len(xyz):
            stage_bad.append(name)
        label = yaml.safe_load((REF / r['files']['.yaml']['local']).read_text())
        gt, raw = truth(label)
        subtypes.update(raw)
        scen, _, fr = row['source_key'].split('/')
        fr = int(fr)
        dist = min((abs(fr - x) for x in fscen.get(scen, [])), default=None)
        frames.append({'kit_id': name, 'key': row['source_key'], 'scenario': scen, 'session': session(scen), 'archive': r['archive'],
                       'infra': label.get('infra'), 'lidar_pose': label['lidar_pose'], 'points': int(len(xyz)),
                       'boxes': {c: sum(g['class'] == c for g in gt) for c in ('vehicle', 'pedestrian', 'truck')},
                       'subtypes_in_range': dict(Counter(g['subtype'] for g in gt)),
                       'frames_to_nearest_forbidden_same_scenario': dist})
    checks['staged_xyz_equals_finite_source_xyz'] = not stage_bad
    checks['all_infrastructure_labels'] = all(f['infra'] is True for f in frames)
    poses = np.array([f['lidar_pose'] for f in frames])
    spread = (poses.max(0) - poses.min(0)).round(3).tolist()
    cfg = json.loads((KIT / 'config.json').read_text())
    cfg_ids = set(cfg.get('detection_test_frame_ids', []))
    notes = {
        'kit_config_json_lists_official_ids': {'count': len(cfg_ids), 'are_forbidden_ids': bool(cfg_ids) and cfg_ids <= {
            ln.strip() for ln in (ROOT / 'starting_kit/track_b/forbidden_frames.txt').read_text().splitlines()},
            'used_by_scorer': 'no: score_detector_kit.py reads source_index.json; config.json is the dummy kit config copied for exporter shape'},
        'scenarios': len({f['scenario'] for f in frames}), 'sessions': len({f['session'] for f in frames}),
        'frames_per_scenario': dict(Counter(Counter(f['scenario'] for f in frames).values())),
        'sensor_pose_spread_xyz_roll_yaw_pitch': spread,
        'scenarios_shared_with_forbidden_frames': sum(f['scenario'] in fscen for f in frames),
        'nearest_forbidden_gap_frames': {'min': min(d for d in (f['frames_to_nearest_forbidden_same_scenario'] for f in frames) if d is not None),
                                         'median': float(np.median([f['frames_to_nearest_forbidden_same_scenario'] for f in frames
                                                                    if f['frames_to_nearest_forbidden_same_scenario'] is not None])),
                                         'within_10_frames_1s': sum(1 for f in frames if (f['frames_to_nearest_forbidden_same_scenario'] or 999) <= 10)},
        'box_totals_in_range': {c: sum(f['boxes'][c] for f in frames) for c in ('vehicle', 'pedestrian', 'truck')},
        'frames_with_class': {c: sum(f['boxes'][c] > 0 for f in frames) for c in ('vehicle', 'pedestrian', 'truck')},
        'raw_subtypes_all_ranges': dict(subtypes),
        'truck_superclass_in_range': dict(sum((Counter({k: v for k, v in f['subtypes_in_range'].items()
                                                        if k in SUPER_CLASS_MAP.get('truck', [])}) for f in frames), Counter())),
        'pedestrian_superclass_in_range': dict(sum((Counter({k: v for k, v in f['subtypes_in_range'].items()
                                                             if k in SUPER_CLASS_MAP.get('pedestrian', [])}) for f in frames), Counter())),
        'publisher_validation_zip': 'v2xreal_lidar128_val.zip: directory audited by Codex (717 agent -1 pairs, 0 forbidden); contents '
                                    'not opened because the organizer notice names public TRAINING frames. Not used here.'}
    split = propose_split(frames)
    A = set(split['selection_sessions'])
    split['selection_ids'] = [f['kit_id'] for f in frames if f['session'] in A]
    split['confirmation_ids'] = [f['kit_id'] for f in frames if f['session'] not in A]
    checks['split_session_disjoint'] = not ({f['session'] for f in frames if f['kit_id'] in split['selection_ids']} &
                                            {f['session'] for f in frames if f['kit_id'] in split['confirmation_ids']})
    checks['split_scenario_disjoint'] = not ({f['scenario'] for f in frames if f['kit_id'] in split['selection_ids']} &
                                             {f['scenario'] for f in frames if f['kit_id'] in split['confirmation_ids']})
    out = {'checks': checks, 'all_passed': all(checks.values()), 'notes': notes, 'split_proposal': split, 'frames': frames,
           'script_sha256': sha(__file__), 'reference_manifest_sha256': sha(REF / 'manifest.json'),
           'kit_source_index_sha256': sha(KIT / 'source_index.json'),
           'problems': {'hash': hash_bad, 'staging': stage_bad}}
    a.out.write_text(json.dumps(out, indent=1))
    (a.out.parent / 'real_split_proposal_v1.json').write_text(json.dumps({
        'status': 'PROPOSAL (not yet used for any decision)', 'unit': 'recording session (scenario timestamp prefix)',
        'selection_ids': split['selection_ids'], 'confirmation_ids': split['confirmation_ids'],
        'selection_sessions': split['selection_sessions'], 'confirmation_sessions': split['confirmation_sessions'],
        'source_keys': {f['kit_id']: f['key'] for f in frames}}, indent=1))
    print(json.dumps({'checks': checks, 'notes': {k: notes[k] for k in ('scenarios', 'sessions', 'scenarios_shared_with_forbidden_frames',
                                                                        'nearest_forbidden_gap_frames', 'box_totals_in_range',
                                                                        'frames_with_class', 'truck_superclass_in_range',
                                                                        'pedestrian_superclass_in_range', 'sensor_pose_spread_xyz_roll_yaw_pitch',
                                                                        'kit_config_json_lists_official_ids')},
                      'split': {k: split[k] for k in ('imbalance_score', 'selection_totals', 'confirmation_totals')},
                      'n_sel': len(split['selection_ids']), 'n_conf': len(split['confirmation_ids'])}, indent=1))


if __name__ == '__main__':
    main()
