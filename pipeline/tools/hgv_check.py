"""Where do european_hgv own returns fall relative to its CARLA bounding box? (box-frame extents)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_carla as A  # noqa: E402
import convert_carla as C  # noqa: E402

nat = Path(sys.argv[1])
bp_name = sys.argv[2] if len(sys.argv) > 2 else 'vehicle.carlamotors.european_hgv'
shown = 0
for cdir in sorted(nat.glob('clip_*')):
    for sp in sorted((cdir / 'semantic').glob('*.npy')):
        rec = json.loads((cdir / 'actors' / sp.name.replace('.npy', '.json')).read_text())
        sem = np.load(sp)
        for a in rec['actors']:
            if a['blueprint'] != bp_name:
                continue
            m = sem['obj'] == a['id']
            if m.sum() < 100:
                continue
            box = C.box_from_record(a, rec)
            off = A.local_offsets(np.c_[sem['x'][m], -sem['y'][m], sem['z'][m]].astype(float), box)
            print(cdir.name, a['id'], 'n', int(m.sum()), 'box lwh', np.round(box[3:6], 2), 'extent local x', np.round([off[:, 0].min(), off[:, 0].max()], 2),
                  'y', np.round([off[:, 1].min(), off[:, 1].max()], 2), 'z', np.round([off[:, 2].min(), off[:, 2].max()], 2),
                  'bbox loc', [round(v, 2) for v in a['center_world_carla']])
            shown += 1
            if shown >= 6:
                sys.exit()
