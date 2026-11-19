"""Shared pieces of the learned pedestrian box refiner.
The refiner sees the points around ONE box, expressed relative to the box centre in a frame whose first axis points away from the
sensor, and predicts how far the true box centre is from the given one (radial, tangential, vertical) and the height error.
It is trained on synthetic site-twin frames only (synthetic boxes, deliberately jittered); real scans are only used at inference,
after levelling with the same fixed ground plane the twin generator uses (canonical frame, ground z = -3.0)."""
import os
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ['BDC_ROOT'] + '/experiments/claude_carla_overnight')
sys.path.insert(0, str(W / 'erroranalysis'))
from groundplane import level_transform  # noqa: E402

GZ, K, RADIUS = -3.0, 48, 0.9
R, T = level_transform(np.load(W / 'twin/median_plane.npy'), GZ)   # sensor frame -> canonical frame: p @ R.T + T
SENSOR_XY = T[:2]                                                    # the sensor origin in the canonical frame


def crop(p, box, rng):
    """p: (N,3) canonical-frame points near the box; box: x y z l w h yaw (canonical). Returns (K,3) points, (3,) aux, u, n or None."""
    c, h = box[:3], box[5]
    d = p - c
    m = (np.hypot(d[:, 0], d[:, 1]) < RADIUS) & (d[:, 2] > -h / 2 + 0.10) & (d[:, 2] < h / 2 + 0.25)
    n = int(m.sum())
    if n < 4:
        return None
    q = d[m]
    u = c[:2] - SENSOR_XY
    rg = float(np.hypot(*u))
    u = u / max(rg, 1e-6)
    x = np.stack([q[:, 0] * u[0] + q[:, 1] * u[1], -q[:, 0] * u[1] + q[:, 1] * u[0], q[:, 2]], 1)
    idx = rng.choice(n, K, replace=n < K)
    return x[idx].astype(np.float32), np.array([rg / 40.0, min(n, 200) / 100.0, h], np.float32), u, n


def build_net():
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.pt = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(256 + 3, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))

        def forward(self, x, aux):
            import torch
            f = self.pt(x).max(1)[0]
            return self.head(torch.cat([f, aux], 1))
    return Net()
