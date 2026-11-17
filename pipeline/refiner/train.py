"""Train the pedestrian box refiner on the synthetic samples (make_data.py). 5% of samples are held out to report the error.
usage: train.py <data npz> <out .pt> [epochs=30]"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

d = np.load(sys.argv[1])
X, A, Y = d['X'].astype(np.float32), d['A'], d['Y']
ep = int(sys.argv[3]) if len(sys.argv) > 3 else 30
rng = np.random.default_rng(0)
perm = rng.permutation(len(X))
nv = len(X) // 20
va, tr = perm[:nv], perm[nv:]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net = C.build_net().to(dev)
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
B = 1024
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=ep * (len(tr) // B + 1))
Xt, At, Yt = (torch.from_numpy(a) for a in (X, A, Y))
print('samples', len(X), 'train', len(tr), 'val', nv, 'device', dev, flush=True)
print('val error if the box is left alone (m): radial %.3f tangential %.3f z %.3f h %.3f' % tuple(np.abs(Y[va]).mean(0)), flush=True)
for e in range(ep):
    net.train()
    p = rng.permutation(tr)
    for i in range(0, len(p), B):
        j = torch.from_numpy(p[i:i + B])
        x = Xt[j].to(dev)
        # augmentation: per-point noise, random point dropout by repeating a kept point, mirror the tangential axis
        x = x + 0.01 * torch.randn_like(x)
        flip = (torch.rand(len(j), 1, device=dev) < 0.5).float() * 2 - 1
        x[:, :, 1] = x[:, :, 1] * flip
        y = Yt[j].to(dev).clone()
        y[:, 1] = y[:, 1] * flip[:, 0]
        loss = torch.nn.functional.smooth_l1_loss(net(x, At[j].to(dev)), y, beta=0.05)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if e % 5 == 4 or e == ep - 1:
        net.eval()
        with torch.no_grad():
            out = torch.cat([net(Xt[va[i:i + 4096]].to(dev), At[va[i:i + 4096]].to(dev)).cpu() for i in range(0, nv, 4096)]).numpy()
        print('epoch %d val error after refining (m): radial %.3f tangential %.3f z %.3f h %.3f' % ((e + 1,) + tuple(np.abs(out - Y[va]).mean(0))), flush=True)
torch.save(net.state_dict(), sys.argv[2])
print('REFINER_TRAINED', sys.argv[2])
