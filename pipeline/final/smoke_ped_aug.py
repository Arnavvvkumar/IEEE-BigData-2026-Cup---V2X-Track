"""Smoke test for canon/pvrcnn_ped_aug.yaml: 40 augmented training samples + 3 GPU train steps. Run from OpenPCDet/tools."""
import sys, time, numpy as np, torch
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils

yaml, data = sys.argv[1], sys.argv[2]
cfg_from_yaml_file(yaml, cfg)
cfg.DATA_CONFIG.DATA_PATH = data
log = common_utils.create_logger()
ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=2, dist=False, workers=4, logger=log, training=True)
print('aug list', [a.NAME for a in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST])
t = time.time(); npts, nbox = [], []
for i in range(40):
    d = ds[i * 37 % len(ds)]
    npts.append(len(d['points'])); nbox.append(len(d['gt_boxes']))
    assert np.isfinite(d['points']).all() and np.isfinite(d['gt_boxes']).all()
print(f'40 samples ok in {time.time()-t:.1f}s | points mean {np.mean(npts):.0f} | boxes mean {np.mean(nbox):.1f}')
model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), ds).cuda().train()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
fn = model_fn_decorator()
it = iter(loader)
for k in range(3):
    b = next(it)
    loss, tb, disp = fn(model, b)
    opt.zero_grad(); loss.backward(); opt.step()
    print('step', k, 'loss', float(loss))
print('SMOKE_OK')
