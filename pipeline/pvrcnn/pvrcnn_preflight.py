"""CPU-only preflight for the scratch batch-1 PV-RCNN config (GPUs hidden; no training, no CUDA kernels).

Checks: config parses with OpenPCDet; grid divisibility and the HeightCompression channel count (256); the CustomDataset
training loader (augmentation + voxelisation + collate, batch 1) on sampled Generator V2 and CARLA frames, with no voxel
truncation; model construction with random init (parameter count; no checkpoint loaded); anchor target assignment on CPU
(per-class positives, and the share of GT boxes whose best anchor BEV IoU is below the unmatched threshold).
spconv in env310 needs a CUDA stream, so the first forward pass is deferred to the GPU smoke command in the handoff.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
REPO = Path.home() / 'projects/detector-program/OpenPCDet'
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network  # noqa: E402
from pcdet.utils import box_utils  # noqa: E402


def out_size(n, k, s, p):
    return (n + 2 * p - k) // s + 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--cfg', type=Path, required=True)
    ap.add_argument('--data-path')
    ap.add_argument('--samples', type=int, default=40)
    ap.add_argument('--carla-prefix', default='carla_')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    assert not torch.cuda.is_available()
    a.cfg, a.out = a.cfg.resolve(), a.out.resolve()
    os.chdir(REPO / 'tools')
    cfg_from_yaml_file(str(a.cfg), cfg)
    if a.data_path:
        cfg.DATA_CONFIG.DATA_PATH = a.data_path
    log = logging.getLogger('pf')
    log.addHandler(logging.StreamHandler())
    res, checks = {'cfg': str(a.cfg), 'data_path': cfg.DATA_CONFIG.DATA_PATH}, {}
    ds, loader, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0, logger=log, training=True)
    grid = ds.grid_size.astype(int)
    res['grid_size_xyz'] = grid.tolist()
    checks['grid_divisible_by_8'] = bool(grid[0] % 8 == 0 and grid[1] % 8 == 0)
    z = grid[2] + 1
    # VoxelBackBone8x z path: conv2 s2 p1, conv3 s2 p1, conv4 s2 p(0,1,1), conv_out k(3,1,1) s(2,1,1) p0
    z2 = out_size(z, 3, 2, 1)
    z3 = out_size(z2, 3, 2, 1)
    z4 = out_size(z3, 3, 2, 0)
    zo = out_size(z4, 3, 2, 0)
    res['backbone_z_path'] = [int(z), int(z2), int(z3), int(z4), int(zo)]
    checks['bev_features_256'] = bool(128 * zo == cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES == 256)
    # OpenPCDet's AnchorGenerator calls .cuda() at construction; keep the tensors on CPU in this process only.
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    res['parameters'] = int(sum(p.numel() for p in model.parameters()))
    res['random_init_no_checkpoint'] = True
    head = model.dense_head
    ids = [info['point_cloud']['lidar_idx'] for info in ds.kitti_infos] if hasattr(ds, 'kitti_infos') else \
        [info['point_cloud']['lidar_idx'] for info in ds.custom_infos]
    rng = np.random.default_rng(0)
    carla = [i for i, f in enumerate(ids) if f.startswith(a.carla_prefix)]
    base = [i for i, f in enumerate(ids) if not f.startswith(a.carla_prefix)]
    pick = list(rng.choice(base, min(a.samples, len(base)), replace=False)) + \
        (list(rng.choice(carla, min(a.samples, len(carla)), replace=False)) if carla else [])
    cap = cfg.DATA_CONFIG.DATA_PROCESSOR[-1].MAX_NUMBER_OF_VOXELS['train']
    thr = {an['class_name']: an['unmatched_threshold'] for an in cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG}
    stats = {'voxels_max': 0, 'truncated_frames': 0, 'frames': 0, 'carla_frames': 0, 'gt': {}, 'gt_weak_anchor': {}, 'pos_anchors': {}}
    anchors = [x.view(-1, x.shape[-1]) for x in head.anchors]
    for i in pick:
        item = ds[int(i)]
        batch = ds.collate_batch([item])
        nv = int(batch['voxels'].shape[0])
        stats['voxels_max'] = max(stats['voxels_max'], nv)
        stats['truncated_frames'] += int(nv >= cap)
        stats['frames'] += 1
        stats['carla_frames'] += int(ids[int(i)].startswith(a.carla_prefix))
        assert np.isfinite(batch['voxels']).all() and np.isfinite(batch['gt_boxes']).all()
        gt = torch.from_numpy(batch['gt_boxes']).float()
        tgt = head.assign_targets(gt_boxes=gt)
        lab = tgt['box_cls_labels'][0]
        for ci, c in enumerate(cfg.CLASS_NAMES, 1):
            stats['pos_anchors'][c] = stats['pos_anchors'].get(c, 0) + int((lab == ci).sum())
            g = gt[0][gt[0][:, 7] == ci][:, :7]
            stats['gt'][c] = stats['gt'].get(c, 0) + len(g)
            if len(g):
                iou = box_utils.boxes3d_nearest_bev_iou(anchors[ci - 1][:, :7], g)
                weak = int((iou.max(0)[0] < thr[c]).sum())
                stats['gt_weak_anchor'][c] = stats['gt_weak_anchor'].get(c, 0) + weak
    res['loader'] = stats
    checks['loader_ran_all_samples'] = stats['frames'] == len(pick)
    checks['no_voxel_truncation'] = stats['truncated_frames'] == 0
    checks['every_class_gets_positive_anchors'] = all(stats['pos_anchors'].get(c, 0) > 0 for c in cfg.CLASS_NAMES)
    res['checks'] = checks
    res['all_passed'] = all(checks.values())
    res['deferred_to_gpu'] = 'spconv/pointnet2 forward + loss + backward (env310 spconv requires CUDA)'
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
