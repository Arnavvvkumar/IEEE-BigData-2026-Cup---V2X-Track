#!/usr/bin/env python
"""Short A100 smoke test.

Checks CUDA, the compiled OpenPCDet ops, and one training step on synthetic
twin frames already on scratch. Does not read real labels, real object points,
or the forbidden frames, and does not submit anything.
"""
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/fs/scratch/<osc_project>/bdc")
TWIN = ROOT / "data" / "claude_twin_v1"
TINY = ROOT / "data" / "smoke_tiny"
CFG = ROOT / "cfg" / "vnx010_all.yaml"


def gpu_and_ops():
    print("torch", torch.__version__, "cuda_build", torch.version.cuda, flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available in the smoke job")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("device", name, "capability", cap, flush=True)
    value = torch.randn(8, device="cuda").sum()
    print("cuda_sum", float(value), flush=True)
    import spconv
    print("spconv", getattr(spconv, "__version__", "ok"), flush=True)
    import pcdet
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    boxes = torch.tensor(
        [[0, 0, 0, 2, 2, 2, 0], [1, 0, 0, 2, 2, 2, 0.1]],
        dtype=torch.float32,
        device="cuda",
    )
    iou = iou3d_nms_utils.boxes_iou3d_gpu(boxes, boxes)
    print("pcdet", pcdet.__file__, flush=True)
    print("iou3d", iou.detach().float().cpu().tolist(), flush=True)
    print("SMOKE_GPU_OK", name, flush=True)


def complete_twin_ids(limit=4):
    points = TWIN / "points"
    labels = TWIN / "labels"
    if not points.is_dir() or not labels.is_dir():
        print("twin points/labels not on scratch yet", flush=True)
        return []
    files = sorted(points.glob("twin_*.npy"))
    # The highest name may be the file rsync is still writing.
    if len(files) > 1:
        files = files[:-1]
    found = []
    for path in files:
        label = labels / f"{path.stem}.txt"
        if not label.is_file():
            continue
        try:
            cloud = np.load(path)
        except Exception as exc:
            print("skip unreadable", path.name, exc, flush=True)
            continue
        if cloud.ndim != 2 or cloud.shape[0] < 10 or cloud.shape[1] < 4:
            print("skip bad shape", path.name, getattr(cloud, "shape", None), flush=True)
            continue
        found.append(path.stem)
        if len(found) >= limit:
            break
    return found


def write_tiny(ids):
    if TINY.exists():
        for child in TINY.rglob("*"):
            if child.is_symlink() or child.is_file():
                child.unlink()
        for child in sorted(TINY.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        TINY.rmdir()
    for sub in ("points", "labels", "ImageSets"):
        (TINY / sub).mkdir(parents=True)
    for frame_id in ids:
        (TINY / "points" / f"{frame_id}.npy").symlink_to(TWIN / "points" / f"{frame_id}.npy")
        (TINY / "labels" / f"{frame_id}.txt").symlink_to(TWIN / "labels" / f"{frame_id}.txt")
    (TINY / "ImageSets" / "train.txt").write_text("".join(f"{i}\n" for i in ids))
    (TINY / "ImageSets" / "val.txt").write_text(ids[0] + "\n")

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.custom.custom_dataset import CustomDataset

    cfg_from_yaml_file(str(CFG), cfg)
    dataset = CustomDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=TINY,
    )
    dataset.set_split("train")
    infos = dataset.get_infos(
        class_names=list(cfg.CLASS_NAMES),
        num_workers=1,
        has_label=True,
        num_features=4,
    )
    with open(TINY / "custom_infos_train.pkl", "wb") as handle:
        pickle.dump(infos, handle)
    with open(TINY / "custom_infos_val.pkl", "wb") as handle:
        pickle.dump(infos[:1], handle)
    print("tiny_frames", ids, "infos", len(infos), flush=True)


def train_one_step(n_frames):
    cmd = [
        sys.executable, "-u", "train.py",
        "--cfg_file", str(CFG),
        "--batch_size", "2" if n_frames >= 2 else "1",
        "--epochs", "1",
        "--workers", "0",
        "--extra_tag", "smoke_debug",
        "--skip_builtin_eval",
        "--logger_iter_interval", "1",
        "--wo_gpu_stat",
        "--max_ckpt_save_num", "1",
        "--set", "DATA_CONFIG.DATA_PATH", str(TINY),
    ]
    print("train_cmd", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(cmd, cwd=str(ROOT / "code" / "OpenPCDet" / "tools"), env=env)
    if completed.returncode != 0:
        print("SMOKE_TRAIN_FAIL", completed.returncode, flush=True)
        return completed.returncode
    print("SMOKE_TRAIN_OK", flush=True)
    return 0


def main():
    gpu_and_ops()
    ids = complete_twin_ids(4)
    if len(ids) < 1:
        print("SMOKE_TRAIN_SKIPPED no complete synthetic twin frame on scratch yet", flush=True)
        return 0
    if len(ids) == 1:
        ids = [ids[0], ids[0]]
        print("only one complete frame; using it twice for a single batch", flush=True)
    write_tiny(ids)
    return train_one_step(len(set(ids)))


if __name__ == "__main__":
    sys.exit(main())
