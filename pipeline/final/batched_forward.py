"""Batch OpenPCDet inference and voxelize on several threads.

Each thread gets its own voxel generator. The generator is the only shared mutable
object in test-time prepare_data (shuffle is off), and Point2VoxelCPU3d is not
safe to share across threads.
"""
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from pcdet.datasets.processor.data_processor import VoxelGeneratorWrapper
from pcdet.models import load_data_to_gpu


class _LocalGenerators:
    def __init__(self, factory):
        self.factory = factory
        self.local = threading.local()

    def generate(self, points):
        gen = getattr(self.local, 'gen', None)
        if gen is None:
            gen = self.factory()
            self.local.gen = gen
        return gen.generate(points)


_FIT_CLOUDS = None
_FIT_FN = None
_FIT_IDS = None


def _fit_index(i):
    return _FIT_FN(_FIT_CLOUDS[_FIT_IDS[i]], np.random.default_rng(i))


def fit_planes(fit_ground, clouds, ids, workers):
    """Same per-frame seeds as the serial loop, so the median plane does not change."""
    global _FIT_CLOUDS, _FIT_FN, _FIT_IDS
    _FIT_CLOUDS = clouds
    _FIT_FN = fit_ground
    _FIT_IDS = list(ids)
    n = max(1, min(int(workers), len(_FIT_IDS)))
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=n) as pool:
        return np.array(pool.map(_fit_index, range(len(_FIT_IDS)), chunksize=4))


def install_thread_voxels(dataset):
    proc = dataset.data_processor
    voxel_fn = None
    for fn in proc.data_processor_queue:
        func = getattr(fn, 'func', None)
        if func is not None and func.__name__ == 'transform_points_to_voxels':
            voxel_fn = fn
            break
    if voxel_fn is None or isinstance(getattr(proc, 'voxel_generator', None), _LocalGenerators):
        return
    config = voxel_fn.keywords['config']

    def factory():
        return VoxelGeneratorWrapper(
            vsize_xyz=config.VOXEL_SIZE,
            coors_range_xyz=proc.point_cloud_range,
            num_point_features=proc.num_point_features,
            max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
            max_num_voxels=config.MAX_NUMBER_OF_VOXELS[proc.mode],
        )

    proc.voxel_generator = _LocalGenerators(factory)


def _prepare_one(dataset, frame_id, pts):
    points = np.pad(np.ascontiguousarray(pts, dtype=np.float32), ((0, 0), (0, 1)))
    return dataset.prepare_data({'points': points, 'frame_id': frame_id})


def forward_collated(dataset, model, prepared, batch):
    try:
        load_data_to_gpu(batch)
        pred_dicts = model(batch)[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print('cuda oom, splitting batch', len(prepared), flush=True)
        return forward_prepared(dataset, model, prepared)
    if len(pred_dicts) != len(prepared):
        raise RuntimeError('prediction count %s != batch %s' % (len(pred_dicts), len(prepared)))
    return pred_dicts


def _prepare_and_collate(dataset, pool, chunk):
    futs = [pool.submit(_prepare_one, dataset, fid, pts) for fid, pts in chunk]
    prepared = [f.result() for f in futs]
    return prepared, dataset.collate_batch(prepared)


def forward_prepared(dataset, model, prepared):
    try:
        batch = dataset.collate_batch(prepared)
        load_data_to_gpu(batch)
        pred_dicts = model(batch)[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prepared) == 1:
            raise
        mid = len(prepared) // 2
        print('cuda oom, splitting batch', len(prepared), flush=True)
        return forward_prepared(dataset, model, prepared[:mid]) + forward_prepared(dataset, model, prepared[mid:])
    if len(pred_dicts) != len(prepared):
        raise RuntimeError('prediction count %s != batch %s' % (len(pred_dicts), len(prepared)))
    return pred_dicts


def iter_chunks(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def run_frames(dataset, model, items, batch_size, workers, on_batch):
    """items: list of (frame_id, Nx3 float points). on_batch(ids, pred_dicts) is called in order.

    If a batch does not fit beside the training run, the chunk is retried smaller.
    """
    install_thread_voxels(dataset)
    if not items:
        return batch_size
    workers = max(1, workers)
    bs = max(1, batch_size)
    chunks = list(iter_chunks(items, bs))
    pool = ThreadPoolExecutor(max_workers=workers)
    side = ThreadPoolExecutor(max_workers=1)
    try:
        pending = side.submit(_prepare_and_collate, dataset, pool, chunks[0])
        for index, chunk in enumerate(chunks):
            nxt = None
            if index + 1 < len(chunks):
                nxt = side.submit(_prepare_and_collate, dataset, pool, chunks[index + 1])
            prepared, batch = pending.result()
            ids = [fid for fid, _pts in chunk]
            on_batch(ids, forward_collated(dataset, model, prepared, batch))
            pending = nxt
    finally:
        side.shutdown(wait=True)
        pool.shutdown(wait=True)
    return bs
