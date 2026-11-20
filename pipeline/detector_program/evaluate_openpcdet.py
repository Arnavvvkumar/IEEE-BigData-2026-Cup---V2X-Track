"""Export fixed-dev predictions and use the challenge's official-format evaluator."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

p=argparse.ArgumentParser()
for name in ('repo','cfg','data','checkpoint','source','evaluator','out'):
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--batch-size',type=int,default=1)
p.add_argument('--workers',type=int,default=0)
p.add_argument('--real-kit', type=Path, help='Eligible public real development fixture with audited labels')
a=p.parse_args()
if a.out.exists(): raise FileExistsError(a.out)
sys.path.insert(0,str(a.repo))
from pcdet.config import cfg,cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network,load_data_to_gpu
from pcdet.utils import common_utils
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
cfg_from_yaml_file(str(a.cfg),cfg);cfg.DATA_CONFIG.DATA_PATH=str(a.data)
logger=common_utils.create_logger()
dataset,loader,_=build_dataloader(dataset_cfg=cfg.DATA_CONFIG,class_names=cfg.CLASS_NAMES,
    batch_size=a.batch_size,dist=False,workers=a.workers,logger=logger,training=False)
model=build_network(model_cfg=cfg.MODEL,num_class=len(cfg.CLASS_NAMES),dataset=dataset)
model.load_params_from_file(filename=str(a.checkpoint),logger=logger,to_cpu=True)
model.cuda().eval(); predictions={}

def real_batches():
    ids = json.loads((a.real_kit / 'detection_test_frame_ids.json').read_text())
    for offset in range(0, len(ids), a.batch_size):
        frames = []
        for frame in ids[offset:offset + a.batch_size]:
            xyz = np.fromfile(a.real_kit / 'detection_test_frames' / f'{frame}.bin', dtype='<f4').reshape(-1, 3)
            points = np.pad(xyz, ((0, 0), (0, 1)))
            frames.append(dataset.prepare_data({'points': points, 'frame_id': frame}))
        yield dataset.collate_batch(frames)

with torch.no_grad():
    for batch in real_batches() if a.real_kit else loader:
        ids=batch['frame_id'].tolist();load_data_to_gpu(batch); outputs,_=model(batch)
        for frame,out in zip(ids,outputs):
            assert frame not in predictions
            boxes=out['pred_boxes'].cpu().numpy(); scores=out['pred_scores'].cpu().numpy(); labels=out['pred_labels'].cpu().numpy()
            assert boxes.ndim==2 and boxes.shape[1]==7 and len(boxes)==len(scores)==len(labels)
            assert np.isfinite(boxes).all() and np.isfinite(scores).all() and (boxes[:,3:6]>0).all()
            assert ((labels>=1)&(labels<=3)).all()
            predictions[frame]=[{'box':box.tolist(),'score':float(score),'class':cfg.CLASS_NAMES[int(label)-1]} for box,score,label in zip(boxes,scores,labels)]
if a.real_kit:
    truth_path = a.real_kit / 'ground_truth.json'
    truth = json.loads(truth_path.read_text())
    assert len(truth) == 50
else:
    truth_path = a.source / 'frames.json'
    records=[r for r in json.loads(truth_path.read_text()) if r['split']=='validate']
    truth={r['frame']:r['ground_truth'] for r in records}
    assert len(truth)==500
assert set(predictions)==set(truth)
spec=importlib.util.spec_from_file_location('official_detection_eval',a.evaluator)
evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
metrics=evaluator.evaluate_3d_detection(predictions,truth,list(cfg.CLASS_NAMES),iou_thresholds=(0.5,0.7),iou_type='3d')
a.out.mkdir(parents=True)
(a.out/'predictions.json').write_text(json.dumps(predictions,allow_nan=False))
result={'scope':'public real development' if a.real_kit else 'fixed synthetic development', 'metrics':metrics,
        'sha256':{'checkpoint':sha(a.checkpoint),'configuration':sha(a.cfg),
                  'source_manifest':sha(truth_path),'evaluator':sha(a.evaluator)}}
(a.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result,indent=2))
