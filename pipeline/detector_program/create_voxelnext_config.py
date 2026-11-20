"""Adapt the official VoxelNeXt config to the three-class custom dataset."""
import argparse
from pathlib import Path
import yaml
p=argparse.ArgumentParser()
for n in ('repo','center','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
c=yaml.safe_load(a.center.read_text())
v=yaml.safe_load((a.repo/'tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml').read_text())
v['CLASS_NAMES']=c['CLASS_NAMES'];v['DATA_CONFIG']=c['DATA_CONFIG']
v['DATA_CONFIG']['DATA_PROCESSOR'][-1]['VOXEL_SIZE']=[0.2,0.2,0.2]
head=v['MODEL']['DENSE_HEAD'];head['CLASS_NAMES_EACH_HEAD']=[c['CLASS_NAMES']]
head['SEPARATE_HEAD_CFG']['HEAD_ORDER']=['center','center_z','dim','rot']
del head['SEPARATE_HEAD_CFG']['HEAD_DICT']['vel']
head['LOSS_CONFIG']['LOSS_WEIGHTS']['code_weights']=[1.,1.,1.,1.,1.,1.,0.2,0.2]
head['POST_PROCESSING']['POST_CENTER_LIMIT_RANGE']=[-40.,-40.,-8.,40.,40.,2.]
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(yaml.safe_dump(v,sort_keys=False));print(a.output)
