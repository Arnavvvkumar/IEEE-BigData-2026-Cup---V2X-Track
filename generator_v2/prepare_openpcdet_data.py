"""Prepare immutable custom dataset for OpenPCDet, preserving original IDs and labels."""
import json,hashlib,argparse,os
from pathlib import Path
import numpy as np
SOURCE=Path(os.environ['BDC_DATA'])/'claude_v2_scale5500_fix1'
CLASSES={'vehicle','pedestrian','truck'}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def build(out,limit=None):
 if out.exists(): raise FileExistsError(out)
 records=json.loads((SOURCE/'frames.json').read_text()); train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='validate']
 assert len(train)==5000 and len(val)==500 and not ({r['frame'] for r in train}&{r['frame'] for r in val})
 assert not ({r['layout_fingerprint'] for r in train}&{r['layout_fingerprint'] for r in val})
 out.mkdir(parents=True);[(out/x).mkdir() for x in ('points','labels','ImageSets')]; paths={}
 for p in SOURCE.glob('*/*/*/*.bin'):paths.setdefault(p.stem,[]).append(p)
 chosen={}; manifest=[]
 for split,rows in [('train',train),('val',val)]:
  if limit:
   if split=='train':
    assert limit>=2
    empty=next(r for r in rows if not r['ground_truth'])
    rows=[r for r in rows if r['frame']!=empty['frame']][:limit-1]+[empty]
   else: rows=rows[:limit]
  chosen[split]=[]
  for r in rows:
   name=r['frame'];src=paths.get(name,[]); assert len(src)==1,(name,src)
   pts=np.fromfile(src[0],dtype='<f4');assert pts.size%4==0;pts=pts.reshape(-1,4);assert np.isfinite(pts).all()
   boxes=r['ground_truth']; lines=[]
   for b in boxes:
    a=np.asarray(b['box'],dtype=np.float64); assert a.shape==(7,) and np.isfinite(a).all() and (a[3:6]>0).all() and b['class'] in CLASSES
    lines.append(' '.join(format(float(x),'.17g') for x in a)+' '+b['class']+'\n')
   np.save(out/'points'/f'{name}.npy',pts); (out/'labels'/f'{name}.txt').write_text(''.join(lines));chosen[split].append(name)
   manifest.append({'id':name,'split':split,'source_sha256':digest(src[0]),'num_points':len(pts),'num_boxes':len(boxes)})
  (out/'ImageSets'/f'{split}.txt').write_text(''.join(x+'\n' for x in chosen[split]))
 meta={'source_manifest_sha256':digest(SOURCE/'frames.json'),'scope':'preflight' if limit else 'full','frames':manifest,'empty_labels':sum(m['num_boxes']==0 for m in manifest)}
 (out/'provenance.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'train':len(chosen['train']),'val':len(chosen['val']),'empty_labels':meta['empty_labels']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args();build(a.out,a.limit)
