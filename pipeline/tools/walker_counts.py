import json
import sys
from pathlib import Path

nat = Path(sys.argv[1])
conv = Path(sys.argv[2])
total = sum(len(json.loads((c / 'clip.json').read_text())['walkers']) for c in nat.glob('clip_*'))
recs = json.loads((conv / 'frames.json').read_text())
seen = {(r['layout_id'], o['id']) for r in recs for o in r['object_details'] if o['class'] == 'pedestrian' and o['semantic_hits'] > 0}
hit = {(r['layout_id'], o['id']) for r in recs for o in r['object_details'] if o['class'] == 'pedestrian' and o['hits'] > 0}
ann = {(r['layout_id'], o['id']) for r in recs for o in r['object_details'] if o['class'] == 'pedestrian' and o['annotated']}
print({'walkers_spawned': total, 'walkers_with_semantic_hits_in_saved_semantic_frames': len(seen), 'walkers_with_any_hits': len(hit), 'walkers_annotated': len(ann)})
