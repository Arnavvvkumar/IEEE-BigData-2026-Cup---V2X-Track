"""Replay final vehicle/truck fusion on screened cached development passes."""
import gc
import json
import sys
import time

from replay import EXP, P1, P2, V1, calib, fusion, metric, read, screened, spec
from audit import add_postprocess, copy_rule, prepared, score, split


def run(size, p2_weight):
    screened()
    start = time.perf_counter()
    d = EXP / 'coop' / f'dev{size}'
    gt = {f:[g for g in gs if g['points']>=5] for f,gs in read(d/'gt_coop.json').items()}
    ids = sorted(gt)
    active, counts = {}, {}
    empty = dict.fromkeys(ids, ())
    base = spec() + [(P1,'pedestrian')]*3 + [(P2,'pedestrian')]*p2_weight
    for k, classes in dict(base+[(V1,'vehicle')]).items():
        data = read(d/'allpred'/f'{k}_allpred.json')
        counts[k] = len(data)
        keep = set(classes.split(',')) - {'pedestrian'}
        if keep:
            active[k] = [{f:[x for x in pp[f] if x['class'] in keep] for f in ids} for pp in data.values()]
            assert all(set(pp)==set(ids) for pp in data.values())
        del data
        gc.collect()
    fusion.C = ['vehicle','truck']
    fusion.ONE_VOTE = {'vehicle'}
    fusion.YAW_CIRC = {'vehicle'}
    fusion.TRUCK_SCORE = ''
    for weight,radius in ((0,.6),(1,.6),(3,.6),(0,.45),(0,.75)):
        seq = base + [(V1,'vehicle')]*weight
        passes = [p for tag,_ in seq for p in active.get(tag,[empty]*counts[tag])]
        fusion.RADIUS['vehicle'] = radius/fusion.SCALE
        fuse = fusion.fuse(passes,ids)
        for a in fuse.values():
            for b in a:
                if b['class']=='vehicle':
                    b['box'][2] += calib.VEH['dz']
                    b['box'][5] += calib.VEH['dh']
        rule = calib.truck_dedupe(calib.truck_rule(fuse, **{**calib.TRUCK,'a':.5,'base':.1}))
        cars = {f:[x for x in rule[f] if x['class']=='vehicle'] for f in ids}
        trucks = add_postprocess(copy_rule(fuse))
        full = metric(trucks,gt,'truck')
        gen = prepared(trucks,gt,True)
        genu = [round(score(split(gen, subset))[0],5) for subset in (ids,ids[::2],ids[1::2])]
        print(json.dumps({'set':size,'p2':p2_weight,'v1':weight,'vehicle_radius':radius,
                          'vehicle':metric(cars,gt,'vehicle'),'truck':full,'genuine':genu,
                          'seconds':round(time.perf_counter()-start,1)}),flush=True)


if __name__ == '__main__':
    run(int(sys.argv[1]),int(sys.argv[2]))
