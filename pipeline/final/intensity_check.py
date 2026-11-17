"""Compare point intensity of real dev scans (coop/dev150/scans) with synthetic training scans (twin_st, all_v1). Dev data only."""
import glob, numpy as np, os, sys
base = '/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight'
D = os.path.expanduser('~/projects/bdc2026-track-b/data')
def load(f):
    if f.endswith('.npy'):
        a = np.load(f)
    else:
        a = np.fromfile(f, np.float32).reshape(-1, 4)
    return a[:, :4]
def stats(name, files):
    rng = np.random.default_rng(0)
    fs = sorted(files); fs = [fs[i] for i in rng.choice(len(fs), min(30, len(fs)), replace=False)]
    a = np.concatenate([load(f) for f in fs]); i = a[:, 3]; r = np.linalg.norm(a[:, :2], axis=1)
    nz = r > 0.5
    q = np.percentile(i[nz], [1, 10, 50, 90, 99])
    print(f'{name:28s} files {len(files):6d} pts/scan {len(a)//len(fs):7d} | intensity q1/10/50/90/99 {np.round(q,3)} | frac==0 {np.mean(i[nz]==0):.3f} | max {i.max():.3f}')
print(os.listdir(base + '/coop/dev150/scans')[:3])
stats('real dev150', glob.glob(base + '/coop/dev150/scans/*'))
for ds in ['claude_canon_all_twin_st_v1', 'claude_canon_all_v1']:
    stats(ds, glob.glob(f'{D}/{ds}/points/*.npy'))
