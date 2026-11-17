#!/bin/bash
# Send only the files still missing on OSC, as one gzip, over a single ssh append.
# The per-file rsync was receiver-window limited: the Ascend login node (load ~30,
# GPFS busy) was not reading the socket. A single sequential write avoids that.
# Already-uploaded files are left in place. Synthetic data only.
set -euo pipefail
cd "$(dirname "$0")"
D="$HOME/projects/bdc2026-track-b/data"
R=/fs/scratch/<osc_project>/bdc/data
STAGE="$HOME/projects/bdc2026-track-b/upload_stage"
LOG="$PWD/upload_fast.log"
mkdir -p "$STAGE"
exec > >(tee -a "$LOG") 2>&1
echo "FAST_START $(date -Is)"

echo "stopping per-file uploader"
# Parent first, so upload_rsync.sh cannot start another attempt.
pgrep -af 'osc/upload_rsync.sh|rsync -aL --partial --inplace -z' || true
pkill -f 'osc/upload_rsync.sh' || true
sleep 1
pkill -f 'rsync -aL --partial --inplace -z' || true
pkill -f 'rsync --server -logDtprze' || true
sleep 2
if pgrep -f 'rsync -aL --partial --inplace -z' >/dev/null; then
  pkill -9 -f 'rsync -aL --partial --inplace -z' || true
fi
echo "per-file uploader stopped $(date -Is)"

echo "=== uplink probe (8 MB to Cloudflare, rsync is stopped) ==="
head -c 8000000 /dev/zero > /tmp/up8.bin
s=$(date +%s.%N)
if curl -s -o /dev/null -X POST --data-binary @/tmp/up8.bin --max-time 90 https://speed.cloudflare.com/__up; then
  e=$(date +%s.%N)
  python3 -c "print('cloudflare_MBps', round(8 / ($e - $s), 3))"
else
  echo "cloudflare probe failed"
fi
rm -f /tmp/up8.bin

SSH="ssh -i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -c aes128-gcm@openssh.com"
$SSH "mkdir -p '$R/incoming'" 2>/dev/null || true

remote_size() {
  local n
  n=$($SSH "stat -c %s '$1' 2>/dev/null || echo 0" 2>/dev/null | grep -E '^[0-9]+$' | tail -n 1 || true)
  echo "${n:-0}"
}

for ds in claude_twin_v1 claude_canon_all_v1; do
  echo "=== manifest $ds $(date -Is) ==="
  raw="$STAGE/$ds.remote.raw"
  if ! $SSH "mkdir -p '$R/$ds' && cd '$R/$ds' && echo COUNT \$(find . -type f -printf '%s\t%P\n' | tee /tmp/bdc_manifest_$ds | wc -l) && cat /tmp/bdc_manifest_$ds" > "$raw" 2>/dev/null; then
    echo "REMOTE_LIST_FAILED $ds"
    exit 1
  fi
  python3 - "$D/$ds" "$raw" "$STAGE/$ds.list" << 'PY'
import os, sys
root, raw_path, list_path = sys.argv[1:]
lines = open(raw_path, encoding="utf-8", errors="replace").read().splitlines()
counts = [ln.split()[1] for ln in lines if ln.startswith("COUNT ")]
if len(counts) != 1:
    raise SystemExit("manifest count missing: %s" % counts)
remote = {}
for ln in lines:
    if "\t" not in ln or not ln[0].isdigit():
        continue
    size, path = ln.split("\t", 1)
    remote[path] = int(size)
if len(remote) != int(counts[0]):
    raise SystemExit("manifest rows %d != count %s" % (len(remote), counts[0]))
missing = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    for name in filenames:
        full = os.path.join(dirpath, name)
        if os.path.islink(full):
            full_stat = os.stat(full)
            size = full_stat.st_size
        else:
            size = os.path.getsize(full)
        rel = os.path.relpath(full, root).replace("\\", "/")
        if remote.get(rel) != size:
            missing.append(rel)
open(list_path, "w").write("".join(p + "\n" for p in missing))
print("remote_files", len(remote), "missing", len(missing))
PY
  nmiss=$(wc -l < "$STAGE/$ds.list")
  if [ "$nmiss" = "0" ]; then
    echo "NOTHING_MISSING $ds"
    continue
  fi
  echo "packing $nmiss files $(date -Is)"
  tar -C "$D/$ds" -h -cf - -T "$STAGE/$ds.list" | gzip -1 > "$STAGE/$ds.tar.gz.tmp"
  mv "$STAGE/$ds.tar.gz.tmp" "$STAGE/$ds.tar.gz"
  ls -lh "$STAGE/$ds.tar.gz"
  dest="$R/incoming/$ds.tar.gz"
  for try in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    rsize=$(remote_size "$dest")
    lsize=$(stat -c %s "$STAGE/$ds.tar.gz")
    rsize=${rsize:-0}
    echo "STREAM $ds try $try remote $rsize local $lsize $(date -Is)"
    if [ "$rsize" = "$lsize" ]; then
      echo "SIZE_MATCH $ds"
      break
    fi
    if [ "$rsize" -gt "$lsize" ]; then
      echo "REMOTE_LARGER $ds"
      exit 1
    fi
    if [ "$try" = "1" ]; then
      ( sleep 25; echo "=== tcp after 25s ==="; ss -tin dst ascend.osc.edu | head -n 12 ) &
    fi
    set +e
    dd if="$STAGE/$ds.tar.gz" bs=4M iflag=skip_bytes,fullblock skip="$rsize" status=progress 2>&1 \
      | $SSH "cat >> '$dest'"
    echo "stream_exit $? $(date -Is)"
    set -e
    sleep 5
  done
  rsize=$(remote_size "$dest")
  lsize=$(stat -c %s "$STAGE/$ds.tar.gz")
  if [ "$rsize" != "$lsize" ]; then
    echo "STREAM_INCOMPLETE $ds remote $rsize local $lsize"
    exit 1
  fi
done

echo "=== extract on OSC $(date -Is) ==="
for ds in claude_twin_v1 claude_canon_all_v1; do
  if [ ! -f "$STAGE/$ds.tar.gz" ]; then
    continue
  fi
  echo "EXTRACT $ds"
  $SSH "tar -C '$R/$ds' -xzf '$R/incoming/$ds.tar.gz' && find '$R/$ds' -type f | wc -l && du -sm '$R/$ds'"
done
echo "FAST_DONE $(date -Is)"
