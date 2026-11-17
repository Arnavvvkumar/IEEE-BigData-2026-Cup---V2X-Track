#!/bin/bash
a=$(awk '/eth0/ {print $10}' /proc/net/dev)
sleep 12
b=$(awk '/eth0/ {print $10}' /proc/net/dev)
python3 -c "b=$b; a=$a; print('upload_Mbps', round((b-a)/12*8/1e6, 2)); print('upload_MBps', round((b-a)/12/1e6, 3))"
echo "shards $(pgrep -fc upload_shard.sh || true)"
echo "rsyncs $(pgrep -fc 'rsync -aL --partial --size-only' || true)"
pgrep -af 'upload_rsync.sh' || echo "old uploader gone"
