#!/bin/bash
# The Windows download flattened the shared-library symlinks. Recreate libX.so.A.B.C -> libX.so.A.B, libX.so.A, libX.so
cd ~/gcp/gt_amd64/lib || exit 1
for f in *.so.*.*.*; do
  [ -f "$f" ] || continue
  base=${f%%.so.*}; ver=${f#*.so.}          # e.g. libglobus_io, 3.8.14
  IFS=. read -r a b c <<< "$ver"
  ln -sf "$f" "$base.so.$a.$b"
  ln -sf "$f" "$base.so.$a"
  ln -sf "$f" "$base.so"
done
# libs with only two version parts (libfoo.so.A.B)
for f in *.so.*.*; do
  [ -f "$f" ] && [ ! -L "$f" ] || continue
  case "$f" in *.so.*.*.*) continue;; esac
  base=${f%%.so.*}; ver=${f#*.so.}; IFS=. read -r a b <<< "$ver"
  ln -sf "$f" "$base.so.$a"; ln -sf "$f" "$base.so"
done
echo "symlinks now: $(find . -maxdepth 1 -type l | wc -l)"
export LD_LIBRARY_PATH="$HOME/gcp/gt_amd64/lib:${LD_LIBRARY_PATH:-}"
echo "missing libs for gridftp server:"; ldd ../sbin/globus-gridftp-server 2>&1 | grep -i 'not found' | head -n 6 || true
cd ~/gcp
./globusconnectpersonal -stop >/dev/null 2>&1; sleep 4
setsid nohup ./globusconnectpersonal -start > ~/gcp_start.log 2>&1 < /dev/null &
sleep 30
./globusconnectpersonal -status
