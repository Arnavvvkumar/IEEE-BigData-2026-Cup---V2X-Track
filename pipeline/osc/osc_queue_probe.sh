#!/bin/bash
# READ-ONLY: why is job 8548278 pending, when does Slurm expect it to start, and how busy / backed up are the GPU partitions?
F='authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$'
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E "$F" | cut -c1-210
source /etc/profile 2>/dev/null
echo "--- my jobs"; squeue -u <osc_user> -o "%i %P %T %M %l %R"
echo "--- expected start"; squeue -j 8548278 --start -o "%i %S %R" 2>/dev/null
echo "--- priority"; sprio -j 8548278 2>/dev/null | head -n 4
echo "--- pending GPU jobs ahead per partition (all users)"
for p in nextgen preemptible-nextgen debug-nextgen longgpu quad preemptible-quad; do
  echo "$p: pending $(squeue -h -p $p -t PD 2>/dev/null | wc -l)  running $(squeue -h -p $p -t R 2>/dev/null | wc -l)"
done
echo "--- GPU nodes free (gpu gres idle per node, nextgen)"
sinfo -p nextgen -N -h -o "%N %t %G %C" 2>/dev/null | awk '{s[$2]++} END{for(k in s) print k, s[k]}'
echo "--- recent job start waits (sacct, nextgen, A100 jobs started today)"
sacct -a -S today -X -n -P --format=Partition,Submit,Start,State,AllocTRES 2>/dev/null | grep -E '^nextgen\|' | grep 'gres/gpu' | grep -v 'Unknown' | awk -F'|' '{print $2" "$3}' | head -n 400 | python3 -c "
import sys, datetime as d
w=[]
for l in sys.stdin:
    a,b=l.split()
    try: w.append((d.datetime.fromisoformat(b)-d.datetime.fromisoformat(a)).total_seconds()/3600)
    except Exception: pass
w.sort()
print('jobs started today:',len(w),'| wait hours median %.1f p90 %.1f max %.1f'%(w[len(w)//2],w[int(len(w)*.9)],w[-1]) if w else 'none')
"
echo "--- my account usage"
sshare -A <osc_project> -u <osc_user> 2>/dev/null | tail -n 3
EOF
