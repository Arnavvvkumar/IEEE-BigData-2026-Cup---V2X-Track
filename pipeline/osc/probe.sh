#!/bin/bash
# Read-only discovery on the OSC login node.
echo "== host"; hostname; echo "HOME=$HOME"; df -h "$HOME" | tail -n 1
echo "== project / scratch"; ls -ld /fs/scratch/<osc_project>; ls /fs/scratch/<osc_project> | head; echo "files in scratch: $(find /fs/scratch/<osc_project> -maxdepth 1 | wc -l)"
echo "== groups"; id
echo "== slurm"; which sbatch squeue sinfo sacctmgr 2>&1 | head -n 5
echo "== partitions"; sinfo -o '%P %a %l %D %G %c %m' 2>&1 | head -n 20
echo "== my accounts"; sacctmgr -n -P show assoc user=$USER format=account,partition,qos 2>&1 | head -n 20
echo "== my jobs"; squeue -u $USER 2>&1 | head -n 5
echo "== modules"; (module avail cuda 2>&1 | head -n 15); (module avail python 2>&1 | head -n 15); (module avail miniconda3 2>&1 | head -n 6)
echo "== gcc / nvcc"; which gcc nvcc 2>&1 | head -n 3
echo "== quota"; (quota -s 2>&1 | head -n 5); (OSCfind 2>&1 | head -n 2)
echo PROBE_DONE
