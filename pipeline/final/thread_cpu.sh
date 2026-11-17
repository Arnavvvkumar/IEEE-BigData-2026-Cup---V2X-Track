#!/bin/bash
pid=$(ps -eo pid,cmd | grep 'final/infer_test.py' | grep -v grep | awk '{print $1; exit}')
echo "pid $pid"
ps -o pcpu,nlwp -p "$pid"
echo '--- top threads ---'
ps -eLo pcpu,tid,comm -p "$pid" | sort -nr | head -n 15
