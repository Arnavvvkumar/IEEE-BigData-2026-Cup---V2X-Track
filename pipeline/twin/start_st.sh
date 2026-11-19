#!/bin/bash
cd "${BDC_ROOT}/experiments/claude_carla_overnight/twin"
setsid nohup bash run_twin_st.sh > run_twin_st.log 2>&1 < /dev/null &
sleep 90
cat run_twin_st.log
