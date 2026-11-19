#!/bin/bash
set -euo pipefail
cd "${BDC_ROOT}/experiments/claude_carla_overnight/boxsize"
E=${BDC_PYTHON}
D=${BDC_DATA}
$E real_box_stats.py > /dev/null
nice -n 10 $E relabel_mix.py --base $D/codex_openpcdet_custom_v1 --carla $D/claude_carla64_town02_v2c \
  --real-stats real_box_stats_selection.json --out $D/claude_v2carla_relabel_mix6000_v1
