#!/bin/bash
cd ~/gcp/gt_amd64/lib
echo "total entries: $(ls | wc -l)"
ls | grep -i -E 'gridftp_server|gfork|ftp_control|libglobus_io|xio' | head -n 20
echo "--- symlinks present?"; find . -maxdepth 1 -type l | wc -l
echo "--- original download folder"
src="${BDC_ROOT}/globusconnectpersonal-latest/globusconnectpersonal-3.3.2/gt_amd64/lib"
ls "$src" 2>/dev/null | wc -l; ls "$src" | grep -i -E 'gridftp_server|gfork' | head
ls ${BDC_OUT}/ | grep -i globus
