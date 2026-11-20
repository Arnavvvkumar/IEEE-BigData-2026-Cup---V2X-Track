#!/bin/bash
# Pause the dataset uploads, then measure raw uplink: (1) to OSC over ssh, (2) to a public speed endpoint.
pkill -f upload_data.sh; pkill -f 'tar -C'; sleep 3
echo "stray tar/ssh: $(pgrep -fc 'tar -C')"
head -c 30000000 /dev/zero | tr '\0' 'x' > /tmp/speed30.bin
s=$(date +%s.%N)
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -c aes128-gcm@openssh.com ${OSC_USER}@ascend.osc.edu 'cat > /dev/null' < /tmp/speed30.bin 2>/dev/null
e=$(date +%s.%N)
echo "OSC ssh upload of 30 MB: $(echo "scale=2; 30 / ($e - $s)" | bc) MB/s"
s=$(date +%s.%N)
curl -s -o /dev/null -X POST --data-binary @/tmp/speed30.bin --max-time 90 https://speed.cloudflare.com/__up
e=$(date +%s.%N)
echo "public endpoint upload of 30 MB: $(echo "scale=2; 30 / ($e - $s)" | bc) MB/s"
rm -f /tmp/speed30.bin
