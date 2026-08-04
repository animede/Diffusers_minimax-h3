#!/bin/bash
LOGFILE=/home/animede/minimax-h3/logs/du_monitor.log
CACHE=/home/animede/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3
while true; do
  SIZE=$(du -sh "$CACHE" 2>/dev/null | cut -f1)
  AVAIL=$(df -h / | tail -1 | awk '{print $4}')
  TS=$(date '+%H:%M:%S')
  echo "$TS cache_size=$SIZE disk_avail=$AVAIL" >> "$LOGFILE"
  # Safety: abort check - if cache exceeds 160GB, something is wrong
  SIZE_GB=$(du -s --block-size=1G "$CACHE" 2>/dev/null | cut -f1)
  if [ -n "$SIZE_GB" ] && [ "$SIZE_GB" -gt 170 ]; then
    echo "$TS WARNING: cache size ${SIZE_GB}GB exceeds 170GB budget!" >> "$LOGFILE"
  fi
  if ! pgrep -f "scripts/download_t2va.py" > /dev/null; then
    echo "$TS download process no longer running, stopping monitor" >> "$LOGFILE"
    break
  fi
  sleep 30
done
