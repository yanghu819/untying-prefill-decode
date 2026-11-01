#!/bin/bash
# usage: watchdog.sh STALL_MIN WATCH_DIR LOG cmd...
# Run cmd (which must resume on restart); kill and rerun it whenever no file under WATCH_DIR changes for STALL_MIN minutes.
S=$1; D=$2; L=$3; shift 3
for try in $(seq 1 40); do
  "$@" >> $L 2>&1 &
  P=$!; start=$(date +%s)
  while kill -0 $P 2>/dev/null; do
    sleep 60
    [ $(( $(date +%s) - start )) -lt $((S * 60)) ] && continue
    [ -n "$(find $D -type f -mmin -$S 2>/dev/null | head -1)" ] && continue
    echo "watchdog $(date -u +%FT%T) no writes in $S min, restart $try" >> $L; kill -9 $P; sleep 5
  done
  wait $P && exit 0
  sleep 120  # e.g. OOM while the GPU is full: back off before retrying
done
exit 1
