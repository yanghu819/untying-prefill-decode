#!/bin/bash
# after_timing.sh CMD...: wait until the timing queue is empty and no timed request runs on this box (so the timing
# pass stays on an otherwise idle machine), then run CMD (e.g. a sweep_depth.sh launch).
R=/dev/shm/big-small-ouro/runs; cd $R
until [ ! -s $R/timing_q.txt ] && ! pgrep -f "[t]ime_requests.py" > /dev/null; do sleep 60; done
exec "$@"
