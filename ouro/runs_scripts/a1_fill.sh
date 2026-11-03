#!/bin/bash
# a1 scheduler while the Lev m < n runs finish: GPU 7 starts on the Ouro-1.4B queue at once; the first four GPUs that
# fall idle train the native 3/3 distillation control (launch_x26_distill.sh H); every other GPU joins the queue.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/grid14b_q.txt; cd $R
idle() { [ -z "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader)" ]; }
nohup ./grid_worker.sh 7 $Q > /dev/null 2>&1 < /dev/null &
while true; do
  free=""; for g in 0 1 2 3 4 5 6; do idle $g && free="$free $g"; done
  sleep 120; still=""; for g in $free; do idle $g && still="$still $g"; done   # idle twice: not a Lev train->eval gap
  set -- $still; [ $# -ge 4 ] && break
done
H="$1,$2,$3,$4"; echo "$(date +%T) H48 on $H"
nohup ./launch_x26_distill.sh H $H 29813 > launch-H48.log 2>&1 < /dev/null &
sleep 300   # let H48 allocate before the workers look for idle GPUs
for g in 0 1 2 3 4 5 6; do case ",$H," in *",$g,"*) ;; *) nohup ./grid_worker.sh $g $Q > /dev/null 2>&1 < /dev/null & ;; esac; done
wait
