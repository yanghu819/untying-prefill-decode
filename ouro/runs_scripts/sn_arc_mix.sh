#!/bin/bash
# ARC-C for the hybrid-state splices on a1 GPU 6, after its native 1/1 MATH-500 shard 2 has finished
source <(sed -n "/^M=/,/^q() /p" sn_head.sh)
q 6 "until ls lmeval/out/SN-native1-math500_think_s2/*/samples_* >/dev/null 2>&1; do sleep 300; done"     "arc 6 C21A" "arc 6 C21G" "arc 6 splice21A" "arc 6 splice21G" "arc 6 splice21"
