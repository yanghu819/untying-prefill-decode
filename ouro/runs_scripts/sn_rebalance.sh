#!/bin/bash
# idle a1 GPUs 5 and 7 take ARC / code / two native 1/1 MATH shards ahead of the sn_head.sh queues (lm1.sh is resumable)
source <(sed -n "/^M=/,/^q() /p" sn_head.sh)
q 5 "arc 5 native1" "arc 5 native2" "arc 5 C11" "arc 5 C21" "math 5 native1 4"
q 7 "code 7 native1" "code 7 native2" "code 7 C11" "code 7 C21" "math 7 native1 3"
