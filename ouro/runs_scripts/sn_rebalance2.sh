#!/bin/bash
# idle a1 GPU 6 takes native 1/1 MATH-500 shard 2
source <(sed -n "/^M=/,/^q() /p" sn_head.sh)
q 6 "math 6 native1 2"
