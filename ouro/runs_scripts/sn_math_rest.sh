#!/bin/bash
# remaining SenseNova MATH-500 shards (native 1/1 and native 2/2 s3, s4), queued after sn_head.sh on a1 GPU 4
source <(sed -n "/^M=/,/^q() /p" sn_head.sh)
q 4 "math 4 native2 3" "math 4 native2 4" "math 4 native1 0" "math 4 native1 1" "math 4 native1 2" "math 4 native1 3" "math 4 native1 4"
