#!/bin/bash
# Second half of the restore: wait for the ModelScope downloads, unpack, and fetch Ouro-2.6B from the HF mirror.
set -e
M=/dev/shm/big-small-ouro
until ! pgrep -f "/dev/shm/restore.sh" >/dev/null; do sleep 10; done
python3 - <<PY
from modelscope.hub.api import HubApi
from modelscope import snapshot_download
HubApi().login(open("/dev/shm/.ms_tok").read().strip())
snapshot_download(repo_id="torushy/big-small-backup", repo_type="dataset", local_dir="/dev/shm/msbak")
snapshot_download(repo_id="nanbeige/Nanbeige4.2-3B-Base", local_dir="$M/nanbeige-3b-base")
print("ms downloads complete", flush=True)
PY
cd $M; for t in /dev/shm/msbak/data/*.tar; do tar -xf $t; done; echo "data extracted"
cd /root/big-small/environments && tar -xf /dev/shm/msbak/env/ouro-env.tar; echo "env extracted"
cd $M/runs && tar -xf /dev/shm/msbak/results/bench.tar && tar -xf /dev/shm/msbak/results/lmeval-out.tar && tar -xf /dev/shm/msbak/results/throughput-grid.tar; echo "results extracted"
cp -rn /dev/shm/msbak/checkpoints/* $M/runs/; echo "checkpoints restored"
HF_ENDPOINT=https://hf-mirror.com /root/big-small/environments/ouro/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('ByteDance/Ouro-2.6B', local_dir='$M/model-2.6b', max_workers=8)
print('ouro downloaded')"
rm -f /dev/shm/.ms_tok
du -sh $M/* /root/big-small/environments/ouro | tr "\n" " "; echo RESTORE_DONE
