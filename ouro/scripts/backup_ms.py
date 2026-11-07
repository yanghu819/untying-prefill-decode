"""Incremental backup of a box's experiment outputs to the private ModelScope repo torushy/untying-prefill-decode-backup
(the /dev/shm working directory is a tmpfs: a container restart loses it).

usage: python3 backup_ms.py BOX [--data]     (BOX = a1 | a2: the folder in the repo; token in /root/.ms.token)
Staged under /dev/shm/ms-backup/BOX, uploaded to BOX/ in the repo; only files new or changed since the last
successful upload (manifest .uploaded.json) are sent:
  lmeval/<run>.tar.gz   lm-eval outputs per run (samples, results), re-packed when the run changed
  bench/<run>.tar.gz    evaluate_bench outputs per run (RULER, HELMET, ...)
  e2e/                  per-request timings (time_requests.py, e2e_latency.py)
  train/<run>/          training setup, logs, progress and the final checkpoint(s) (hard links, no copy)
  queue_logs/           grid_worker logs (which job ran where and when)
  data/ (--data)        the prepared training mixture and benchmark files, once
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BOX, DATA = sys.argv[1], "--data" in sys.argv
M = Path("/dev/shm/big-small-ouro"); R = M / "runs"
STAGE = Path("/dev/shm/ms-backup") / BOX
REPO = "torushy/untying-prefill-decode-backup"
STAGE.mkdir(parents=True, exist_ok=True)


def newest(d):
    return max((p.stat().st_mtime for p in d.rglob("*") if p.is_file()), default=0)


def pack(src, dst):
    """tar.gz src into dst if src changed since dst was written."""
    if dst.exists() and dst.stat().st_mtime >= newest(src):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    subprocess.run(["tar", "czf", str(tmp), "-C", str(src.parent), src.name], check=True)
    tmp.rename(dst)


def link(src, dst):
    """Stage src: a hard link for a final checkpoint (never rewritten), else a snapshot copy (logs and timings keep
    growing while a run is live, and an upload fails if a file changes under it)."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime and dst.stat().st_size == src.stat().st_size \
            and (src.suffix == ".pt" or dst.stat().st_ino != src.stat().st_ino):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    if src.suffix == ".pt":
        try:
            return os.link(src, dst)
        except OSError:
            pass
    shutil.copy2(src, dst)


for d in sorted((R / "lmeval" / "out").iterdir()):
    if d.is_dir():
        pack(d, STAGE / "lmeval" / f"{d.name}.tar.gz")
for d in sorted((R / "bench").iterdir()):
    if d.is_dir():
        pack(d, STAGE / "bench" / f"{d.name}.tar.gz")
for f in (R / "e2e").glob("*"):
    if f.is_file():
        link(f, STAGE / "e2e" / f.name)
for d in sorted(R.iterdir()):
    if d.is_dir() and (d / "setup.json").exists():
        for f in list(d.glob("*.json")) + list(d.glob("*.jsonl")) + list(d.glob("*.log")) + list(d.glob("TRAIN_DONE")):
            link(f, STAGE / "train" / d.name / f.name)
        cks = sorted(d.glob("checkpoint-*.pt"))
        if cks:
            link(cks[-1], STAGE / "train" / d.name / cks[-1].name)
for f in R.glob("*_q.txt.log"):
    link(f, STAGE / "queue_logs" / f.name)
if DATA:
    for name in ("mix-data-48m", "bench", "bench-mini"):
        if (M / name).exists():
            pack(M / name, STAGE / "data" / f"{name}.tar.gz")

manifest_path = STAGE.parent / f".uploaded-{BOX}.json"
done = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
todo = [p for p in STAGE.rglob("*") if p.is_file() and not p.name.endswith(".tmp")
        and done.get(str(p.relative_to(STAGE))) != [p.stat().st_size, int(p.stat().st_mtime)]]
print(f"{BOX}: {len(todo)} new or changed files, {sum(p.stat().st_size for p in todo) / 1e9:.2f} GB")
if todo:
    out = STAGE.parent / f"outbox-{BOX}"
    shutil.rmtree(out, ignore_errors=True)
    for p in todo:
        q = out / p.relative_to(STAGE); q.parent.mkdir(parents=True, exist_ok=True); os.link(p, q)
    from modelscope.hub.api import HubApi
    api = HubApi(); api.login(open("/root/.ms.token").read().strip())
    for attempt in range(3):   # the hub skips files already uploaded on a retry
        try:
            api.upload_folder(REPO, folder_path=str(out), path_in_repo=BOX, commit_message=f"backup {BOX}: {len(todo)} files")
            break
        except Exception as e:
            print(f"{BOX}: upload attempt {attempt + 1} failed: {str(e)[:200]}", flush=True)
    else:
        sys.exit(1)
    for p in todo:
        done[str(p.relative_to(STAGE))] = [p.stat().st_size, int(p.stat().st_mtime)]
    manifest_path.write_text(json.dumps(done))
    shutil.rmtree(out, ignore_errors=True)
    print(f"{BOX}: uploaded")
