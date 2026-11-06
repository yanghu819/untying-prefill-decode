"""Provenance of every number in the paper: for each run behind a paper_numbers.py key, the exact command, evaluation
configuration, software versions, times, checkpoint hash and training record, plus a cleaned log and the model's
outputs. Run on each evaluation box (where the runs live); paper_numbers.py must sit next to this file.

    python provenance.py RUNS_DIR OUT_DIR

OUT_DIR/runs/<run>.json            command(s), config, versions, times, checkpoint sha256, item count, output sha256
OUT_DIR/logs/<run>.log             evaluation log without progress bars
OUT_DIR/training/<run>/            setup.json, result.json, train.jsonl and train.log (no progress bars) of each checkpoint
OUT_DIR/outputs/<run>.jsonl.gz     per item: id, prompt tail, model output, reference and score (generative tasks)
OUT_DIR/logs/{e2e,throughput,lev-training}/   logs of the timing, throughput and Lev runs
"""
import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_numbers as PN  # noqa: E402

BAR = re.compile(r"\r|\d+%\|")          # tqdm progress lines
IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
HOST = socket.gethostname()
_SHA = {}


def scrub(text):
    """Drop machine identifiers (host name, IP addresses) from logs before they leave the box."""
    return IP.sub("<ip>", text.replace(HOST, "<host>"))


def sha256(path):
    path = str(path)
    if path not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _SHA[path] = h.hexdigest()
    return _SHA[path]


def clean_log(src, dst):
    if not os.path.exists(src):
        return False
    lines = [l for l in open(src, errors="replace") if not BAR.search(l)]
    dst.write_text(scrub("".join(lines[-400:])))
    return True


def commands(runs, run):
    """Queue log lines that launched this run (grid_worker.sh writes 'HH:MM:SS gpuN: <command>')."""
    pat = re.compile(rf"(?<![\w-]){re.escape(run)}(?![\w-])")
    out = []
    for f in sorted(set(glob.glob(f"{runs}/*_q.txt.log")) | set(glob.glob(f"{runs}/jobs*.log"))):
        out += [f"{Path(f).name}: {l.strip()}" for l in open(f, errors="replace") if pat.search(l)]
    return out


def checkpoint_of(text):
    m = re.search(r"checkpoint=([^,\s\"]+)", text or "") or re.search(r"(/[^,\s\"]+/checkpoint-[\w.]+\.pt)", text or "")
    return m.group(1) if m else None


def training(ckpt, out):
    """Copy the training record of the run that produced `ckpt`; return its summary."""
    run_dir = Path(ckpt).parent
    dst = out / "training" / run_dir.name
    if not dst.exists():
        dst.mkdir(parents=True)
        for name in ("setup.json", "result.json", "progress.json"):
            if (run_dir / name).exists():
                shutil.copy(run_dir / name, dst / name)
        if (run_dir / "train.jsonl").exists():
            with open(run_dir / "train.jsonl", "rb") as f, gzip.open(dst / "train.jsonl.gz", "wb") as g:
                g.write(f.read())
        clean_log(run_dir / "train.log", dst / "train.log")
    return dict(run=run_dir.name, checkpoint=Path(ckpt).name, sha256=sha256(ckpt), bytes=os.path.getsize(ckpt))


def lm_record(runs, run, out):
    rec = dict(run=run, kind="lm-evaluation-harness")
    res = sorted(glob.glob(f"{runs}/lmeval/out/{run}/*/results_*.json"))
    if res:
        d = json.load(open(res[-1]))
        cfg = d.get("config", {})
        rec.update(model_args=cfg.get("model_args"), limit=cfg.get("limit"), n_shot=d.get("n-shot"),
                   n_samples=d.get("n-samples"), task_versions=d.get("versions"), task_hashes=d.get("task_hashes"),
                   lm_eval_version=d.get("lm_eval_version"), transformers_version=d.get("transformers_version"),
                   start=datetime.fromtimestamp(d["date"], timezone.utc).isoformat() if d.get("date") else None,
                   seconds=d.get("total_evaluation_time_seconds"), results=d.get("results"))
    samples = sorted(glob.glob(f"{runs}/lmeval/out/{run}/*/samples_*.jsonl"))
    rec["samples"] = [dict(file=Path(s).name, sha256=sha256(s), items=sum(1 for _ in open(s))) for s in samples]
    for side in glob.glob(f"{runs}/lmeval/out/{run}/evalplus_*.jsonl"):
        rec.setdefault("evalplus", []).append(dict(file=Path(side).name, sha256=sha256(side)))
    rec["log"] = clean_log(f"{runs}/lmeval/logs/{run}.log", out / "logs" / f"{run}.log")
    gen = [s for s in samples if not re.search(r"samples_(mmlu|hellaswag|arc_|winogrande|gpqa)", Path(s).name)]
    if gen:   # model outputs of generative tasks
        with gzip.open(out / "outputs" / f"{run}.jsonl.gz", "wt") as g:
            for s in gen:
                task = Path(s).name.split("samples_")[1].rsplit("_", 1)[0]
                for line in open(s):
                    r = json.loads(line)
                    arg = r.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
                    g.write(json.dumps(dict(task=task, doc_id=r["doc_id"], filter=r.get("filter"),
                                            prompt_tail=arg[-300:] if isinstance(arg, str) else None,
                                            output=r["resps"][0][0], target=r.get("target"),
                                            **{k: v for k, v in r.items() if k in ("exact_match", "acc", "math_verify",
                                                                                     "prompt_level_strict_acc")}))
                            + "\n")
    return rec


def bench_record(runs, run, out):
    rec = dict(run=run, kind="evaluate_bench (RULER / HELMET / BBH / MMLU-Pro scripts)")
    files = sorted(glob.glob(f"{runs}/bench/{run}/s*/*.jsonl"))
    rec["outputs"] = [dict(file=str(Path(f).relative_to(f"{runs}/bench")), sha256=sha256(f), items=sum(1 for _ in open(f)))
                      for f in files]
    logs = sorted(glob.glob(f"{runs}/bench/{run}/s*.log"))
    if logs:
        text = "".join(f"== {Path(l).name}\n" + "".join(x for x in open(l, errors="replace") if not BAR.search(x))[-20000:]
                       for l in logs)
        (out / "logs" / f"{run}.log").write_text(scrub(text))
        rec["log"] = True
    if files:
        with gzip.open(out / "outputs" / f"{run}.jsonl.gz", "wt") as g:
            for f in files:
                for line in open(f):
                    g.write(line)
    return rec


def extras(runs, out):
    """Logs of the timing, throughput and Lev runs, whose results are already stored under ouro/results."""
    groups = {"e2e": f"{runs}/e2e/*.log", "throughput": f"{runs}/throughput-sb4/*.log",
              "lev-training": "/dev/shm/levwork/train-*.log"}
    for name, pattern in groups.items():
        files = sorted(glob.glob(pattern))
        if files:
            (out / "logs" / name).mkdir(exist_ok=True)
        for f in files:
            clean_log(f, out / "logs" / name / Path(f).name)


def main(runs, out):
    out = Path(out)
    for sub in ("runs", "logs", "training", "outputs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    names = {}
    for key in PN.keys():
        kind, name, *_ = key.split(":")
        if kind == "T":
            for d in glob.glob(f"{runs}/lmeval/out/{name}_s[0-9]"):
                names[Path(d).name] = "L"
        else:
            names[name] = {"L": "L", "P": "L", "B": "B"}.get(kind, kind)
    done = 0
    for run, kind in sorted(names.items()):
        exists = glob.glob(f"{runs}/lmeval/out/{run}") if kind == "L" else glob.glob(f"{runs}/bench/{run}")
        if not exists:
            continue
        rec = lm_record(runs, run, out) if kind == "L" else bench_record(runs, run, out)
        rec["commands"] = [scrub(c) for c in commands(runs, run)]
        ck = checkpoint_of(rec.get("model_args")) or next((checkpoint_of(c) for c in rec["commands"] if checkpoint_of(c)), None)
        if ck and os.path.exists(ck):
            rec["checkpoint"] = training(ck, out)
        (out / "runs" / f"{run}.json").write_text(json.dumps(rec, indent=1, default=str))
        done += 1
    extras(runs, out)
    print(f"provenance: {done} runs written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
