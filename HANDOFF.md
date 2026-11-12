# Handoff (2026-11-11)

For the next agent taking over *Untying Prefill and Decode Loops in Looped Language Models*.
Read this file, then `NOTES.md` (lab log), `REPRODUCE.md` (result -> script map) and `DEPRECATED.md`. Cluster addresses
and login are **not** in this public repo: they are in `~/.b300-handoff/CLUSTER.md` on the user's Mac.

## 1. Where things stand

- **Paper**: `paper/latex/main.tex`, restructured on the advisor's review (2026-11-07) into a **4-page arXiv paper**
  (body = Introduction..Conclusion must end on page 4; Limitations, references and appendix follow). The user submits
  to arXiv personally (never submit or post in their name).
- **Story (advisor order)**: (1) where the time goes: prefill 0.1-3% of time on chain-of-thought tasks (Figure 1);
  (2) prefill depth at fixed decode depth, same weights, m = 1..16: 11-bench average 47.2 -> 56.8 at n=1, gain stops
  at 2-4 loops on most benchmarks, also for an adapter trained to m=16, on Ouro-2.6B-Thinking and on Ouro-1.4B
  (Figure 2, Appendix C/D); (3) same measured budget (two clean timing passes, maps folded into the prompt cache):
  at n=1 Ours 4/1 beats Original 1/1 and Distilled 1/1 at equal time, a second decode loop buys more on long CoT, at
  n=2 Ours / Distilled / prefill-4 LoRA are within noise (Figure 3, Table 1); (4) method: aligned loop adapter
  (Ours); maps matter for long prompts (frozen maps: RULER-16K 64.7 -> 49.8); exceptions: 4-shot MATH-500 (0-shot
  reverses it), HELMET exact recall. Naming: Original / Distilled / Ours, "m/n".
- **Seeds in Table 1 caption**: Ours 56.7+-0.1 / 67.1+-0.2, Distilled 2/2 65.1+-0.2 (3 seeds each).
- **Guards**: every number quoted in the text is checked in `ouro/scripts/paper_numbers.py` (`!!` lines); the pipeline
  must print "0 printed values differ" and no `!!`.
- **Done since 11-09**: baseline LR check (Distilled at x0.3 / x3 and rank 80 x3: averages move <= 0.6, never reach
  Ours; Setup sentence + Appendix "Baseline learning rate"); Distilled 1/1 and 2/2 three seeds; depth gain over three
  seeds (9.0 +- 0.6, 9.1 +- 0.2, Section 4); maps frozen in the aligned layout (RULER-16K -14.9, seed 2 -22.3; 4K/8K
  within 1.4); Ouro-1.4B depth sweep (Appendix D).
- **Status 2026-11-11: ready for arXiv.** Data frozen; red team round 5 (3/5, no number mismatch) addressed; body 4
  pages; every quoted number guarded; provenance regenerated and scanned; arXiv bundle `paper/arxiv/arxiv.tar.gz`
  checked (all files present, main.bbl 93 entries). The user submits it personally.
- **Still running, not in the paper** (add to the appendix later if wanted, the paper does not depend on them):
  rank-80 Distilled seeds (1/1 s2, 2/2 s2 and s3; `R80_SEEDS`), wide adapter seeds 2/3 and their depth sweeps
  (X26-a-wide-s{2,3}, X26ws{2,3}-*; `plateau()` needs a seed prefix to use them).
- Code benchmarks need `runs/score_evalplus_all.sh` on each box after their lm-eval runs.
- Backups: `ouro/scripts/backup_ms.py BOX [--data]` (every 2 h, log /root/pn/backup-BOX.log) to the private
  ModelScope repo torushy/untying-prefill-decode-backup; /dev/shm is a tmpfs and is lost on a restart.
- Sync and pipeline: `~/.b300-handoff/pn_sync.sh [export|pipeline|all]`.
- **Operational lessons (2026-11-09)**: queues are per box; before moving queue lines between boxes check that every
  checkpoint they use exists on the target. `grid_worker.sh GPU Q MAX` only starts a job when the GPU has fewer than
  MAX compute processes (use MAX=2 next to a training job). The a2 box is the timing box: time only when it is idle. Never overwrite a launcher script (`runs/*.sh`) while an instance
  of it is still running: bash reads scripts as it goes, and the running copy then fails after training (11-10:
  fixed4-s3 and lora-s3 lost their eval step; re-run the launcher, which skips the finished training).
- **v2 trainable-prefill check (11-12)**: seed 1 done and in Appendix "Where the Gain Stops" (paragraph "Training the
  prefill as well", Table `tab:prefilltrn` from `paper_figs.tab_prefilltrn`): X26-a-pf8 (prefill LoRA) vs X26-a-w8
  (frozen), both m in {1,2,3,4,6,8}, prompts cut to 2,048 tokens; the stop does not move (16 settings: later 2, same 8,
  earlier 6), RULER-16K peaks rise (63.7 -> 66.0 at n=1, 75.2 -> 81.0 at n=2). **In flight**: seed 2 of both arms
  (X26-a-w8-s2 on a1 from 05:20, X26-a-pf8-s2 on a2 from 05:02, then sweeps X26w8s2-* / X26pf8s2-*; launcher log
  runs/{w8s2,pf8s2}.log). When they land: score evalplus on both boxes, add a seed loop to `prefill_trainable()` like
  `plateau_seeds`, report the stop over the seeds. Seed 3 (X26-a-w8-s3 on a1, X26-a-pf8-s3 on a2) starts by itself when each box's
  seed-2 sweep queue drains (/root/pn/after_sweep_*s3.sh, armed 11-12 17:45). After that: ARR checklist TODOs with the user if the paper also goes
  to ARR.

## 2. The user's rules (all still in force)

- Reply in **Chinese**, concise. Lead with the finding, not the process.
- **One core story** in the main text; odd/side findings go to the appendix with a one-line pointer. No new side
  experiments; only runs that test a core claim. Never silently drop a contradicting result: appendix + Limitations.
- Terminology: **"prefill m / decode n"**; the method is **LoopBridge** (user 2026-11-11; "Ours" in tables, "loop
  adapter" in older appendix text). Never deep/shallow (also not "deeper prefill": say "more prefill loops"),
  read/write, connector, 深读浅写. Title: *Untying Prefill and Decode Loops in Looped Language Models* (user 11-11;
  a "Deep Prompt, Shallow Decoder" title used 11-07..11-11 broke this rule and was reverted).
  No claim that more prefill loops are better; the best (m, n) depends on the task. **Run both directions (m < n and
  m > n) everywhere.**
- Lead with strengths ("what we can do"), don't dwell on weaknesses, never claim omnipotence; no invented metrics, no
  jargon (a CS336-level reader must follow), Kaiming-He-style simplicity, torch-style pseudocode.
- **Figures** (user was explicit, several times): every figure must tell one story that a reader gets at a glance.
  Use only chart types common in big-lab LLM tech reports. The user can really only read **Pareto-front-style line
  charts**: so every result figure uses one style — grey line = native n/n, red line = loop adapter, small labels at the
  points (`frontier()` in `paper_figs.py`). No heatmaps, difference bars, dumbbells, 3-D, or anything unusual. Captions
  start with the finding in bold. In the group report, each section title is the figure's story in one sentence.
- **Reproducibility**: every number in the paper is recomputed by `ouro/scripts/paper_numbers.py` from
  `ouro/results/items/` and checked against hard-coded expectations ("0 printed values differ"). New numbers go into
  its CMP/MEANS/guards first, never hand-copied into LaTeX. Generated tables are `\input` via
  `\csname @@input\endcsname ../figs/<tab>.tex` and inlined by `build_arxiv.py`.
- **Repo**: only `yanghu819/untying-prefill-decode` (public). Keep it core; mark leftovers in `DEPRECATED.md` or delete.
  Push at every check-in; log in `NOTES.md` (English). `yanghu819/big-small` is archived — never commit there.
- **Group report**: local HTML only (never publish an artifact). Current: `~/Documents/ChatGPT/B300-xinchen/ouro/
  results/组会汇报-v53.html`, generated by `make_report.py` in the same folder; older versions in `old/`. The report
  shows only current content — no history, no outdated framing.
- Use Codex as an independent reviewer for claims and figures (`codex exec -s read-only -c tools.web_search=true
  --skip-git-repo-check -o OUT.md "<prompt>"` from the repo root; model gpt-6-astra, effort ultra). Don't over-trust it.
- The user wants maximum impact (also being found by research agents: README, `llms.txt`, `CITATION.cff`).

## 3. Pipeline (always in this order before committing or asking for a review)

```bash
# (if new runs) export per-item scores from both boxes and merge: ~/.b300-handoff/pn_sync.sh
cd ouro && uv run --with numpy --with math-verify python scripts/paper_numbers.py compute       # must end "0 printed values differ"
uv run --with numpy --with matplotlib python scripts/paper_figs.py           # all figures + generated tables
cd ../paper/latex && tectonic -X compile main.tex                            # check: no Overfull, no undefined refs
cd .. && python3 build_arxiv.py                                              # arXiv tarball + paper/*.pdf refresh
# before git commit: grep the diff for the cluster address, its ports and token prefixes listed in
# ~/.b300-handoff/CLUSTER.md (never write them in this repo); the grep must come back clean
```

No system numpy on the Mac: always `uv run --with ...`. LaTeX is `tectonic`.

## 4. Pitfalls we actually hit (each cost hours)

**Cluster / ops**
- A dropped SSH socket during `pn_sync` once **wiped `ouro/results/items`** (export returned nothing). The script now
  aborts if a box is unreachable or an export has ≤100 files. If items vanish: `git checkout -- ouro/results`.
- `ssh host 'cat > f && nohup ... &' < localfile` can leave `f` **empty**. Copy files with `scp -o ControlPath=...`.
- **Kill only by exact PID.** `pkill -f`/`pgrep -f` patterns matched and killed the calling shell more than once.
- Ouro-2.6B-Thinking **MATH shards hang for hours** (happened 3×). Check the log mtime; kill the exact PID and rerun —
  the lm-eval runs are resumable (`watchdog.sh` exists for this).
- Box-to-box copies: go **directly** box→box with agent forwarding (~275 MB/s). Relaying through the Mac is <4 MB/s.
- `hf-mirror` times out from the boxes; **ModelScope** works (`modelscope.cn/models/<org>/<name>/resolve/master/<f>`;
  its API lists sha256 per file = HF LFS blob name, use it to verify identical weights).
- `/dev/shm` on a2 is ~60% full; check `df -h /dev/shm` before saving checkpoints.
- Code on the boxes lives in `/root/big-small/` (a plain copy, **not a git repo**, old layout); the run launchers are
  copied into `/dev/shm/big-small-ouro/runs/`. When you change a script locally, scp it to the box; `pn_sync.sh` pushes
  the local `paper_numbers.py` before exporting.
- Only use the free guaranteed boxes (保障资源). **No purchases or renewals**, no elastic instances. Don't clean other
  people's data on shared storage. Don't open always-on-top/focus-stealing windows on the user's Mac.
- Reading console credentials through the browser is blocked by the permission classifier; ask the user.

**Evaluation**
- Code benchmarks (HumanEval/MBPP+) need `--confirm_run_unsafe_code` or lm-eval refuses the task.
- HELMET natives once ran on the full 6,646 items instead of `bench-mini` (1,108): always check item counts (`n` in
  `paper_numbers.json`) — comparisons are on shared items only.
- Ouro-1.4B native keys must map to the 1.4B runs (`t14()` in `paper_numbers.py`); a wrong mapping silently compared
  1.4B adapters with 2.6B natives.
- Thinking checkpoints on raw HELMET prompts can stop at once; we decode so an answer cannot be empty, for every
  configuration (documented in the appendix).
- GPQA questions are **never redistributed** (only per-item scores keyed by index). LevBench's **locked test split
  must never be evaluated** — development set only, answerable items, bootstrap over record groups (as Lev does).
- Prompt format changes the best m: on the base model, prefill 1 helps 4-shot MATH-500 but hurts 0-shot MATH-500.

**Paper / figures**
- A reviewer was given a **stale PDF** (figures regenerated, LaTeX not recompiled). Always run the full pipeline above.
- `build_arxiv.py`'s `plain()` must handle `$(m,n)$` (regex fixed once). Numbers quoted from generated tables must also
  be added to the guards in `paper_numbers.py`.
- Codex caught real overclaims: "property of the task", "same decode cost" (equal decode loops ≠ equal latency:
  prefill 4 / decode 2 is 1.27× native 2/2 per token), "fourteen benchmarks" (13), "every benchmark" (12/13). Check
  every count and universal claim against `paper_numbers.json`.
- When deleting a function from `paper_numbers.py`, check you did not also delete a module-level block between it and
  the next `def` (this removed `LEV_CMP` once).

**Never**
- Print or commit passwords/tokens; commit box IPs, ports or instance IDs to the public repo; put the user's email in
  the paper; post on GitHub (issues/discussions) in the user's name (attempt was blocked — don't retry).
