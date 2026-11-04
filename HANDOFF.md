# Handoff (2026-11-03)

For the next agent taking over *Untying Prefill and Decode Loops in Looped Language Models*. Read this file, then
`NOTES.md` (lab log), `REPRODUCE.md` (result → script map) and `DEPRECATED.md`. Cluster addresses and login are
**not** in this public repo: they are in `~/.b300-handoff/CLUSTER.md` on the user's Mac.

## 1. Where things stand

- **Paper**: `paper/latex/main.tex` (ACL review format, anonymous). Body ends on page 7 (limit 8), 17 pages with
  appendix. Codex's last review: weak accept. arXiv package builds (`paper/arxiv/arxiv.tar.gz`); **the user submits
  it with their own account** (never submit or post in their name).
- **Target**: ACL 2027 via ARR, deadline **2027-01-04 AoE**. `paper/ARR_CHECKLIST.md` has draft answers; its TODOs
  (licenses, total GPU hours, AI-assistance wording, anonymous code link, preprint policy) need the user.
- **Story (main text, one story only)**: a loop adapter makes prefill loops m and decode loops n independently
  choosable on a frozen looped LM (one stored prompt KV cache).
  1. The best m depends on the task (task map, Figure 1).
  2. Choosing m per benchmark on held-out halves beats native n/n significantly on 12/13 benchmarks at n=1 and 10/13
     at n=2 (Table 1; Ouro-1.4B: 5/5 and 3/5).
  3. Thinking model: prefill 4 / decode 2 is close to native 4/4, above a LoRA-distillation control.
  4. Fewer decode loops → 0.55× end-to-end time.
  5. Looped decision models (Lev, Section 4.5): more prefill than decision loops beats a same-training control on 12/13
     (m, n) pairs, fewer loses on all 13 (both directions, Table `tab:lev`). The user wants striking results in the
     main text.
  Appendix: Lev setup and checks, hybrid SenseNova-Loop-9B, seeds, design variants, prompt-format effects.
- **In flight (2026-11-03)**:
  - Same-training controls for Table 1: LoRA distillation of native 1/1, 2/2, 3/3 on base Ouro-2.6B
    (`launch_x26_distill.sh`, runs `X26-{E,D,H}48-*`; E/D on a2, H on a1). `after_x26_distill.sh` queues their 13
    benchmarks when training ends; `paper_numbers.py` `taskmap_select` then fills `distilled` / `diff_distilled`. Put
    the result into Table 1 (a column or a second table), the abstract and Limitations — whatever it shows.
  - Ouro-1.4B replication on 10 benchmarks (`grid14b_q.txt`) and full GSM8K / MMLU / 1.4B RULER (`fullsets_q.txt`) on
    a1 `grid_worker.sh` queues. When done: add the new benchmarks to `TASK_MAP_14`, update the 1.4B counts and guards.
  - Sync and pipeline: `~/.b300-handoff/pn_sync.sh [export|pipeline|all]` (exports on both boxes, merges into
    `ouro/results/items`, runs numbers/figures/LaTeX/arXiv on a2 and copies outputs back).
- **Next, in order of value** (open Codex objections, paper/codex_review_untie6.md):
  0. Evaluation protocol = Ouro's Table 16 (NOTES 2026-11-03): queues protocol_q.txt (full BBH / MMLU-Pro, HellaSwag,
     Winogrande) and evalplus_q.txt (code, chat-format evalplus; score with score_evalplus.py, then pn_sync). Still to
     queue: evalplus for T26-Df (evalplus_q.sh T26 on a1 after it trains), the 1.4B controls (X14-*), and the same
     protocol for Ouro-1.4B (protocol_q.sh 14). Then switch paper_numbers to strict GSM8K/MATH, evalplus code and full
     BBH / MMLU-Pro, and rewrite the MATH / code text: the one-prefill-loop gains there are answer format, not reasoning.
  1. distillation controls into Table 1 (E48/D48/H48 now; corrected E48f/D48f/H48f with `--first-target` on a1 (E) and a2
     (D on GPUs 0-3, H on 4-7) by `controls_first_chain.sh`; then also a corrected Thinking control T26-Df);
  2. time the distilled 2/2 Thinking control with `ouro/scripts/e2e_latency.py` on the same 64 GSM8K problems
     (and a larger fixed random sample; report TTFT, tokens, cap rates);
  3. MATH-500 parser adjudication: responses where full-response math_verify and strict extraction disagree, S14 vs
     S44 and S11 vs S41; report intended-answer correctness and format separately;
  4. overlap audit: done for mix-data-48m (Appendix A; results unchanged without overlapping items); still to run on
     mix-think-48m for the Thinking results;
  5. Codex review round 7 with the PDF (`~/.b300-handoff/codex_review.sh 7`) until no substantive objections remain;
  6. ARR checklist TODOs with the user → anonymized code mirror → final polish.

## 2. The user's rules (all still in force)

- Reply in **Chinese**, concise. Lead with the finding, not the process.
- **One core story** in the main text; odd/side findings go to the appendix with a one-line pointer. No new side
  experiments; only runs that test a core claim. Never silently drop a contradicting result: appendix + Limitations.
- Terminology: **"prefill m / decode n"**, **"loop adapter"**. Never deep/shallow, read/write, connector, 深读浅写.
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
cd ouro && uv run --with numpy python scripts/paper_numbers.py compute       # must end "0 printed values differ"
uv run --with numpy --with matplotlib python scripts/paper_figs.py           # all figures + generated tables
cd ../paper/latex && tectonic -X compile main.tex                            # check: no Overfull, no undefined refs
cd .. && python3 build_arxiv.py                                              # arXiv tarball + paper/*.pdf refresh
# before git commit: grep the diff for 150\.242|3002[0-9]|3004[0-9]|\.pw\b|token  (must be clean)
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
