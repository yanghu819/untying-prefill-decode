# Deprecated code

These pieces are **not used by the paper**. They stay only because live code still imports them or because a
launcher in `ouro/runs_scripts/` refers to them. Do not extend them, cite them, or build new experiments on them.
Anything here can be deleted once its importer is cleaned up.

| What | Why it is deprecated | Still referenced by |
|---|---|---|
| `ouro/refresh_model.py`, arm `R` in `ouro/train_batched.py` | "refresh" decoding (re-prefill generated chunks); stopped line, it only saves latency and overlaps speculative decoding | `train_batched.py` import |
| `ouro/evaluate_gsm8k.py`, `ouro/evaluate_prefix_suffix.py` | early evaluators (3-shot GSM8K for the first four paths; PG19 NLL); replaced by `evaluate_bench.py` + lm-evaluation-harness | `evaluate_adapter.py` import |
| arms `A`, `B`, `C` (per-cell adapters), `U`, `V`, `W` (earlier universal adapters), `T`/`Q` (two-source map), `J`/`K` (shallower prefill models) | superseded by the single-bank universal loop adapter, arm `S`, which every paper result uses | `adapter_model.py`, `batched.py`, `train_batched.py` branches |
| `ouro/runs_scripts/launch_v.sh` | trains arm `V` (superseded by `S`) | — |

Paper arms, for contrast: `S` (the loop adapter, all main results), `O`/`P` (shared answer KV variant, appendix),
`M` (gated first-loop branch, appendix), `S` with `--freeze-maps` (maps fixed at identity, "What the maps do"),
`D` (LoRA distillation of native 2/2, `launch_t26d.sh`).

Removed outright (in git history if ever needed): probe and "answers without a final answer" analyses, the
training-free SenseNova appendix, `latest.json`, long-context throughput records, unused per-item score files, old
copies of figures, and Codex reviews of superseded drafts.
