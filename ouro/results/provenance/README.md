# Provenance

Every number in the paper is recomputed by `ouro/scripts/paper_numbers.py` from the per-item scores in
`ouro/results/items/`. This directory records where each of those item sets came from. It is written by
`ouro/scripts/provenance.py`, run on the evaluation machines after the last evaluation; machine names and addresses
are removed from logs.

| path | content |
|---|---|
| `runs/<run>.json` | one evaluation run: model arguments (model, prefill/decode condition, checkpoint), shots, limits, task versions and hashes, lm-evaluation-harness and transformers versions, start time and duration, the harness's aggregate results, sha256 and item count of every raw sample file, the queue command(s) that launched it, and the checkpoint's sha256 and training run |
| `logs/<run>.log` | the run's evaluation log, progress bars removed |
| `logs/e2e/`, `logs/throughput/`, `logs/lev-training/` | logs of the end-to-end timing, decode-throughput and Lev training runs (their results are in `ouro/results/e2e`, `throughput-sb4`, `lev`) |
| `training/<run>/` | every checkpoint behind a result: `setup.json` (arguments, data manifest, model revision, trainable parameters), `result.json`, the full per-step training log `train.jsonl.gz` and `train.log` |
| `outputs/<run>.jsonl.gz` | the model's output for every item of generative tasks (GSM8K, MATH-500, BBH, MMLU-Pro, IFEval, code, RULER, HELMET): item id, end of the prompt, output, reference and score |

Run names map to paper keys as in `paper_numbers.py`: `L:<run>:<metric>` (lm-evaluation-harness), `P:<run>:...`
(evalplus scoring of an lm-evaluation-harness run), `T:<prefix>:...` (runs `<prefix>_s0`, `_s1`, ...),
`B:<run>:<bench>` (our benchmark scripts). Log-likelihood tasks (MMLU, ARC, HellaSwag, Winogrande, GPQA) store no
generated text; their per-item scores are in `ouro/results/items/`. Checkpoints themselves are not in the repository;
their sha256 identifies them.
