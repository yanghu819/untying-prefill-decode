# Official Ouro-1.4B inference source

Source: https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1

Revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. `modeling_ouro.py`,
`configuration_ouro.py`, `config.json`, and `README.md` are unmodified files
from that revision. No weight or tokenizer data is vendored here.

The official model card declares Apache-2.0. The configuration source retains
its Qwen/Hugging Face copyright and Apache-2.0 header. This directory's LICENSE
is the canonical Apache-2.0 text; the HF snapshot did not contain a separate
LICENSE file.

Our inference adapter is outside this directory, in `ouro/model.py`. The pinned
Transformers 4.54.1 Cache API has read-only key/value cache properties and requires
`get_mask_sizes`; the upstream custom cache does not implement that API. The
adapter supplies those API members while retaining upstream attention and
cache-update computations. Its asymmetric variant retains the final prefill
loop's 24 banks and commits a generated token only on its final decode loop.

The official model always executes `total_ut_steps` before output selection.
The adapter changes that count for each prefill/decode call and explicitly
selects its final loop, with threshold 1.0 and no adaptive early exit.
