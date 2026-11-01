"""Read deep, write shallow on Lev looped decision models (OpenLAIR/lev), training-free.

A Lev request is a state (the context) followed by branches (each question with its options); the pointer head reads
the decide and option tokens of a branch after every loop. Here the state is read with READ loops and the branches are
computed with k = 1..steps loops, attending at every layer to the state's keys and values of the last state loop
(no new parameters; Lev's own LoRA and head unchanged). Equivalent reference construction: at each layer of a branch
loop, the state rows are replaced by their inputs at the last of READ state loops.

Sanity: READ = 1 reproduces native depth 1 at write 1 exactly. READ = steps at write k = steps is NOT native depth
`steps`: in the split, branch loops 1..steps-1 already read the state's last-loop KV (native reads loop-k KV at loop k).
Native depths are what Lev's scripts/jevbench_eval.py reports without LEV_READ.

    LEV_READ=4 python ouro/lev/lev_rdws_eval.py --run <Lev checkpoint> --out <dir> --jevbench <jevbench checkout>
(arguments as Lev's scripts/jevbench_eval.py; run from a Lev checkout on PYTHONPATH)
"""
import os
import runpy
import sys
from pathlib import Path

import torch

LEV = Path(os.environ.get("LEV_ROOT", "."))
sys.path.insert(0, str(LEV))
import kev.model as kmodel   # noqa: E402
import kev.ouro as kouro   # noqa: E402

READ = int(os.environ.get("LEV_READ", "0"))
STATE_LEN = [None]

_hidden_batch = kmodel.DecisionModel.hidden_batch


def hidden_batch(self, encs):
    STATE_LEN[0] = [e["seg"].count(0) for e in encs]   # per row: state tokens come first (rows are right-padded)
    return _hidden_batch(self, encs)


_forward = kouro.OuroModel.forward


def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, **kw):
    if not READ:
        return _forward(self, input_ids, attention_mask, position_ids, past_key_values, use_cache, **kw)
    x0 = self.embed_tokens(input_ids)
    B, L = input_ids.shape
    if position_ids is None:
        position_ids = torch.arange(L, device=x0.device)[None].expand(B, -1)
    c = self.config
    inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.float32, device=x0.device) / c.head_dim))
    freqs = position_ids[:, :, None].float() * inv_freq[None, None, :]
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos()[:, None].to(x0.dtype), emb.sin()[:, None].to(x0.dtype)
    mask = self._mask(attention_mask, B, L, x0.device)
    lens = torch.tensor(STATE_LEN[0], device=x0.device)
    is_state = (torch.arange(L, device=x0.device)[None] < lens[:, None])[..., None]   # [B, L, 1]
    # read: READ loops; keep every layer's input at the last loop (its keys and values are the state's deep cache)
    x, deep_inputs = x0, []
    for r in range(READ):
        for layer in self.layers:
            if r == READ - 1:
                deep_inputs.append(x)
            x = layer(x, cos, sin, mask)
        x = self.norm(x)
    deep = x
    # write: branch rows run 1..steps loops; state rows are the deep inputs at every layer
    z, states = x0, []
    for r in range(self.steps):
        for j, layer in enumerate(self.layers):
            z = layer(torch.where(is_state, deep_inputs[j], z), cos, sin, mask)
        z = self.norm(z)
        states.append(torch.where(is_state, deep, z))
    return kouro.OuroOutput(last_hidden_state=states[-1], step_hidden_states=tuple(states))


kmodel.DecisionModel.hidden_batch = hidden_batch
kouro.OuroModel.forward = forward

if __name__ == "__main__":
    # LEV_SCRIPT: the Lev script to run under the patch (default JevBench; scripts/depth_rows.py for LevBench rows)
    if os.environ.get("LEV_MODULE"):   # e.g. kev.train: train under the read-deep forward
        sys.argv[0] = os.environ["LEV_MODULE"]
        runpy.run_module(os.environ["LEV_MODULE"], run_name="__main__", alter_sys=True)
    else:
        script = LEV / os.environ.get("LEV_SCRIPT", "scripts/jevbench_eval.py")
        sys.argv[0] = str(script)
        runpy.run_path(str(script), run_name="__main__")
