"""Refresh-aware training for the 4/1 adapter+LoRA path (same params as arm B).

Inference (ouro/evaluate_refresh.py): generated tokens are grouped into blocks
of N. Inside a block the 1-loop reader decodes; when the block closes, the
writer re-runs 4 loops on it and the adapter-mapped last-loop KV replaces the
reader KV for those positions; the writer predicts the next token.

Training reproduces exactly that cache in one parallel pass. For suffix input
position i in block b = i // N, the reader query sees:
    prefix                     -> adapter(writer loop-4 KV)
    suffix positions in blocks < b -> adapter(writer loop-4 KV)
    suffix positions in block b, <= i -> reader 1-loop KV
Targets predicted by the writer (the first suffix target and the target after
each block's last token) equal the teacher and are excluded from the loss.
"""

import torch
import torch.nn.functional as F

from ouro.adapter_model import AdapterSystem, PrefixState, backbone
from ouro.model import NativeCache


def refresh_mask(prefix_len, suffix_len, block, device):
    i = torch.arange(suffix_len, device=device)
    bi = i // block
    mapped = bi[None, :] < bi[:, None]
    own = (bi[None, :] == bi[:, None]) & (i[None, :] <= i[:, None])
    prefix = torch.ones(suffix_len, prefix_len, dtype=torch.bool, device=device)
    return torch.cat((prefix, mapped, own), dim=1)[None, None]


def loss_positions(suffix_len, block, device):
    # Position i predicts target i+1; the writer predicts it when i closes a block.
    i = torch.arange(suffix_len, device=device)
    return (i % block) != (block - 1)


class RefreshSystem(AdapterSystem):
    def __init__(self, writer, reader):
        super().__init__(writer, reader, "B")

    def forward(self, prefix_ids, suffix_ids, block):
        depth = self.writer.config.num_hidden_layers
        teacher_first, prefix_state, teacher_cache = self.writer_prefix(prefix_ids, 4)
        suffix_input = suffix_ids[:, :-1]
        P, S = prefix_ids.shape[1], suffix_input.shape[1]
        with torch.no_grad():
            self.capture_enabled = True
            hidden, teacher_cache = backbone(self.writer, suffix_input, teacher_cache, 4)
            self.capture_enabled = False
            suffix_state = PrefixState(
                list(self.raw_keys),
                [teacher_cache.value_cache[3 * depth + l][:, :, -S:] for l in range(depth)],
                teacher_cache.cos, teacher_cache.sin,
            )
            self.raw_keys = [None] * depth
            teacher_logp = self.writer.lm_head(hidden).float().log_softmax(-1)
        del teacher_cache, hidden
        mapped_prefix = self.adapter(prefix_state)
        mapped_suffix = self.adapter(suffix_state)
        cache = NativeCache(depth)
        for l in range(depth):
            cache.key_cache.append(torch.cat((mapped_prefix.key_cache[l], mapped_suffix.key_cache[l]), dim=2))
            cache.value_cache.append(torch.cat((mapped_prefix.value_cache[l], mapped_suffix.value_cache[l]), dim=2))
        cache._seen_tokens = P + S
        mask = refresh_mask(P, S, block, prefix_ids.device)
        positions = torch.arange(P, P + S, device=prefix_ids.device)
        self.reader.config.total_ut_steps = 1
        self.reader.model.total_ut_steps = 1
        outputs, _, _ = self.reader.model(
            input_ids=suffix_input,
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            position_ids=positions[None],
            cache_position=positions + S,
            past_key_values=cache,
            use_cache=True,
        )
        student_logp = self.reader.lm_head(outputs.last_hidden_state).float().log_softmax(-1)
        keep = loss_positions(S, block, prefix_ids.device)
        kl = F.kl_div(student_logp, teacher_logp, reduction="none", log_target=True).sum(-1)[0]
        kl_sum = kl[keep].sum()
        with torch.no_grad():
            target = suffix_ids[:, 1:, None]
            ce = -student_logp.gather(-1, target)[0, :, 0][keep].sum()
            teacher_ce = -teacher_logp.gather(-1, target)[0, :, 0][keep].sum()
        return kl_sum, torch.stack((kl_sum.detach(), ce, teacher_ce, keep.sum().to(ce.dtype)))
