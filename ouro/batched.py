"""Batched (padded) forwards for the adapter arms and the refresh-aware arm.

Prefixes are left-padded, suffixes right-padded. RoPE scores depend only on
relative position, so a shared position offset per row leaves every valid
score unchanged; pad keys are masked, pad targets excluded from the loss.
Returns the same (kl_sum, stats) contract as the per-sample forwards.
"""

import torch
import torch.nn.functional as F

from ouro.adapter_model import LiveCache, PrefixState, backbone, first_loop_bank, set_steps, shallow_branch
from ouro.model import NativeCache


def pad_batch(samples, device, pad_id=0):
    P = max(len(s["prefix_ids"]) for s in samples)
    S = max(len(s["suffix_ids"]) for s in samples)
    B = len(samples)
    prefix = torch.full((B, P), pad_id, dtype=torch.long)
    pmask = torch.zeros((B, P), dtype=torch.bool)
    suffix = torch.full((B, S), pad_id, dtype=torch.long)
    smask = torch.zeros((B, S), dtype=torch.bool)
    for b, s in enumerate(samples):
        p, q = s["prefix_ids"], s["suffix_ids"]
        prefix[b, P - len(p):] = torch.tensor(p)
        pmask[b, P - len(p):] = True
        suffix[b, : len(q)] = torch.tensor(q)
        smask[b, : len(q)] = True
    return prefix.to(device), pmask.to(device), suffix.to(device), smask.to(device)


def _teacher(system, prefix, pmask, suffix_input, full_mask, capture):
    depth = system.writer.config.num_hidden_layers
    first, prefix_state, cache = system.writer_prefix(prefix, 4, pmask.long())
    with torch.no_grad():
        system.capture_enabled = capture
        hidden, cache = backbone(system.writer, suffix_input, cache, 4, full_mask.long())
        system.capture_enabled = False
        logp = system.writer.lm_head(hidden).float().log_softmax(-1)
        S = suffix_input.shape[1]
        suffix_state = None
        if capture:
            suffix_state = PrefixState(
                list(system.raw_keys),
                [cache.value_cache[3 * depth + l][:, :, -S:] for l in range(depth)],
                cache.cos, cache.sin)
            system.raw_keys = [None] * depth
    return first, prefix_state, logp, suffix_state


def _kl(student_logp, teacher_logp, suffix, keep):
    kl = F.kl_div(student_logp, teacher_logp, reduction="none", log_target=True).sum(-1)
    kl_sum = kl[keep].sum()
    with torch.no_grad():
        target = suffix[:, 1:, None]
        ce = -student_logp.gather(-1, target)[..., 0][keep].sum()
        tce = -teacher_logp.gather(-1, target)[..., 0][keep].sum()
    return kl_sum, torch.stack((kl_sum.detach(), ce, tce, keep.sum().to(ce.dtype)))


def adapter_forward(system, samples, device):
    """Arms A/B (prefill 4) and C (prefill 1): plain 1-loop reader suffix."""
    prefix, pmask, suffix, smask = pad_batch(samples, device)
    suffix_input = suffix[:, :-1]
    full = torch.cat((pmask, smask[:, :-1]), dim=1)
    _, teacher_prefix, teacher_logp, _ = _teacher(system, prefix, pmask, suffix_input, full, False)
    if system.arm == "C":
        _, student_prefix, _ = system.writer_prefix(prefix, 1, pmask.long())
    else:
        student_prefix = teacher_prefix
    cache = system.adapter(student_prefix)
    hidden, _ = backbone(system.reader, suffix_input, cache, 1, full.long())
    student_logp = system.reader.lm_head(hidden).float().log_softmax(-1)
    keep = smask[:, 1:]  # target j+1 valid; target 0 (writer-predicted) never enters
    return _kl(student_logp, teacher_logp, suffix, keep)


def loopmap_forward(system, samples, device):
    """Arms X<p><d>: decode model (d loops, LoRA) reads only mapped prefill KV (p loops) for prompt positions < n-1,
    runs the last prompt token and the continuation itself; KL to the p4d4 teacher on every target incl. the first."""
    prefix, pmask, suffix, smask = pad_batch(samples, device)
    suffix_input = suffix[:, :-1]
    full = torch.cat((pmask, smask[:, :-1]), dim=1)
    first, prefix_state, teacher_logp, _ = _teacher(system, prefix, pmask, suffix_input, full, False)
    if system.prefill_steps != 4:
        with torch.no_grad():
            _, prefix_state, _ = system.writer_prefix(prefix, system.prefill_steps, pmask.long())
    n = prefix.shape[1] - 1
    cache = system.loop_adapter.build(prefix_state, n, system.decode_steps)
    reader_in = torch.cat((prefix[:, -1:], suffix_input), dim=1)
    if system.loop_adapter.shallow:  # arm M: gated branch over the first prefill loop's prompt KV
        with shallow_branch(cache, *first_loop_bank(prefix_state, n), system.loop_adapter):
            hidden, _ = backbone(system.reader, reader_in, cache, system.decode_steps, full.long())
    else:
        hidden, _ = backbone(system.reader, reader_in, cache, system.decode_steps, full.long())
    student_logp = system.reader.lm_head(hidden).float().log_softmax(-1)
    teacher_all = torch.cat((first.float().log_softmax(-1)[:, None], teacher_logp), dim=1)
    keep = torch.cat((smask[:, :1], smask[:, 1:]), dim=1).bool()
    kl = F.kl_div(student_logp, teacher_all, reduction="none", log_target=True).sum(-1)
    kl_sum = kl[keep].sum()
    with torch.no_grad():
        target = suffix[:, :, None]
        ce = -student_logp.gather(-1, target)[..., 0][keep].sum()
        tce = -teacher_all.gather(-1, target)[..., 0][keep].sum()
    return kl_sum, torch.stack((kl_sum.detach(), ce, tce, keep.sum().to(ce.dtype)))


def loopmap_selflive_forward(system, samples, device, faithful=False, gamma=0.1):
    """Arm O: arm S's mapped deep prompt bank + the self-live shared answer history (the author's self-live, applied to the
    decoder's own answer tokens). Pass 1 (no grad): per-loop own K/V, record the final decode loop's K/V of the answer.
    Pass 2: every decode loop reads [mapped prompt | that final-loop answer history for strictly earlier tokens | its own
    live K/V for the current token] — exactly what a decoder with one answer bank sees at inference.
    faithful=True (arm P, the original self-live recipe): the per-loop pass carries gradients and is the MAIN loss (the S
    objective); its final-loop answer K/V seeds one fixed-point iteration — a no-grad pass in the shared (self-live) view
    re-records the history from the model's own compressed pass — and the self-live pass is an AUXILIARY loss, weight gamma."""
    if system.decode_steps == 1:  # one decode loop: nothing to share on the answer side
        return loopmap_forward(system, samples, device)
    prefix, pmask, suffix, smask = pad_batch(samples, device)
    suffix_input = suffix[:, :-1]
    full = torch.cat((pmask, smask[:, :-1]), dim=1)
    first, prefix_state, teacher_logp, _ = _teacher(system, prefix, pmask, suffix_input, full, False)
    if system.prefill_steps != 4:
        with torch.no_grad():
            _, prefix_state, _ = system.writer_prefix(prefix, system.prefill_steps, pmask.long())
    B, P = prefix.shape
    n, d = P - 1, system.decode_steps
    prompt = system.loop_adapter.build(prefix_state, n, d)
    reader_in = torch.cat((prefix[:, -1:], suffix_input), dim=1)
    L = reader_in.shape[1]
    valid = torch.cat((pmask[:, -1:], smask[:, :-1]), dim=1).bool()
    i = torch.arange(L, device=device)
    pre = pmask[:, None, :-1].bool().expand(B, L, n)
    causal = (i[None, :] <= i[:, None])[None] & valid[:, None, :]
    past = (i[None, :] < i[:, None])[None] & valid[:, None, :]
    own = (i[None, :] == i[:, None])[None].expand(B, L, L)
    positions = torch.arange(n, n + L, device=device)[None].expand(B, L)
    set_steps(system.reader, d)

    def run(cache, mask):
        out, _, _ = system.reader.model(
            input_ids=reader_in, attention_mask={"full_attention": mask[:, None], "sliding_attention": mask[:, None]},
            position_ids=positions, cache_position=positions[0], past_key_values=cache, use_cache=True)
        return out.last_hidden_state

    teacher_all = torch.cat((first.float().log_softmax(-1)[:, None], teacher_logp), dim=1)
    keep = torch.cat((smask[:, :1], smask[:, 1:]), dim=1).bool()

    def kl_of(hidden):
        logp = system.reader.lm_head(hidden).float().log_softmax(-1)
        return F.kl_div(logp, teacher_all, reduction="none", log_target=True).sum(-1)[keep].sum(), logp

    c1 = LiveCache(prompt, d, training=True)
    c1.record_loop = d - 1
    if faithful:  # per-loop answer banks, with gradients: the main (S) loss
        main_kl, _ = kl_of(run(c1, torch.cat((pre, causal), dim=2)))
    else:
        with torch.no_grad():
            run(c1, torch.cat((pre, causal), dim=2))
    history = [c1.recorded[l] for l in range(c1.depth)]
    del c1
    if faithful:  # one fixed-point iteration: history from the model's own compressed (shared-bank) pass
        with torch.no_grad():
            cf = LiveCache(prompt, d, history=history, training=True)
            cf.record_loop = d - 1
            run(cf, torch.cat((pre, past, own), dim=2))
            history = [cf.recorded[l] for l in range(cf.depth)]
        del cf
    c2 = LiveCache(prompt, d, history=history, training=True)
    live_kl, student_logp = kl_of(run(c2, torch.cat((pre, past, own), dim=2)))
    kl_sum = main_kl + gamma * live_kl if faithful else live_kl
    with torch.no_grad():
        target = suffix[:, :, None]
        ce = -student_logp.gather(-1, target)[..., 0][keep].sum()
        tce = -teacher_all.gather(-1, target)[..., 0][keep].sum()
    return kl_sum, torch.stack((kl_sum.detach(), ce, tce, keep.sum().to(ce.dtype)))


def native_lora_forward(system, samples, device):
    """Arms D/E: native 2/2 or 1/1 Ouro with reader LoRA on prompt and continuation."""
    prefix, pmask, suffix, smask = pad_batch(samples, device)
    suffix_input = suffix[:, :-1]
    full = torch.cat((pmask, smask[:, :-1]), dim=1)
    _, _, teacher_logp, _ = _teacher(system, prefix, pmask, suffix_input, full, False)
    depth, steps = system.reader.config.num_hidden_layers, system.prefill_steps
    _, cache = backbone(system.reader, prefix, NativeCache(steps * depth), steps, pmask.long())
    hidden, _ = backbone(system.reader, suffix_input, cache, steps, full.long())
    student_logp = system.reader.lm_head(hidden).float().log_softmax(-1)
    return _kl(student_logp, teacher_logp, suffix, smask[:, 1:])


def delta_forward(system, samples, device):
    """Arm F: 1-loop LoRA reader runs the prompt itself; its prompt KV gets an
    additive delta mapped from the 4-loop writer's last-loop KV."""
    prefix, pmask, suffix, smask = pad_batch(samples, device)
    suffix_input = suffix[:, :-1]
    full = torch.cat((pmask, smask[:, :-1]), dim=1)
    _, prefix_state, teacher_logp, _ = _teacher(system, prefix, pmask, suffix_input, full, False)
    if system.prefill_steps != 4:  # shallower prefill model (arms J/K): its own last-loop KV; teacher stays p4d4
        with torch.no_grad():
            _, prefix_state, _ = system.writer_prefix(prefix, system.prefill_steps, pmask.long())
    _, cache = system.delta_prefix(prefix, prefix_state, pmask.long())
    hidden, _ = backbone(system.reader, suffix_input, cache, system.decode_steps, full.long())
    student_logp = system.reader.lm_head(hidden).float().log_softmax(-1)
    return _kl(student_logp, teacher_logp, suffix, smask[:, 1:])


def refresh_forward(system, samples, blocks, device):
    prefix, pmask, suffix, smask = pad_batch(samples, device)
    suffix_input = suffix[:, :-1]
    B, P = prefix.shape
    S = suffix_input.shape[1]
    depth = system.writer.config.num_hidden_layers
    full = torch.cat((pmask, smask[:, :-1]), dim=1)
    _, prefix_state, teacher_logp, suffix_state = _teacher(system, prefix, pmask, suffix_input, full, True)
    mp, ms = system.adapter(prefix_state), system.adapter(suffix_state)
    cache = NativeCache(depth)
    for l in range(depth):
        cache.key_cache.append(torch.cat((mp.key_cache[l], ms.key_cache[l]), dim=2))
        cache.value_cache.append(torch.cat((mp.value_cache[l], ms.value_cache[l]), dim=2))
    cache._seen_tokens = P + S
    i = torch.arange(S, device=device)
    block = torch.tensor(blocks, device=device)[:, None]            # [B,1]
    bi = i[None] // block                                           # [B,S]
    mapped = bi[:, None, :] < bi[:, :, None]                        # [B,S,S]
    own = (bi[:, None, :] == bi[:, :, None]) & (i[None, None, :] <= i[None, :, None])
    pre = pmask[:, None, :].expand(B, S, P)
    mask = torch.cat((pre, mapped, own), dim=2)[:, None]
    positions = torch.arange(P, P + S, device=device)
    system.reader.config.total_ut_steps = 1
    system.reader.model.total_ut_steps = 1
    out, _, _ = system.reader.model(
        input_ids=suffix_input,
        attention_mask={"full_attention": mask, "sliding_attention": mask},
        position_ids=positions[None].expand(B, S), cache_position=positions + S,
        past_key_values=cache, use_cache=True)
    student_logp = system.reader.lm_head(out.last_hidden_state).float().log_softmax(-1)
    keep = smask[:, 1:] & ((i[None] % block) != (block - 1))
    return _kl(student_logp, teacher_logp, suffix, keep)


class HistoryCache(NativeCache):
    """Read-only per-layer history (prompt + continuation, teacher last loop); every decode loop reads the same bank and
    the current token's own K/V are appended per call, never stored."""

    def __init__(self, keys, values):
        super().__init__(None)
        self.key_cache, self.value_cache, self.depth = keys, values, len(keys)
        self._seen_tokens = keys[0].shape[2]

        self.record_loop, self.recorded = None, {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        loop, layer = divmod(layer_idx, self.depth)
        if loop == self.record_loop:  # keep the final decode loop's own K/V (self-history for a second pass)
            self.recorded[layer] = (key_states.detach(), value_states.detach())
        return (torch.cat((self.key_cache[layer], key_states), dim=2),
                torch.cat((self.value_cache[layer], value_states), dim=2))


def kvshare_forward(system, samples, device):
    """Arms Z<d>: trained KV sharing (BeNice-style self-live). Prompt and earlier continuation tokens are read from the
    native teacher's last-loop KV; each continuation token reads its own current-loop K/V at every decode loop, exactly
    as at inference with one shared bank. The LoRA decode model runs d loops; KL to the p4d4 teacher."""
    prefix, pmask, suffix, smask = pad_batch(samples, device)
    suffix_input = suffix[:, :-1]
    B, P = prefix.shape
    S = suffix_input.shape[1]
    depth = system.writer.config.num_hidden_layers
    full = torch.cat((pmask, smask[:, :-1]), dim=1)
    with torch.no_grad():
        _, _, cache = system.writer_prefix(prefix, 4, pmask.long())
        hidden, cache = backbone(system.writer, suffix_input, cache, 4, full.long())
        teacher_logp = system.writer.lm_head(hidden).float().log_softmax(-1)
        history = HistoryCache([cache.key_cache[3 * depth + l] for l in range(depth)],
                               [cache.value_cache[3 * depth + l] for l in range(depth)])
    del cache, hidden
    i = torch.arange(S, device=device)
    pre = pmask[:, None, :].expand(B, S, P)
    past = (i[None, :] < i[:, None])[None] & smask[:, None, :-1]
    own = (i[None, :] == i[:, None])[None].expand(B, S, S)
    mask = torch.cat((pre, past, own), dim=2)[:, None]
    positions = torch.arange(P, P + S, device=device)
    system.reader.config.total_ut_steps = system.decode_steps
    system.reader.model.total_ut_steps = system.decode_steps
    out, _, _ = system.reader.model(
        input_ids=suffix_input,
        attention_mask={"full_attention": mask, "sliding_attention": mask},
        position_ids=positions[None].expand(B, S), cache_position=positions + S,
        past_key_values=history, use_cache=True)
    student_logp = system.reader.lm_head(out.last_hidden_state).float().log_softmax(-1)
    return _kl(student_logp, teacher_logp, suffix, smask[:, 1:])


def kvshare_self_forward(system, samples, device):
    """Arms Y<d>: KV sharing trained like the loop-map arms. The decode model also predicts the first output token (it
    runs the last prompt token itself), and continuation history is the decode model's OWN final-loop K/V from a first
    no-grad pass (one fixed-point iteration) instead of the teacher's, so train-time history matches inference better.
    Prompt history (all but the last prompt token) is the native p4 prefill's last-loop KV, as at inference."""
    prefix, pmask, suffix, smask = pad_batch(samples, device)
    suffix_input = suffix[:, :-1]
    B, P = prefix.shape
    depth, d = system.writer.config.num_hidden_layers, system.decode_steps
    full = torch.cat((pmask, smask[:, :-1]), dim=1)
    with torch.no_grad():
        first, _, cache = system.writer_prefix(prefix, 4, pmask.long())
        hidden, cache = backbone(system.writer, suffix_input, cache, 4, full.long())
        teacher_all = torch.cat((first.float().log_softmax(-1)[:, None],
                                 system.writer.lm_head(hidden).float().log_softmax(-1)), dim=1)
        bank_k = [cache.key_cache[3 * depth + l] for l in range(depth)]
        bank_v = [cache.value_cache[3 * depth + l] for l in range(depth)]
    del cache, hidden
    reader_in = torch.cat((prefix[:, -1:], suffix_input), dim=1)        # positions P-1 .. P+S-2
    L = reader_in.shape[1]
    valid = torch.cat((pmask[:, -1:], smask[:, :-1]), dim=1)
    i = torch.arange(L, device=device)
    pre = pmask[:, None, :-1].expand(B, L, P - 1)
    past = (i[None, :] < i[:, None])[None] & valid[:, None, :]
    own = (i[None, :] == i[:, None])[None].expand(B, L, L)
    mask = torch.cat((pre, past, own), dim=2)[:, None]
    positions = torch.arange(P - 1, P - 1 + L, device=device)
    system.reader.config.total_ut_steps = d
    system.reader.model.total_ut_steps = d

    def run(history):
        out, _, _ = system.reader.model(
            input_ids=reader_in, attention_mask={"full_attention": mask, "sliding_attention": mask},
            position_ids=positions[None].expand(B, L), cache_position=positions + L,
            past_key_values=history, use_cache=True)
        return out.last_hidden_state

    with torch.no_grad():  # pass 1: history = teacher last loop; record the student's own final-loop K/V
        h1 = HistoryCache([k[:, :, : P - 1 + L] for k in bank_k], [v[:, :, : P - 1 + L] for v in bank_v])
        h1.record_loop = d - 1
        run(h1)
        own_kv = h1.recorded
    del h1
    h2 = HistoryCache([torch.cat((bank_k[l][:, :, : P - 1], own_kv[l][0]), dim=2) for l in range(depth)],
                      [torch.cat((bank_v[l][:, :, : P - 1], own_kv[l][1]), dim=2) for l in range(depth)])
    student_logp = system.reader.lm_head(run(h2)).float().log_softmax(-1)
    keep = smask
    kl = F.kl_div(student_logp, teacher_all, reduction="none", log_target=True).sum(-1)
    kl_sum = kl[keep].sum()
    with torch.no_grad():
        target = suffix[:, :, None]
        ce = -student_logp.gather(-1, target)[..., 0][keep].sum()
        tce = -teacher_all.gather(-1, target)[..., 0][keep].sum()
    return kl_sum, torch.stack((kl_sum.detach(), ce, tce, keep.sum().to(ce.dtype)))
