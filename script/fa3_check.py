"""Compare FA3 with PyTorch's built-in FA2 behind varlen_attn on one real packed batch, on 1 GPU.

A. The attention op alone: error against an fp32 per-document reference, and fwd+bwd time, for
   FA2 (varlen_attn), FA3-A (varlen_attn after activate_flash_attention_impl("FA3")) and
   FA3-B (flash_attn_interface.flash_attn_varlen_func called directly).
B. GPT-2 small as in src/train_gpt2.py, compiled, without DDP or the optimizer: fwd+loss+bwd time
   and the loss and gradients of the same batch, FA2 vs FA3-A vs FA3-B.

Run from the project root through sbatch, never on the login node.
"""
import sys
import traceback

import torch
import torch.nn.functional as F
from flash_attn_interface import flash_attn_varlen_func
from torch.nn.attention import (
    activate_flash_attention_impl, current_flash_attention_impl, restore_flash_attention_impl,
)
from torch.nn.attention.varlen import varlen_attn
from transformers import AttentionInterface, GPT2Config, GPT2LMHeadModel

sys.path.insert(0, "src")
from data_load import get_batch, load_rows

SEQ_LEN = 1024
ROWS = 64        # MICRO_BATCH_SIZE in train_gpt2.py
HEADS, HEAD_DIM = 12, 64
ITERS = 50

device = "cuda"
torch.manual_seed(0)
torch.set_float32_matmul_precision("high")

input_ids, position_ids, labels, cu_seqlens = get_batch(load_rows("data/c4", "train", SEQ_LEN), 0, ROWS, device)
lens = cu_seqlens.diff().float()
print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}")
print(f"{len(lens)} segments in {ROWS} rows; mean length {lens.mean().item():.0f}, "
      f"token-weighted mean length {((lens ** 2).sum() / lens.sum()).item():.0f}")


def bench(fn):
    """Mean ms per call over ITERS calls, after 10 warmup calls."""
    for _ in range(10):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


# A. The attention op alone ---------------------------------------------------
n = ROWS * SEQ_LEN
q, k, v = (torch.randn(n, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
dout = torch.randn(n, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
scale = HEAD_DIM ** -0.5


def fwd_bwd(attn):
    """(out, dq, dk, dv) of attn() for the shared q, k, v and dout."""
    for t in (q, k, v):
        t.grad = None
    out = attn()
    out.backward(dout)
    return out, q.grad, k.grad, v.grad


def reference():
    """fp32 causal attention run on each segment by itself."""
    qf, kf, vf = (t.detach().float().requires_grad_() for t in (q, k, v))
    bounds = cu_seqlens.tolist()
    out = torch.cat([
        F.scaled_dot_product_attention(*(t[s:e].transpose(0, 1) for t in (qf, kf, vf)), is_causal=True,
                                       scale=scale).transpose(0, 1)
        for s, e in zip(bounds[:-1], bounds[1:])
    ])
    out.backward(dout.float())
    return out, qf.grad, kf.grad, vf.grad


def varlen():  # FA2, or FA3 once activated
    return varlen_attn(q, k, v, cu_seqlens, cu_seqlens, SEQ_LEN, SEQ_LEN, scale=scale, window_size=(-1, 0))


def fa3_direct():
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, SEQ_LEN, SEQ_LEN, softmax_scale=scale, causal=True)


ref = reference()


def check_op(label, attn):
    """Print max abs / relative error against ref for out, dq, dk, dv, and the fwd+bwd time."""
    errors = "  ".join(
        f"{name} {(got.float() - want).abs().max().item():.1e}/{((got.float() - want).norm() / want.norm()).item():.1e}"
        for name, got, want in zip(("out", "dq", "dk", "dv"), fwd_bwd(attn), ref)
    )
    ms = bench(lambda: fwd_bwd(attn))
    print(f"{label:6s} fwd+bwd {ms:7.3f} ms   error vs fp32 (max abs/rel): {errors}")
    return ms


print("\nA. attention op alone, one layer (the model runs 12 per micro-batch)")
ms_fa2 = check_op("FA2", varlen)
ms_fa3a = None
try:
    activate_flash_attention_impl("FA3")
    print(f"active impl: {current_flash_attention_impl()}")
    ms_fa3a = check_op("FA3-A", varlen)
except Exception:
    traceback.print_exc()
restore_flash_attention_impl()
ms_fa3b = check_op("FA3-B", fa3_direct)
fa3a = f"{ms_fa2 / ms_fa3a:.3f}x" if ms_fa3a else "failed"
print(f"speedup vs FA2: FA3-A {fa3a}, FA3-B {ms_fa2 / ms_fa3b:.3f}x")


# B. GPT-2 small as in train_gpt2.py ------------------------------------------
DIRECT_FA3 = False  # True: flash_attn_varlen_func (FA3-B); False: varlen_attn (FA2, or FA3-A once activated)


def varlen_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                     cu_seq_lens_q=None, max_length_q=None, **kwargs):
    """The same function as in train_gpt2.py, plus the FA3-B switch."""
    b, h, l, d = query.shape
    q, k, v = (t.transpose(1, 2).reshape(b * l, h, d) for t in (query, key, value))
    if DIRECT_FA3:
        # Dynamo can't trace an autograd.Function given the same tensor twice, hence the clone.
        out = flash_attn_varlen_func(q, k, v, cu_seq_lens_q, cu_seq_lens_q.clone(), max_length_q, max_length_q,
                                     softmax_scale=scaling, causal=True)
    else:
        out = varlen_attn(q, k, v, cu_seq_lens_q, cu_seq_lens_q, max_length_q, max_length_q,
                          scale=scaling, window_size=(-1, 0))
    return out.view(b, l, h, d), None


AttentionInterface.register("varlen", varlen_attention)
config = GPT2Config(
    vocab_size=50304, n_positions=SEQ_LEN, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    use_cache=False, attn_implementation="varlen",
)
with torch.device(device):
    raw_model = GPT2LMHeadModel(config)
raw_model.loss_type = "ForCausalLM"
model = torch.compile(raw_model, fullgraph=True)  # recompiles once when DIRECT_FA3 flips
torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)


def use(label):
    """Switch the attention to FA2, FA3-A or FA3-B."""
    global DIRECT_FA3
    if current_flash_attention_impl():
        restore_flash_attention_impl()
    if label == "FA3-A":
        activate_flash_attention_impl("FA3")
    DIRECT_FA3 = label == "FA3-B"


def train_micro_step():
    """One micro-batch of training without the optimizer: fwd, loss and bwd."""
    raw_model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(
            input_ids=input_ids, position_ids=position_ids, labels=labels,
            cu_seq_lens_q=cu_seqlens, max_length_q=SEQ_LEN,
        ).loss
    loss.backward()
    return loss


def loss_and_grads():
    loss = train_micro_step()
    return loss.item(), torch.cat([p.grad.float().flatten() for p in raw_model.parameters()])


labels_ab = ("FA2", "FA3-A", "FA3-B")
print("\nB. GPT-2 small, compiled, fwd+loss+bwd on one micro-batch (no DDP, no optimizer)")
use("FA2")
loss_fa2, grads_fa2 = loss_and_grads()  # compiles
for label in labels_ab[1:]:
    use(label)
    loss, grads = loss_and_grads()
    print(f"{label:6s} loss {loss:.6f} vs FA2 {loss_fa2:.6f}, diff {abs(loss - loss_fa2):.1e}; "
          f"grad relative diff {((grads - grads_fa2).norm() / grads_fa2.norm()).item():.1e}")
    del grads
del grads_fa2

# The three alternate twice, so a drift in GPU clocks doesn't count as a speedup.
times = {label: [] for label in labels_ab}
for _ in range(2):
    for label in labels_ab:
        use(label)
        ms = bench(train_micro_step)
        times[label].append(ms)
        print(f"{label:6s} {ms:7.2f} ms per micro-batch, {n / ms * 1e3:,.0f} tokens/s on 1 GPU")
mean = {label: sum(t) / len(t) for label, t in times.items()}
for label in labels_ab[1:]:
    print(f"{label} vs FA2: {mean['FA2'] / mean[label]:.3f}x per micro-batch "
          f"({(mean['FA2'] / mean[label] - 1) * 100:+.1f}%)")
