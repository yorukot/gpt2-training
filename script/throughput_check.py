"""Time training-step variants on 2 GPUs with the model, optimizers and data of src/train_gpt2.py,
to pick the loss path, the DDP graph setting and the micro-batch size.

V0  HF labels= path: GPT2LMHeadModel also returns the logits, so autograd builds a zero grad for them
V1  LMLoss: GPT-2 + next-token CE returning only the loss
V2  V1 with torch._dynamo.config.optimize_ddp = False (one graph instead of DDPOptimizer's pieces)
V3  V1 with 128x1 micro-batches instead of 64x2 (the same 256 rows per step)
V4  V2 with 128x1 micro-batches

Run from the project root through sbatch with torchrun --nproc_per_node=2, never on the login node.
"""
import gc
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from flash_attn_interface import flash_attn_varlen_func
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AttentionInterface, GPT2Config, GPT2LMHeadModel

sys.path.insert(0, "src")
from data_load import load_rows, train_batches
from muon import Muon

SEQ_LEN = 1024
ROWS_PER_STEP = 128  # per GPU, MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS in train_gpt2.py
WARMUP_STEPS, TIMED_STEPS = 5, 30

# The optimizer settings of train_gpt2.py, held at their peak values.
EMBED_LR, EMBED_WEIGHT_DECAY, SCALAR_LR = 3e-3, 0.1, 3e-3
BETAS, ADAM_EPS = (0.9, 0.98), 1e-10
MUON_LR, MUON_WEIGHT_DECAY, MUON_MOMENTUM = 0.02, 0.05, 0.95
GRAD_CLIP = 1.0

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
device = f"cuda:{os.environ['LOCAL_RANK']}"
torch.cuda.set_device(device)
torch.set_float32_matmul_precision("high")
train_rows = load_rows("data/c4", "train", SEQ_LEN)
if rank == 0:
    print(f"{world}x {torch.cuda.get_device_name()}, torch {torch.__version__}")


def varlen_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                     cu_seq_lens_q=None, max_length_q=None, **kwargs):
    """The same function as in train_gpt2.py."""
    b, h, l, d = query.shape
    q, k, v = (t.transpose(1, 2).reshape(b * l, h, d) for t in (query, key, value))
    out = flash_attn_varlen_func(q, k, v, cu_seq_lens_q, cu_seq_lens_q.clone(), max_length_q, max_length_q,
                                 softmax_scale=scaling, causal=True)
    return out.view(b, l, h, d), None


AttentionInterface.register("varlen", varlen_attention)
config = GPT2Config(
    vocab_size=50304, n_positions=SEQ_LEN, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    use_cache=False, attn_implementation="varlen",
)


class LMLoss(nn.Module):
    """GPT-2 + next-token CE that returns only the loss, so autograd never builds a zero grad for the logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, targets, cu_seqlens, max_len):
        h = self.model.transformer(input_ids=input_ids, position_ids=position_ids,
                                   cu_seq_lens_q=cu_seqlens, max_length_q=max_len).last_hidden_state
        logits = F.linear(h, self.model.transformer.wte.weight)  # the tied lm_head
        return F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))


def shift(labels):
    """HF ForCausalLMLoss's targets: labels moved left by one, with -100 at the end of each row."""
    return F.pad(labels[:, 1:], (0, 1), value=-100)


def make_optimizers(raw_model):
    """AdamW and Muon split the parameters as in train_gpt2.py."""
    blocks = [(b.attn.c_attn.weight, b.attn.c_proj.weight, b.mlp.c_fc.weight, b.mlp.c_proj.weight)
              for b in raw_model.transformer.h]
    muon_ids = {id(p) for block in blocks for p in block}
    rest = [p for p in raw_model.parameters() if id(p) not in muon_ids]
    embeds, scalars = [p for p in rest if p.dim() >= 2], [p for p in rest if p.dim() < 2]
    adamw = torch.optim.AdamW(
        [
            {"params": embeds, "lr": EMBED_LR, "weight_decay": EMBED_WEIGHT_DECAY},
            {"params": scalars, "lr": SCALAR_LR, "weight_decay": 0.0},
        ],
        betas=BETAS, eps=ADAM_EPS, fused=True,
    )
    return [adamw, Muon(blocks, lr=MUON_LR, weight_decay=MUON_WEIGHT_DECAY, momentum=MUON_MOMENTUM)]


def run(label, use_hf, optimize_ddp, micro_rows):
    """Print and return (ms per step, loss of the first micro-batch at the initial weights)."""
    torch._dynamo.reset()
    torch._dynamo.config.optimize_ddp = optimize_ddp
    torch.manual_seed(42)  # the same initial weights in every variant
    with torch.device(device):
        raw_model = GPT2LMHeadModel(config)
    raw_model.loss_type = "ForCausalLM"
    model = DDP(torch.compile(raw_model if use_hf else LMLoss(raw_model), fullgraph=True), device_ids=[device])
    optimizers = make_optimizers(raw_model)
    accum = ROWS_PER_STEP // micro_rows
    batches = train_batches(train_rows, micro_rows, rank, world, device)

    def train_step():
        losses = []
        for micro in range(accum):
            input_ids, position_ids, labels, cu_seqlens = next(batches)
            torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)
            model.require_backward_grad_sync = micro == accum - 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_hf:
                    loss = model(input_ids=input_ids, position_ids=position_ids, labels=labels,
                                 cu_seq_lens_q=cu_seqlens, max_length_q=SEQ_LEN).loss
                else:
                    loss = model(input_ids, position_ids, shift(labels), cu_seqlens, SEQ_LEN)
            (loss / accum).backward()
            losses.append(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        return losses

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    first_loss = train_step()[0].item()  # compiles
    first_step = time.perf_counter() - start
    for _ in range(WARMUP_STEPS):
        train_step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        train_step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / TIMED_STEPS * 1e3
    peak = torch.cuda.max_memory_allocated() / 2**30
    if rank == 0:
        tokens_per_sec = ROWS_PER_STEP * world * SEQ_LEN / ms * 1e3
        print(f"{label}: first step {first_step:5.1f} s, {ms:6.1f} ms/step, {tokens_per_sec:,.0f} tok/s, "
              f"peak {peak:5.1f} GiB, first loss {first_loss:.6f}", flush=True)

    del model, optimizers, raw_model, batches
    gc.collect()
    torch.cuda.empty_cache()
    return ms, first_loss


VARIANTS = [  # (label, use_hf, optimize_ddp, micro_rows)
    ("V0 HF labels, optimize_ddp on, 64x2", True, True, 64),
    ("V1 LMLoss, optimize_ddp on, 64x2", False, True, 64),
    ("V2 LMLoss, optimize_ddp off, 64x2", False, False, 64),
    ("V3 LMLoss, optimize_ddp on, 128x1", False, True, 128),
    ("V4 LMLoss, optimize_ddp off, 128x1", False, False, 128),
]
results = {variant[0]: run(*variant) for variant in VARIANTS}
if rank == 0:
    base_ms, base_loss = results[VARIANTS[0][0]]
    print("\nvs V0 (step time; first-loss diff only where the first micro-batch is the same 64 rows):")
    for label, (ms, loss) in results.items():
        diff = f", first loss diff {abs(loss - base_loss):.1e}" if label.endswith("64x2") else ""
        print(f"  {label}: {base_ms / ms:.3f}x ({(base_ms / ms - 1) * 100:+.1f}%){diff}")
dist.destroy_process_group()
