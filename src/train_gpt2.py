import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from flash_attn_interface import flash_attn_varlen_func
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AttentionInterface, GPT2Config, GPT2LMHeadModel

from data_load import load_rows, load_teacher, teacher_batches, train_batches
from muon import Muon

# Hyperparameters ------------------------
DATA_DIR = "data/c4"
SEQ_LEN = 1024                 # rows packed to this by prepare_data.py
VOCAB_SIZE = 50304             # GPT-2's 50257 padded to a multiple of 64 for faster matmuls;
                               # resized back to 50257 before saving
MICRO_BATCH_SIZE = 64          # packed rows per GPU per forward pass
GRAD_ACCUM_STEPS = 2           # 64 * 2 * 2 GPUs = 256 rows per step
TRAIN_SECONDS = 29 * 60 + 20   # stop training this long after the script starts; the rest of the
                               # 30-min Slurm limit covers torchrun startup (~17 s), dumping the raw
                               # weights and wandb.finish(). script/finalize.py converts and evaluates later.

EMBED_LR = 3e-3               # AdamW on the tied wte/lm_head and wpe
EMBED_WEIGHT_DECAY = 0.1
SCALAR_LR = 3e-3               # AdamW on the LayerNorms and biases, no weight decay
WARMUP_STEPS = 300             # AdamW only; Muon starts at its full LR
BETAS = (0.9, 0.98)
ADAM_EPS = 1e-10

MUON_LR = 0.01                 # Muon on the 2D weights in the blocks
MUON_WEIGHT_DECAY = 0.05       # decoupled: weights shrink by MUON_LR * MUON_WEIGHT_DECAY per step at peak
MUON_MOMENTUM = 0.95
MUON_MOMENTUM_START = 0.85     # raised linearly to MUON_MOMENTUM over MUON_MOMENTUM_WARMUP steps
MUON_MOMENTUM_WARMUP = 300

MIN_LR_RATIO = 0.0             # final LR of every group, as a fraction of its peak LR
DECAY_START = 0.5              # fraction of the training time at full LR before the decay to MIN_LR_RATIO
                               # (0.0 = decay right away, 0.5 = trapezoid)
DECAY_SHAPE = "linear"         # "linear", "cosine" or "1-sqrt" (LR falls fast first, then slowly)
GRAD_CLIP = 1.0
DROPOUT = 0.0

TEACHER_DIR = "/work/u9603854/gpt2/teacher/gpt-j-6b"  # GPT-J-6B's top-8 per position (script/teacher_logits.py)
KD_ROWS = (0, 1_861_632)       # (first, end): train rows distilled from the teacher, the rest plain CE;
                               # None = no distillation. (0, 1_861_632) = teacher chunks 0-1817, the
                               # whole run (~6,500 steps x 256 rows at ~986k tok/s)
KD_CE_WEIGHT = 0.6             # weight of the hard-label CE inside KD_ROWS; the KD loss gets the rest

LOG_EVERY = 10
SEED = 42
# ------------------------------------

START = time.time()
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
device = f"cuda:{os.environ['LOCAL_RANK']}"
torch.cuda.set_device(device)
torch.manual_seed(SEED)
torch.set_float32_matmul_precision("high")
run_name = os.environ.get("RUN_NAME", "gpt2-local")  # Set by train.sbatch.

tokens_per_step = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS * world  # including padding

train_rows = load_rows(DATA_DIR, "train", SEQ_LEN)


def lr_decay(progress):
    """LR multiplier: 1 until DECAY_START, then a DECAY_SHAPE decay to MIN_LR_RATIO.

    progress is the fraction of the training time done, from step 0 (after startup) to TRAIN_SECONDS.
    """
    t = min(1.0, max(0.0, (progress - DECAY_START) / (1 - DECAY_START)))  # how far into the decay
    decay = {"linear": t, "cosine": 0.5 * (1 - math.cos(math.pi * t)), "1-sqrt": math.sqrt(t)}[DECAY_SHAPE]
    return 1 - (1 - MIN_LR_RATIO) * decay


def elapsed_seconds():
    """Seconds since START on rank 0, shared so every rank sets the same lr and stops together."""
    t = torch.tensor(time.time() - START, device=device)
    dist.broadcast(t, 0)
    return t.item()


def varlen_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                     cu_seq_lens_q=None, max_length_q=None, **kwargs):
    """Causal attention inside each document of the packed rows, split at cu_seq_lens_q.

    Same math as sdpa with HF's document mask, but no (rows, 1, SEQ_LEN, SEQ_LEN) mask is built
    and no work is spent on blocks across documents. Uses the FlashAttention 3 (Hopper) kernels,
    ~5.7% faster per micro-batch than PyTorch's built-in FA2 (script/fa3_check.py).
    Training only: nothing of it is saved.
    """
    b, h, l, d = query.shape  # HF passes (batch, heads, seq, head_dim); varlen wants (tokens, heads, head_dim)
    q, k, v = (t.transpose(1, 2).reshape(b * l, h, d) for t in (query, key, value))
    # Dynamo can't trace an autograd.Function given the same tensor twice, hence the clone.
    out = flash_attn_varlen_func(q, k, v, cu_seq_lens_q, cu_seq_lens_q.clone(), max_length_q, max_length_q,
                                 softmax_scale=scaling, causal=True)
    return out.view(b, l, h, d), None


# A name HF has no mask function for, so it skips building the 4D mask.
AttentionInterface.register("varlen", varlen_attention)


class LMLoss(nn.Module):
    """GPT-2 plus the next-token loss, returning (loss, CE, KD) and no logits.

    GPT2LMHeadModel also returns the logits, so the compiled backward builds a (rows, SEQ_LEN, vocab)
    zero grad for them every micro-step; returning only the loss is ~2% faster per step and 12 GiB
    lighter, with the same loss (script/throughput_check.py). The saved weights are raw_model's.

    loss = (1 - kd_weight) * CE + kd_weight * KD, both means over the scored tokens. KD is the cross
    entropy against the teacher's top k (t_idx, t_logp: full-vocab log-probs) plus one bucket for the
    rest of its probability, so the student keeps the teacher's mass outside the top k instead of
    pushing it to 0. kd_weight is a tensor, so switching distillation on and off never recompiles;
    without a teacher the batch gets zeros for t_idx/t_logp and kd_weight 0.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, targets, cu_seqlens, max_len, t_idx, t_logp, kd_weight):
        h = self.model.transformer(input_ids=input_ids, position_ids=position_ids,
                                   cu_seq_lens_q=cu_seqlens, max_length_q=max_len).last_hidden_state
        # The tied lm_head, bf16 under autocast. Upcasting only inside the logsumexp and the gathers keeps
        # a fp32 copy of the (rows, SEQ_LEN, vocab) logits out of memory: 986k vs 916k tok/s with the
        # same loss (.cache/kd_bench, K2 vs K1).
        logits = F.linear(h, self.model.transformer.wte.weight)
        scored = targets != -100
        lse = logits.float().logsumexp(-1)
        ce = lse - logits.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1).float()
        s_top = logits.gather(-1, t_idx).float() - lse.unsqueeze(-1)  # the student's log-probs of the teacher's top k
        p_top = t_logp.exp()
        p_rest = (1 - p_top.sum(-1)).clamp_min(0)
        s_rest = torch.log1p(-s_top.exp().sum(-1).clamp(max=1 - 1e-6))
        kd = -(p_top * s_top).sum(-1) - p_rest * s_rest
        count = scored.sum()
        ce = torch.where(scored, ce, 0).sum() / count
        kd = torch.where(scored, kd, 0).sum() / count
        return (1 - kd_weight) * ce + kd_weight * kd, ce.detach(), kd.detach()


def shift(labels):
    """HF ForCausalLMLoss's targets: labels moved left by one, with -100 at the end of each row."""
    return F.pad(labels[:, 1:], (0, 1), value=-100)


# GPT2Config defaults are GPT-2 small (12 layers, 12 heads, 768 dims) with GPT-1 style
# N(0, 0.02) init. Only the vocab padding differs from stock, and it is undone before saving.
config = GPT2Config(
    vocab_size=VOCAB_SIZE, n_positions=SEQ_LEN, resid_pdrop=DROPOUT, embd_pdrop=DROPOUT, attn_pdrop=DROPOUT,
    use_cache=False, attn_implementation="varlen",
)
with torch.device(device):
    raw_model = GPT2LMHeadModel(config)
raw_model.loss_type = "ForCausalLM"  # unset, HF logs a warning inside forward
# One compiled graph instead of DDPOptimizer's ~14 pieces: ~2% faster per step on 2 GPUs
# (script/throughput_check.py). The grad all-reduce then waits for the whole backward, which costs less on NVLink.
torch._dynamo.config.optimize_ddp = False
model = DDP(torch.compile(LMLoss(raw_model), fullgraph=True), device_ids=[device])

# Muon on the 2D weights in the blocks; AdamW on the embeddings (wte/lm_head and wpe), with weight
# decay, and on the biases and LayerNorms, without. Optimizer only, so the saved weights are stock.
blocks = [(b.attn.c_attn.weight, b.attn.c_proj.weight, b.mlp.c_fc.weight, b.mlp.c_proj.weight)
          for b in raw_model.transformer.h]
muon_ids = {id(p) for block in blocks for p in block}
rest = [p for p in raw_model.parameters() if id(p) not in muon_ids]  # Tied wte/lm_head is listed once.
embeds, scalars = [p for p in rest if p.dim() >= 2], [p for p in rest if p.dim() < 2]
assert len(embeds) == 2, "only wte and wpe should be left for AdamW among the matrices"
adamw = torch.optim.AdamW(
    [
        {"params": embeds, "lr": EMBED_LR, "base_lr": EMBED_LR, "weight_decay": EMBED_WEIGHT_DECAY},
        {"params": scalars, "lr": SCALAR_LR, "base_lr": SCALAR_LR, "weight_decay": 0.0},
    ],
    betas=BETAS, eps=ADAM_EPS, fused=True,
)
muon = Muon(blocks, lr=MUON_LR, weight_decay=MUON_WEIGHT_DECAY, momentum=MUON_MOMENTUM_START)
optimizers = [adamw, muon]

if rank == 0:
    # The constants only: `F` (torch.nn.functional) is upper case too but is a module.
    hyperparams = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple))}
    wandb.init(entity="cerulean-labs", project="gpt2-training", name=run_name, config=hyperparams)

max_steps = len(train_rows[0]) // (MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS * world)  # until the data runs out
if rank == 0:
    print(f"Training for {TRAIN_SECONDS}s (at most {max_steps} steps), {tokens_per_step:,} tokens per step including padding")

teacher, top_k = None, 1  # without distillation the zero teacher tensors only need some shape
if KD_ROWS is not None:
    *teacher, done_chunks, chunk_rows = load_teacher(TEACHER_DIR, SEQ_LEN)
    top_k = teacher[0].shape[-1]
    missing = [c for c in range(KD_ROWS[0] // chunk_rows, -(-KD_ROWS[1] // chunk_rows)) if c not in done_chunks]
    assert not missing, f"KD_ROWS {KD_ROWS} needs teacher chunks that are not done: {missing[:10]}"
no_teacher = (torch.zeros(MICRO_BATCH_SIZE, SEQ_LEN, top_k, dtype=torch.long, device=device),
              torch.zeros(MICRO_BATCH_SIZE, SEQ_LEN, top_k, device=device))
kd_on, kd_off = (torch.tensor(w, device=device) for w in (1 - KD_CE_WEIGHT, 0.0))
alignment_checked = False

batches = train_batches(train_rows, MICRO_BATCH_SIZE, rank, world, device)
teachers = teacher_batches(teacher, MICRO_BATCH_SIZE, rank, world, device, KD_ROWS)
last_log = time.time()
last_check = None  # elapsed at the previous check, for the step time
step = 0
for step in range(max_steps):
    if step % LOG_EVERY == 0:  # checked every LOG_EVERY steps, since the broadcast waits for the GPU
        elapsed = elapsed_seconds()
        # Stop if the next LOG_EVERY steps would end past TRAIN_SECONDS, so training never overruns it.
        step_seconds = (elapsed - last_check) / LOG_EVERY if last_check is not None else 0.0
        if elapsed + LOG_EVERY * step_seconds >= TRAIN_SECONDS:
            break
        last_check = elapsed
        if step == 0:
            train_start = elapsed  # the schedule runs from here, after startup, to TRAIN_SECONDS
        decay = lr_decay((elapsed - train_start) / (TRAIN_SECONDS - train_start))
        warmup = min(1.0, (step + 1) / WARMUP_STEPS)
        for group in adamw.param_groups:
            group["lr"] = group["base_lr"] * warmup * decay
        muon.param_groups[0]["lr"] = MUON_LR * decay
        muon.param_groups[0]["momentum"] = MUON_MOMENTUM_START + (
            MUON_MOMENTUM - MUON_MOMENTUM_START) * min(1.0, step / MUON_MOMENTUM_WARMUP)

    for micro in range(GRAD_ACCUM_STEPS):
        input_ids, position_ids, labels, cu_seqlens = next(batches)
        teacher_batch = next(teachers)
        torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)  # the document count varies; compile once for all
        targets = shift(labels)
        if teacher_batch is not None and not alignment_checked:
            alignment_checked = True
            if rank == 0:  # an off-by-one between the teacher and the targets would show far less
                hit = (teacher_batch[0] == targets.unsqueeze(-1)).any(-1)[targets != -100].float().mean().item()
                print(f"Teacher alignment at step {step}: target in the teacher's top-{top_k} for {hit:.3f} "
                      f"of the scored tokens (~0.76 expected)")
        t_idx, t_logp = teacher_batch or no_teacher
        kd_weight = kd_on if teacher_batch is not None else kd_off
        model.require_backward_grad_sync = micro == GRAD_ACCUM_STEPS - 1  # sync grads on the last one
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ce, kd = model(input_ids, position_ids, targets, cu_seqlens, SEQ_LEN, t_idx, t_logp, kd_weight)
        (loss / GRAD_ACCUM_STEPS).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)

    if rank == 0 and step % LOG_EVERY == 0:
        now = time.time()
        wandb.log({
            "train/loss": loss.item(),
            "train/ce": ce.item(),
            # Without a teacher kd is computed against the zero tensors, so only log it with one.
            **({"train/kd": kd.item()} if teacher_batch is not None else {}),
            "train/kd_weight": kd_weight.item(),
            "train/lr": adamw.param_groups[0]["lr"],
            "train/muon_lr": muon.param_groups[0]["lr"],
            "train/muon_momentum": muon.param_groups[0]["momentum"],
            "train/grad_norm": grad_norm.item(),
            "train/tokens_per_sec": tokens_per_step * (LOG_EVERY if step else 1) / (now - last_log),
            "time/elapsed": elapsed,
        }, step=step)
        last_log = now
    # No eval during training: each one took 3.4 s, 2.3% of TRAIN_SECONDS over 12 evals (run AJ).
else:
    if rank == 0:
        print(f"Warning: ran out of training data at step {max_steps} before {TRAIN_SECONDS}s")

# Only the raw weights (padded vocab) are written inside the 30-min limit. script/finalize.py, which
# train.sbatch queues behind this job, saves the stock GPT-2 model to outputs/<run> and runs the eval.
if rank == 0:
    os.makedirs("outputs", exist_ok=True)
    torch.save(raw_model.state_dict(), f"outputs/{run_name}.raw.pt")
    print(f"Saved the raw weights to outputs/{run_name}.raw.pt after {step} steps")
    wandb.finish()
dist.destroy_process_group()
