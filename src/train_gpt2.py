import math
import os
import time

import torch
import torch.distributed as dist
import wandb
from flash_attn_interface import flash_attn_varlen_func
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AttentionInterface, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from data_load import get_batch, load_rows, train_batches
from muon import Muon

# Hyperparameters ------------------------
DATA_DIR = "data/c4"
SEQ_LEN = 1024                 # rows packed to this by prepare_data.py
VOCAB_SIZE = 50304             # GPT-2's 50257 padded to a multiple of 64 for faster matmuls;
                               # resized back to 50257 before saving
MICRO_BATCH_SIZE = 64          # packed rows per GPU per forward pass
GRAD_ACCUM_STEPS = 2           # 64 * 2 * 2 GPUs = 256 rows per step
TRAIN_SECONDS = 29 * 60        # stop training this long after the script starts; the rest of the
                               # 30-min Slurm limit covers torchrun startup, saving and the final eval

EMBED_LR = 3e-3               # AdamW on the tied wte/lm_head and wpe
EMBED_WEIGHT_DECAY = 0.1
SCALAR_LR = 3e-3               # AdamW on the LayerNorms and biases, no weight decay
WARMUP_STEPS = 300             # AdamW only; Muon starts at its full LR
BETAS = (0.9, 0.98)
ADAM_EPS = 1e-10

MUON_LR = 0.02                 # Muon on the 2D weights in the blocks
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

LOG_EVERY = 10
EVAL_EVERY = 500
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
val_rows = load_rows(DATA_DIR, "val", SEQ_LEN)


@torch.no_grad()
def evaluate():
    """Corpus loss over the val rows like the judge: total loss / total scored tokens."""
    raw_model.eval()
    num_rows = len(val_rows[0])
    total = torch.zeros(2, device=device)  # (loss sum, token count)
    for start in range(rank * MICRO_BATCH_SIZE, num_rows, MICRO_BATCH_SIZE * world):
        input_ids, position_ids, labels, cu_seqlens = get_batch(
            val_rows, start, min(start + MICRO_BATCH_SIZE, num_rows), device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = raw_model(
                input_ids=input_ids, position_ids=position_ids, labels=labels,
                cu_seq_lens_q=cu_seqlens, max_length_q=SEQ_LEN,
            ).loss
        count = (labels[:, 1:] != -100).sum()  # HF shifts labels; the first token is not scored
        total += torch.stack([loss.float() * count, count.float()])
    dist.all_reduce(total)
    raw_model.train()
    return (total[0] / total[1]).item()


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

# GPT2Config defaults are GPT-2 small (12 layers, 12 heads, 768 dims) with GPT-1 style
# N(0, 0.02) init. Only the vocab padding differs from stock, and it is undone before saving.
config = GPT2Config(
    vocab_size=VOCAB_SIZE, n_positions=SEQ_LEN, resid_pdrop=DROPOUT, embd_pdrop=DROPOUT, attn_pdrop=DROPOUT,
    use_cache=False, attn_implementation="varlen",
)
with torch.device(device):
    raw_model = GPT2LMHeadModel(config)
raw_model.loss_type = "ForCausalLM"  # unset, HF logs a warning inside forward, which breaks the compiled graph
model = DDP(torch.compile(raw_model, fullgraph=True), device_ids=[device])

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
    hyperparams = {k: v for k, v in globals().items() if k.isupper()}
    wandb.init(entity="cerulean-labs", project="gpt2-training", name=run_name, config=hyperparams)

max_steps = len(train_rows[0]) // (MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS * world)  # until the data runs out
if rank == 0:
    print(f"Training for {TRAIN_SECONDS}s (at most {max_steps} steps), {tokens_per_step:,} tokens per step including padding")

batches = train_batches(train_rows, MICRO_BATCH_SIZE, rank, world, device)
last_log = time.time()
step = 0
for step in range(max_steps):
    if step % LOG_EVERY == 0:  # checked every LOG_EVERY steps, since the broadcast waits for the GPU
        elapsed = elapsed_seconds()
        if elapsed >= TRAIN_SECONDS:
            break
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
        torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)  # the document count varies; compile once for all
        model.require_backward_grad_sync = micro == GRAD_ACCUM_STEPS - 1  # sync grads on the last one
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(
                input_ids=input_ids, position_ids=position_ids, labels=labels,
                cu_seq_lens_q=cu_seqlens, max_length_q=SEQ_LEN,
            ).loss / GRAD_ACCUM_STEPS
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)

    if rank == 0 and step % LOG_EVERY == 0:
        now = time.time()
        wandb.log({
            "train/loss": loss.item() * GRAD_ACCUM_STEPS,
            "train/lr": adamw.param_groups[0]["lr"],
            "train/muon_lr": muon.param_groups[0]["lr"],
            "train/muon_momentum": muon.param_groups[0]["momentum"],
            "train/grad_norm": grad_norm.item(),
            "train/tokens_per_sec": tokens_per_step * (LOG_EVERY if step else 1) / (now - last_log),
            "time/elapsed": elapsed,
        }, step=step)
        last_log = now

    if step % EVAL_EVERY == 0 and step > 0:
        val_loss = evaluate()
        if rank == 0:
            wandb.log({"eval/loss": val_loss, "eval/perplexity": math.exp(val_loss)}, step=step)
else:
    if rank == 0:
        print(f"Warning: ran out of training data at step {max_steps} before {TRAIN_SECONDS}s")

# Back to stock GPT-2's 50257 rows (wte and the tied lm_head), so the saved model is plain GPT-2.
# The dropped rows are never targets, so removing them only lowers the loss.
raw_model.resize_token_embeddings(GPT2Config().vocab_size)

# Save before the final eval so a timeout during eval still keeps the model.
if rank == 0:
    out_dir = f"outputs/{run_name}"
    raw_model.save_pretrained(out_dir)
    GPT2Tokenizer.from_pretrained(f"{DATA_DIR}/tokenizer").save_pretrained(out_dir)
    print(f"Saved to {out_dir}")

val_loss = evaluate()
if rank == 0:
    wandb.log({"eval/loss": val_loss, "eval/perplexity": math.exp(val_loss)}, step=step)
    print(f"Final eval loss {val_loss:.4f}, perplexity {math.exp(val_loss):.2f}")
    wandb.finish()
dist.destroy_process_group()
