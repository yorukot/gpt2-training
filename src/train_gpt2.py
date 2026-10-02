import math
import os
import time

import torch
import torch.distributed as dist
import wandb
from torch.nn.attention.varlen import varlen_attn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AttentionInterface, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from data_load import get_batch, load_rows, train_batches

# Hyperparameters ------------------------
DATA_DIR = "data/c4"
SEQ_LEN = 1024                 # rows packed to this by prepare_data.py
VOCAB_SIZE = 50304             # GPT-2's 50257 padded to a multiple of 64 for faster matmuls;
                               # resized back to 50257 before saving
MICRO_BATCH_SIZE = 64          # packed rows per GPU per forward pass
GRAD_ACCUM_STEPS = 2           # 64 * 2 * 2 GPUs = 256 rows per step
TRAIN_SECONDS = 29 * 60        # stop training this long after the script starts; the rest of the
                               # 30-min Slurm limit covers torchrun startup, saving and the final eval

LR = 1.8e-3
MIN_LR = 1.8e-4
WARMUP_STEPS = 150
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
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


def lr_at(step, elapsed):
    """Linear warmup over WARMUP_STEPS, times cosine decay to MIN_LR over TRAIN_SECONDS."""
    warmup = min(1.0, (step + 1) / WARMUP_STEPS)
    progress = min(1.0, elapsed / TRAIN_SECONDS)
    return warmup * (MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress)))


def elapsed_seconds():
    """Seconds since START on rank 0, shared so every rank sets the same lr and stops together."""
    t = torch.tensor(time.time() - START, device=device)
    dist.broadcast(t, 0)
    return t.item()


def varlen_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                     cu_seq_lens_q=None, max_length_q=None, **kwargs):
    """Causal attention inside each document of the packed rows, split at cu_seq_lens_q.

    Same math as sdpa with HF's document mask, but no (rows, 1, SEQ_LEN, SEQ_LEN) mask is built
    and no work is spent on blocks across documents. Training only: nothing of it is saved.
    """
    b, h, l, d = query.shape  # HF passes (batch, heads, seq, head_dim); varlen wants (tokens, heads, head_dim)
    q, k, v = (t.transpose(1, 2).reshape(b * l, h, d) for t in (query, key, value))
    # window_size must be the tuple (-1, 0) to be causal; a list silently turns causal masking off.
    out = varlen_attn(q, k, v, cu_seq_lens_q, cu_seq_lens_q, max_length_q, max_length_q,
                      scale=scaling, window_size=(-1, 0))
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

# Weight decay on matrices only, not on biases and LayerNorms.
params = list(raw_model.parameters())  # Tied wte/lm_head is listed once.
optimizer = torch.optim.AdamW(
    [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": WEIGHT_DECAY},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ],
    lr=LR, betas=BETAS, fused=True,
)

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
        lr = lr_at(step, elapsed)
        for group in optimizer.param_groups:
            group["lr"] = lr

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
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if rank == 0 and step % LOG_EVERY == 0:
        now = time.time()
        wandb.log({
            "train/loss": loss.item() * GRAD_ACCUM_STEPS,
            "train/lr": lr,
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
