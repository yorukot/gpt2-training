"""Turn a run's raw weights into the stock GPT-2 checkpoint, then score it on the val rows.

train_gpt2.py only dumps outputs/<run>.raw.pt (padded 50304-row vocab) inside the 30-min limit. This:
1. resizes the vocab back to GPT-2's 50257 (the dropped rows are never targets, so removing them only
   lowers the loss) and saves the stock GPT2LMHeadModel and tokenizer to outputs/<run>, the folder that
   gets pushed to the Hub;
2. loads that folder back the way the judge does and computes the val loss like the old in-training
   evaluate(): packed val rows with documents kept apart, bf16 autocast, total loss / scored tokens;
3. writes eval/loss and eval/perplexity to the run's W&B summary.
Without a .raw.pt (runs from before this script) it only evaluates outputs/<run> and prints the result.
"""
import math
import os
import sys
from typing import cast

import torch
import wandb
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from data_load import get_batch, load_rows

# Hyperparameters ------------------------
DATA_DIR = "data/c4"
SEQ_LEN = 1024
VOCAB_SIZE = 50304             # the padded vocab train_gpt2.py trains with
BATCH_ROWS = 64                # val rows per forward pass
# ------------------------------------

run_name = sys.argv[1]
out_dir, raw_path = f"outputs/{run_name}", f"outputs/{run_name}.raw.pt"
device = "cuda"

converted = os.path.exists(raw_path)
if converted:
    # The same config as training (no dropout, no cache), with HF's default attention.
    config = GPT2Config(vocab_size=VOCAB_SIZE, n_positions=SEQ_LEN, resid_pdrop=0.0, embd_pdrop=0.0,
                        attn_pdrop=0.0, use_cache=False)
    model = GPT2LMHeadModel(config)
    model.load_state_dict(torch.load(raw_path, map_location="cpu"))
    model.resize_token_embeddings(GPT2Config().vocab_size)  # wte and the tied lm_head
    model.save_pretrained(out_dir)
    GPT2Tokenizer.from_pretrained(f"{DATA_DIR}/tokenizer").save_pretrained(out_dir)
    print(f"Saved the stock GPT-2 model to {out_dir}")

# What the judge loads: the saved folder as a plain GPT2LMHeadModel.
# With output_loading_info the result is (model, info), which the type hints don't say.
model, info = cast(tuple[GPT2LMHeadModel, dict], GPT2LMHeadModel.from_pretrained(out_dir, output_loading_info=True))
assert not any(info.values()) and model.config.vocab_size == 50257, f"not a stock GPT-2 checkpoint: {info}"
model.loss_type = "ForCausalLM"  # unset, HF logs a warning inside forward
# ty misreads PreTrainedModel.to, which HF wraps with functools.wraps, as taking the model first.
model = model.to(device).eval()  # ty: ignore[invalid-argument-type]

# Positions restart at every document, so HF builds a mask that keeps the documents apart.
val_rows = load_rows(DATA_DIR, "val", SEQ_LEN)
num_rows = len(val_rows[0])
total = torch.zeros(2, dtype=torch.float64, device=device)  # (loss sum, scored tokens)
with torch.no_grad():
    for start in range(0, num_rows, BATCH_ROWS):
        input_ids, position_ids, labels, _ = get_batch(val_rows, start, min(start + BATCH_ROWS, num_rows), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, position_ids=position_ids, labels=labels).loss
        count = (labels[:, 1:] != -100).sum()  # HF shifts labels; the first token is not scored
        total += torch.stack([loss.double() * count, count.double()])
val_loss = (total[0] / total[1]).item()
print(f"{run_name}: val loss {val_loss:.4f}, perplexity {math.exp(val_loss):.2f}")

if converted:
    runs = list(wandb.Api().runs("cerulean-labs/gpt2-training", filters={"display_name": run_name}))
    if runs:
        runs[0].summary.update({"eval/loss": val_loss, "eval/perplexity": math.exp(val_loss)})
        print(f"Wrote eval/loss to W&B run {runs[0].id}")
    else:
        print(f"No W&B run named {run_name}; the eval is only printed")
