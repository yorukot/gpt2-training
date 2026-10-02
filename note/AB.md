lab4-yoru-AB-488543: varlen (flash) attention inside documents with fullgraph compile, vocab padded to 50304 (resized back before saving), 2B-token data/c4 (2,096,873 rows), pinned-memory batches, inductor cache in .cache/inductor
---
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
