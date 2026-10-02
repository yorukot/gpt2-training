lab4-yoru-AI-488912: AD with a 1-sqrt cooldown instead of linear (DECAY_SHAPE 1-sqrt, DECAY_START 0.5); curved cooldown vs AD; ~959k tok/s, 6,160 steps, 29:22 total, final val loss 3.3167 (ppl 27.57)
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
MIN_LR = 0.0
WARMUP_STEPS = 150
DECAY_START = 0.5              # fraction of the training time at full LR before the decay to MIN_LR
                               # (0.0 = decay right away, 0.5 = trapezoid)
DECAY_SHAPE = "1-sqrt"         # "linear", "cosine" or "1-sqrt" (LR falls fast first, then slowly)
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0
DROPOUT = 0.0

LOG_EVERY = 10
EVAL_EVERY = 500
SEED = 42
