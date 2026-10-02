lab4-yoru-AD-488712: AC with a 50% trapezoid schedule (DECAY_START 0.5: full LR for the first half of training time, then linear decay to 0); ~936k tok/s, 6,010 steps, 29:41 total, final val loss 3.3149 (ppl 27.52)
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
DECAY_START = 0.5              # fraction of the training time at full LR before the linear decay
                               # to MIN_LR (0.0 = decay right away, 0.5 = trapezoid)
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0
DROPOUT = 0.0

LOG_EVERY = 10
EVAL_EVERY = 500
SEED = 42
