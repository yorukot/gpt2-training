lab4-yoru-AC-488707: AB + LR schedule linear decay to 0 right after warmup (DECAY_START 0.0, MIN_LR 0), progress measured from step 0 instead of script start; ADAM_EPS constant added (still 1e-8); ~951k tok/s, 6,100 steps, 29:35 total, final val loss 3.3287 (ppl 27.90)
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
DECAY_START = 0.0              # fraction of the training time at full LR before the linear decay
                               # to MIN_LR (0.0 = decay right away, 0.5 = trapezoid)
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0
DROPOUT = 0.0

LOG_EVERY = 10
EVAL_EVERY = 500
SEED = 42
