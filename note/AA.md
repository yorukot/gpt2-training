lab4-yoru-AA-488399: first run on the new packed data/c4 with document masking, new ~/gpt2/.venv-gpu (torch 2.14.0+cu130, transformers 5.17.0); ~819k tok/s, data ran out at step 4548 (26:12 total), final val loss 3.3943 (ppl 29.79)
---
DATA_DIR = "data/c4"
SEQ_LEN = 1024                 # rows packed to this by prepare_data.py
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
