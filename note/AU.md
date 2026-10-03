lab4-yoru-AU-490626: first distillation run: AS + GPT-J-6B teacher (top-8 + rest bucket, T=1) on KD_ROWS (0, 327,680) = steps 0-1,280 (~19% of the run), loss 0.2*CE + 0.8*KD there, plain CE after; vs AS; ~911k tok/s (also after KD: the new loss code is ~11% slower than AS), 5,870 steps (1.54B tokens), 29:36 total, final val loss 3.2494 (ppl 25.77), worse than AS by 0.018 ~ its 13% token deficit, so net KD gain ~0
---
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
KD_ROWS = (0, 327_680)         # (first, end): train rows distilled from the teacher, the rest plain CE;
                               # None = no distillation. (0, 327_680) = steps 0-1,280 of AS's 6,760
KD_CE_WEIGHT = 0.2             # weight of the hard-label CE inside KD_ROWS; the KD loss gets the rest

LOG_EVERY = 10
SEED = 42
