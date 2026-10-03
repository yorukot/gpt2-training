lab4-yoru-AY-491242: AX's settings (KD_CE_WEIGHT 0.4) on the new pipeline: TRAIN_SECONDS 29*60+20, stop predicted before overrunning it, only the raw weights dumped in the job; stock GPT-2 save and val eval in script/finalize.py (job 491243, queued afterany); control for AZ/BA
---
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
KD_CE_WEIGHT = 0.4             # weight of the hard-label CE inside KD_ROWS; the KD loss gets the rest

LOG_EVERY = 10
SEED = 42
