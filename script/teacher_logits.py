import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from data_load import PAD, load_rows

# Hyperparameters ------------------------
TEACHER = "EleutherAI/gpt-j-6b"
REVISION = "float16"           # half the download of the fp32 main branch
DATA_DIR = "data/c4"
OUT_DIR = "/work/u9603854/gpt2/teacher/gpt-j-6b"  # ~52 GB when full; home has no room for it
SEQ_LEN = 1024
GPT2_VOCAB = 50257             # GPT-J has 50400 logits: GPT-2's 50257 plus unused padding
TOP_K = 8
CHUNK_ROWS = 1024              # train rows per unit of work, ~35 s on one GPU; each gets a done marker
TEACHER_ROWS = 1563 * CHUNK_ROWS  # 1,600,512 rows of room; a 29-min run uses ~1.54M
TOKENS_PER_BATCH = 32_768      # documents per forward pass, counted with padding
GEN_SECONDS = 26 * 60          # start no new chunk after this; the rest of the 30-min limit covers the last one
# ------------------------------------

START = time.time()
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
device = f"cuda:{os.environ['LOCAL_RANK']}"
torch.cuda.set_device(device)

out = Path(OUT_DIR)
(out / "done").mkdir(parents=True, exist_ok=True)
files = {"idx": (out / "train_idx.bin", np.uint16), "logp": (out / "train_logp.bin", np.float16)}
for path, dtype in files.values():
    # Every rank makes sure the file has its full size; truncating to the same size twice is harmless.
    path.touch()
    size = TEACHER_ROWS * SEQ_LEN * TOP_K * np.dtype(dtype).itemsize
    if path.stat().st_size < size:
        os.truncate(path, size)
if rank == 0:
    meta = {"teacher": TEACHER, "revision": REVISION, "top_k": TOP_K, "rows": TEACHER_ROWS,
            "chunk_rows": CHUNK_ROWS, "input": "<|endoftext|> + document[:-1]"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

tokens, positions = load_rows(DATA_DIR, "train", SEQ_LEN)
model = AutoModelForCausalLM.from_pretrained(TEACHER, revision=REVISION, dtype=torch.bfloat16).to(device).eval()


@torch.no_grad()
def teacher_chunk(row_tokens, row_positions):
    """Top-k teacher log-probs for these packed rows, as (idx, logp), each (rows, SEQ_LEN, TOP_K).

    Entry [r, j] is the teacher's distribution for the token at j + 1, the one the student predicts
    from position j. The log-probs are over the full vocab, not renormalized over the top k.
    Each document runs on its own as <|endoftext|> + the document minus its last token, which keeps
    its length: the output at p has read up to d[p-1], so it is the distribution for d[p] and goes
    to entry p - 1. Entries with no next token in the same document (a document's last token,
    padding) stay 0; the student's label there is -100.
    """
    flat_tokens = row_tokens.reshape(-1).astype(np.int64)
    starts = np.flatnonzero(row_positions.reshape(-1) == 0)
    ends = np.append(starts[1:], flat_tokens.size)
    # Longest first, so each batch pads its documents to about the same length.
    docs = sorted(
        [(s, e) for s, e in zip(starts, ends) if e - s > 1 and (flat_tokens[s:e] != PAD).any()],
        key=lambda doc: doc[1] - doc[0], reverse=True,
    )
    idx = np.zeros((flat_tokens.size, TOP_K), np.uint16)
    logp = np.zeros((flat_tokens.size, TOP_K), np.float16)
    i = 0
    while i < len(docs):
        batch = docs[i : i + max(1, TOKENS_PER_BATCH // (docs[i][1] - docs[i][0]))]
        i += len(batch)
        # Right padding, which causal attention never looks at, so no attention mask is needed.
        inputs = torch.full((len(batch), batch[0][1] - batch[0][0]), PAD, dtype=torch.long)
        for b, (s, e) in enumerate(batch):
            inputs[b, 1 : e - s] = torch.from_numpy(flat_tokens[s : e - 1])  # inputs[b, 0] stays <|endoftext|>
        logits = model(input_ids=inputs.to(device)).logits[..., :GPT2_VOCAB]
        top = logits.float().log_softmax(-1).topk(TOP_K, dim=-1)
        top_idx, top_logp = top.indices.cpu().numpy(), top.values.half().cpu().numpy()
        for b, (s, e) in enumerate(batch):
            idx[s : e - 1] = top_idx[b, 1 : e - s]
            logp[s : e - 1] = top_logp[b, 1 : e - s]
    return idx.reshape(*row_tokens.shape, TOP_K), logp.reshape(*row_tokens.shape, TOP_K)


# Chunks go round-robin over the ranks, so the finished ones form an almost unbroken prefix of the
# train rows, which the student reads in order. Chunks with a done marker are skipped, so a later
# job picks up where this one stopped.
for chunk in range(rank, TEACHER_ROWS // CHUNK_ROWS, world):
    if time.time() - START > GEN_SECONDS:
        break
    marker = out / "done" / f"{chunk:05d}"
    if marker.exists():
        continue
    started = time.time()
    r0, r1 = chunk * CHUNK_ROWS, (chunk + 1) * CHUNK_ROWS
    row_tokens, row_positions = tokens[r0:r1], positions[r0:r1]
    chunk_out = dict(zip(("idx", "logp"), teacher_chunk(row_tokens, row_positions)))
    for name, (path, dtype) in files.items():
        with open(path, "r+b") as f:
            f.seek(r0 * SEQ_LEN * TOP_K * np.dtype(dtype).itemsize)
            f.write(chunk_out[name].tobytes())
    marker.touch()

    # Alignment check: the true next token should be in the stored top k about 76% of the time
    # (eval_teacher.py on val); an off-by-one would show far less.
    target = row_tokens[:, 1:]
    scored = (row_positions[:, 1:] != 0) & (target != PAD)
    in_top = (chunk_out["idx"][:, :-1] == target[..., None]).any(-1)[scored].mean()
    mass = np.exp(chunk_out["logp"][:, :-1].astype(np.float32)).sum(-1)[scored].mean()
    seconds = time.time() - started
    print(
        f"rank {rank} chunk {chunk} rows {r0:,}-{r1:,}: target in top-{TOP_K} {in_top:.3f}, "
        f"top-{TOP_K} mass {mass:.3f}, {(row_tokens != PAD).sum() / seconds:,.0f} tok/s, "
        f"{time.time() - START:.0f}s since start"
    )
