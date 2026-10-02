import bisect
import hashlib
import math
import multiprocessing
import os
import shutil

import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Hyperparameters ------------------------
OUT_DIR = "data/c4"
SEQ_LEN = 1024  # the judge only scores each document's first 1024 tokens
TRAIN_TOKENS = 2_000_000_000  # real (non-padding) tokens to keep
JUDGE_DOCS = 100_000  # the judge's test set: first 100k of validation.shuffle(seed=42)
VAL_DOCS = 10_000  # our validation set: the next 10k documents after the judge's
BATCH_DOCS = 10_000  # documents tokenized and packed together
SHUFFLE_BUFFER = 10_000
NUM_WORKERS = 64  # train processes, each streaming its own share of C4's 1024 train shards
SEED = 42
# ------------------------------------

# Same tokenizer and truncation as the judge; no BOS/EOS, so each document starts at position 0.
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
PAD = tokenizer.eos_token_id
seen = set()  # hashes of every document kept so far, plus the judge's documents


def doc_hash(text):
    """Hash of the text with case and whitespace normalized, for exact deduplication."""
    return hashlib.blake2b(
        " ".join(text.lower().split()).encode(), digest_size=8
    ).digest()


def pack(docs):
    """Best-fit documents into rows of SEQ_LEN tokens; returns (tokens, positions), each (rows, SEQ_LEN).

    Positions restart at 0 at every document, so the model's document masking keeps documents
    apart. The padding at the end of a row is PAD tokens with positions restarting at 0 too.
    """
    rows = []  # documents in each row
    free = []  # sorted (free space, row index)
    for doc in docs:
        i = bisect.bisect_left(
            free, (len(doc), -1)
        )  # the row with the least space that fits
        if i == len(free):
            rows.append([])
            space, row = SEQ_LEN, len(rows) - 1
        else:
            space, row = free.pop(i)
        rows[row].append(doc)
        bisect.insort(free, (space - len(doc), row))

    tokens = np.full((len(rows), SEQ_LEN), PAD, dtype=np.uint16)
    positions = np.zeros((len(rows), SEQ_LEN), dtype=np.uint16)
    for r, row_docs in enumerate(rows):
        start = 0
        for doc in row_docs:
            tokens[r, start : start + len(doc)] = doc
            positions[r, start : start + len(doc)] = np.arange(len(doc))
            start += len(doc)
        positions[r, start:] = np.arange(SEQ_LEN - start)
    return tokens, positions


def write_rows(docs, name, max_tokens):
    """Deduplicate, tokenize and pack docs into <name>_tokens.bin and <name>_positions.bin."""
    total = 0
    with (
        open(f"{OUT_DIR}/{name}_tokens.bin", "wb") as f_tokens,
        open(f"{OUT_DIR}/{name}_positions.bin", "wb") as f_positions,
    ):
        for batch in docs.iter(batch_size=BATCH_DOCS):
            texts = []
            for text in batch["text"]:
                h = doc_hash(text)
                if h not in seen:
                    seen.add(h)
                    texts.append(text)
            ids = tokenizer(texts, truncation=True, max_length=SEQ_LEN)["input_ids"]
            tokens, positions = pack(ids)
            tokens.tofile(f_tokens)
            positions.tofile(f_positions)
            total += sum(len(doc) for doc in ids)
            print(
                f"{name}: kept {len(texts):,}/{len(batch['text']):,} docs, {total:,} tokens",
                flush=True,
            )
            if total >= max_tokens:
                break


os.makedirs(OUT_DIR, exist_ok=True)
tokenizer.save_pretrained(f"{OUT_DIR}/tokenizer")

# C4's files by direct download URL. load_dataset("allenai/c4") lists the repo through the Hub API
# first, which gets rate limited (HTTP 429); plain file downloads don't.
C4_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en"
VAL_FILES = [f"{C4_URL}/c4-validation.{i:05d}-of-00008.json.gz" for i in range(8)]
TRAIN_FILES = [f"{C4_URL}/c4-train.{i:05d}-of-01024.json.gz" for i in range(1024)]

# Load validation like the judge: the same 8 files in the same order, so shuffle(seed=42) gives
# the same documents. Its 100k documents are never written, only marked as seen so no copy of
# them gets into train or val.
validation = load_dataset("json", data_files={"validation": VAL_FILES}, split="validation")
validation = validation.shuffle(seed=SEED)
seen.update(doc_hash(text) for text in validation.select(range(JUDGE_DOCS))["text"])
write_rows(validation.select(range(JUDGE_DOCS, JUDGE_DOCS + VAL_DOCS)), "val", math.inf)



def train_worker(worker):
    """Write train_<worker>_*.bin from this worker's shards. Each worker starts from the judge and
    val hashes in `seen` (inherited through fork), so duplicates across workers are not removed."""
    shard = train.shard(num_shards=NUM_WORKERS, index=worker)
    shard = shard.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)
    write_rows(shard, f"train_{worker}", TRAIN_TOKENS // NUM_WORKERS)


# Workers inherit this through fork; each streams its own share of the 1024 files.
train = load_dataset("json", data_files={"train": TRAIN_FILES}, split="train", streaming=True)


with multiprocessing.get_context("fork").Pool(NUM_WORKERS) as pool:
    pool.map(train_worker, range(NUM_WORKERS))

# Join the workers' parts into train_tokens.bin and train_positions.bin.
for kind in ("tokens", "positions"):
    with open(f"{OUT_DIR}/train_{kind}.bin", "wb") as out:
        for worker in range(NUM_WORKERS):
            part = f"{OUT_DIR}/train_{worker}_{kind}.bin"
            with open(part, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(part)
