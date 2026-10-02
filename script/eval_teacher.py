import math
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

from data_load import PAD, load_rows

# Hyperparameters ------------------------
DATA_DIR = "data/c4"
SEQ_LEN = 1024
GPT2_VOCAB = 50257             # GPT-J has 50400 logits: GPT-2's 50257 plus unused padding
TOKENS_PER_BATCH = 32_768      # documents per forward pass, counted with padding
TOP_K = 8                      # how many entries a stored teacher distribution would keep
REVISIONS = {"EleutherAI/gpt-j-6b": "float16"}  # half the download of the fp32 main branch
STUDENT_LOSS = 3.3149          # lab4-yoru-AD-488712, our best so far, on the same val documents
# ------------------------------------

teacher = sys.argv[1]
revision = REVISIONS.get(teacher)
device = "cuda"

# The teacher reads our GPT-2 token ids directly, so its ids must mean the same tokens.
gpt2_vocab = GPT2Tokenizer.from_pretrained(f"{DATA_DIR}/tokenizer").get_vocab()
teacher_vocab = AutoTokenizer.from_pretrained(teacher, revision=revision).get_vocab()
assert {t: i for t, i in teacher_vocab.items() if i < GPT2_VOCAB} == gpt2_vocab, "vocab differs from GPT-2"

model = AutoModelForCausalLM.from_pretrained(teacher, revision=revision, dtype=torch.bfloat16).to(device).eval()


def documents(rows):
    """Each document of the packed rows, without the padding segments at the row ends."""
    tokens, positions = (x.reshape(-1).astype(np.int64) for x in rows)
    docs = np.split(tokens, np.flatnonzero(positions == 0)[1:])
    return [doc for doc in docs if (doc != PAD).any()]


# Longest first, so each batch pads its documents to about the same length.
docs = sorted(documents(load_rows(DATA_DIR, "val", SEQ_LEN)), key=len, reverse=True)
batches, start = [], 0
while start < len(docs):
    size = max(1, TOKENS_PER_BATCH // len(docs[start]))
    batches.append(docs[start : start + size])
    start += size


@torch.no_grad()
def score(bos):
    """Judge-style stats over the val documents, each on its own with tokens 1.. as the targets.

    With bos the teacher reads <|endoftext|> + the document (GPT-2 style models saw that separator
    before every document in training) instead of starting at the first token. The targets are the
    same either way, since the judge never scores a document's first token. Padding goes on the
    right, which causal attention never looks at, so no attention mask is needed.
    """
    totals = torch.zeros(4, dtype=torch.float64, device=device)  # loss, top-k mass, top-1 hits, target in top-k
    count = 0
    for batch in batches:
        inputs = torch.full((len(batch), len(batch[0])), PAD, dtype=torch.long)
        targets = torch.full_like(inputs, -100)
        for i, doc in enumerate(torch.from_numpy(doc) for doc in batch):
            n = len(doc)
            if bos:
                inputs[i, 1:n] = doc[:-1]  # inputs[i, 0] is already PAD, i.e. <|endoftext|>
                targets[i, 1:n] = doc[1:]
            else:
                inputs[i, :n] = doc
                targets[i, : n - 1] = doc[1:]
        targets[targets == PAD] = -100  # like evaluate() in train_gpt2.py
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(input_ids=inputs).logits[..., :GPT2_VOCAB]
        scored = targets != -100
        logp = logits[scored].float().log_softmax(-1)
        target = targets[scored]
        top = logp.topk(TOP_K, dim=-1)
        totals += torch.stack([
            -logp.gather(1, target[:, None]).sum(),
            top.values.exp().sum(),
            (top.indices[:, 0] == target).sum(),
            (top.indices == target[:, None]).any(-1).sum(),
        ]).double()
        count += scored.sum().item()
    return (totals / count).tolist(), count


input_tokens = sum(len(doc) for doc in docs)
print(f"{teacher}: {len(docs):,} val documents, {input_tokens:,} tokens, {len(batches)} batches")
for bos in (False, True):
    torch.cuda.synchronize()
    started = time.time()
    (loss, mass, top1, in_top), count = score(bos)
    seconds = time.time() - started
    print(
        f"{teacher} | {'<|endoftext|> + doc' if bos else 'doc only (judge)'} | "
        f"loss {loss:.4f} (ppl {math.exp(loss):.2f}, {loss - STUDENT_LOSS:+.4f} vs student) | "
        f"top-{TOP_K} mass {mass:.3f} | top-1 acc {top1:.3f} | target in top-{TOP_K} {in_top:.3f} | "
        f"{count:,} scored tokens | {input_tokens / seconds:,.0f} tok/s"
    )
