import numpy as np
import torch

PAD = 50256  # GPT-2's <|endoftext|>, which prepare_data.py writes as padding


def load_rows(data_dir, name, seq_len):
    """The packed rows from prepare_data.py: (tokens, positions), each (rows, seq_len) uint16."""
    tokens = np.memmap(f"{data_dir}/{name}_tokens.bin", dtype=np.uint16, mode="r").reshape(-1, seq_len)
    positions = np.memmap(f"{data_dir}/{name}_positions.bin", dtype=np.uint16, mode="r").reshape(-1, seq_len)
    return tokens, positions


def get_batch(rows, start, end, device):
    """Rows start..end-1 as (input_ids, position_ids, labels, cu_seqlens) on device.

    Positions restart at 0 at every document (and at the padding). cu_seqlens holds where each
    of those segments starts in the flattened rows, plus the total length, so attention can stay
    inside its document. A document's first token gets label -100, so the previous document's last
    token is never trained to predict it; padding gets -100 too.
    """
    tokens, positions = rows
    tokens, positions = tokens[start:end].astype(np.int32), positions[start:end].astype(np.int32)
    cu_seqlens = np.append(np.flatnonzero(positions.reshape(-1) == 0), positions.size).astype(np.int32)
    # Pinned memory lets the copies run without waiting for the GPU to finish its queued work.
    input_ids, position_ids, cu_seqlens = (
        torch.from_numpy(x).pin_memory().to(device, non_blocking=True) for x in (tokens, positions, cu_seqlens)
    )
    input_ids, position_ids = input_ids.long(), position_ids.long()
    labels = input_ids.masked_fill((position_ids == 0) | (input_ids == PAD), -100)
    return input_ids, position_ids, labels, cu_seqlens


def train_batches(rows, batch_size, rank, world, device):
    """Read rows in order forever; each rank takes every world-th batch."""
    start = rank * batch_size
    while True:
        yield get_batch(rows, start, start + batch_size, device)
        start += batch_size * world
