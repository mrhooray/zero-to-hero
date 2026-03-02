import torch
import torch.nn.functional as F
from datasets import load_dataset

from data import enc


ds = load_dataset("Rowan/hellaswag", split="validation")
MAX_LEN = 256
_actual_max = max(
    len(enc.encode_ordinary(ex["ctx"])) + len(enc.encode_ordinary(" " + end))
    for ex in ds
    for end in ex["endings"]
)
print(f"hellaswag max token length: {_actual_max} (MAX_LEN={MAX_LEN})")
assert _actual_max <= MAX_LEN, (
    f"hellaswag max token length {_actual_max} exceeds MAX_LEN {MAX_LEN}"
)


def evaluate(model, device, block_size, rank=0, world_size=1):
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)

    with torch.inference_mode():
        for i, example in enumerate(ds):
            if i % world_size != rank:
                continue

            ctx_tokens = enc.encode_ordinary(example["ctx"])
            label = int(example["label"])

            # build 4 sequences and track where each ending starts
            seqs, end_starts = [], []
            for ending in example["endings"]:
                end_tokens = enc.encode_ordinary(" " + ending)
                tokens = (ctx_tokens + end_tokens)[-block_size:]
                seqs.append(tokens)
                end_starts.append(len(tokens) - len(end_tokens))

            # pad all 4 to fixed MAX_LEN for consistent shape with torch.compile
            t = torch.zeros(4, MAX_LEN, dtype=torch.long, device=device)
            for j, seq in enumerate(seqs):
                t[j, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

            # single forward pass; dummy Y to get logits at all positions
            logits, _ = model(t, torch.zeros_like(t))  # (4, T, vocab)

            # per-ending loss over ending tokens only
            losses = []
            for j in range(4):
                end_start = end_starts[j]
                end_len = len(seqs[j]) - end_start
                end_logits = logits[j, end_start - 1 : end_start - 1 + end_len]
                end_target = t[j, end_start : end_start + end_len]
                losses.append(F.cross_entropy(end_logits, end_target).item())

            pred = int(torch.tensor(losses).argmin())
            correct += int(pred == label)
            total += 1

    return correct, total
