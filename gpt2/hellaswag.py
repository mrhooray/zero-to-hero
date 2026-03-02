import torch
import torch.nn.functional as F
from datasets import load_dataset

from data import enc


ds = load_dataset("Rowan/hellaswag", split="validation")


def evaluate(model, device, block_size, rank=0, world_size=1):
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)

    with torch.inference_mode():
        for i, example in enumerate(ds):
            if i % world_size != rank:
                continue

            ctx_tokens = enc.encode_ordinary(example["ctx"])
            label = int(example["label"])
            losses = []

            for ending in example["endings"]:
                end_tokens = enc.encode_ordinary(" " + ending)
                tokens = (ctx_tokens + end_tokens)[-block_size:]
                end_start = len(tokens) - len(end_tokens)

                # 1, T
                t = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                # pass dummy Y to get logits at all positions
                # model only returns last token when Y is None
                logits, _ = model(t, torch.zeros_like(t))
                end_logits = logits[0, end_start - 1 : -1]
                end_target = t[0, end_start:]
                loss = F.cross_entropy(end_logits, end_target)
                losses.append(loss.item())

            pred = int(torch.tensor(losses).argmin())
            correct += int(pred == label)
            total += 1

    return correct, total
