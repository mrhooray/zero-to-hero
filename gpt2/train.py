import math
import time

import torch

import hellaswag
from data import DataLoader, enc
from modules import Config, GPT


def nice(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple


vocab_size = nice(enc.n_vocab, 256)
block_size = 1024
emb_dim = 768
num_heads = 12
num_layers = 12
dropout = 0.2
batch_size = 16
steps = 1024 * 1
lr_max = 3e-4
lr_min = lr_max * 0.1
lr_warmup_steps = 128
weight_decay = 0.1
eval_interval = 1
eval_batches = 1
hellaswag_interval = 256

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

autocast_dtype = torch.bfloat16 if device == "cuda" else None

print(f"vocab_size: {vocab_size}")
print(f"block_size: {block_size}")
print(f"emb_dim: {emb_dim}")
print(f"num_heads: {num_heads}")
print(f"num_layers: {num_layers}")
print(f"dropout: {dropout}")
print(f"batch_size: {batch_size}")
print(f"steps: {steps}")
print(f"lr_max: {lr_max}")
print(f"lr_min: {lr_min}")
print(f"lr_warmup_steps: {lr_warmup_steps}")
print(f"weight_decay: {weight_decay}")
print(f"eval_interval: {eval_interval}")
print(f"eval_batches: {eval_batches}")
print(f"hellaswag_interval: {hellaswag_interval}")
print(f"device: {device}")

torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

loader_train = DataLoader("train", batch_size, block_size, device)
loader_val = DataLoader("val", batch_size, block_size, device)

config = Config(
    vocab_size=vocab_size,
    block_size=block_size,
    emb_dim=emb_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
)
model = GPT(config).to(device)
model = torch.compile(model)
decay_params = [p for p in model.parameters() if p.dim() >= 2]
no_decay_params = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=lr_max,
    betas=(0.9, 0.95),
    eps=1e-8,
    fused=device == "cuda",
)
print(f"num_params: {sum(p.numel() for p in model.parameters()):,}")


@torch.inference_mode()
def estimate_loss():
    out = {}
    for split, loader in [("train", loader_train), ("val", loader_val)]:
        losses = torch.zeros(eval_batches, device=device)
        for i in range(eval_batches):
            X, Y = loader.next_batch()
            with torch.autocast(
                device_type=device,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                _, loss = model(X, Y)
            losses[i] = loss
        out[split] = losses.mean().item()
    return out


def get_lr(step):
    if step < lr_warmup_steps:
        return lr_max * (step + 1) / lr_warmup_steps
    progress = (step - lr_warmup_steps) / (steps - lr_warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


model.train()
t0 = time.time()
for step in range(steps):
    X, Y = loader_train.next_batch()
    with torch.autocast(
        device_type=device, dtype=autocast_dtype, enabled=autocast_dtype is not None
    ):
        _, loss = model(X, Y)
    for param_group in optimizer.param_groups:
        param_group["lr"] = get_lr(step)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % eval_interval == 0:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        tokens_per_sec = batch_size * block_size * eval_interval / elapsed
        model.eval()
        losses = estimate_loss()
        model.train()
        print(
            f"step {step:8d} | train {losses['train']:8.4f} | val {losses['val']:8.4f} | lr {get_lr(step):8.2e} | {elapsed * 1000:8.2f}ms | {tokens_per_sec:8.0f} tok/s"
        )
        t0 = time.time()

    if step % hellaswag_interval == 0:
        model.eval()
        correct, total = hellaswag.evaluate(model, device, block_size)
        model.train()
        c, t = correct.item(), total.item()
        print(f"step {step:8d} | hellaswag {c}/{t} ({c / t:.4f})")
