import time

import torch

from data import DataLoader, enc
from modules import Config, GPT

vocab_size = enc.n_vocab
block_size = 1024
emb_dim = 768
num_heads = 12
num_layers = 12
dropout = 0.2
batch_size = 16
steps = 1024 * 128
lr = 3e-4
eval_interval = 1024
eval_batches = 256

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"vocab_size: {vocab_size}")
print(f"block_size: {block_size}")
print(f"emb_dim: {emb_dim}")
print(f"num_heads: {num_heads}")
print(f"num_layers: {num_layers}")
print(f"dropout: {dropout}")
print(f"batch_size: {batch_size}")
print(f"steps: {steps}")
print(f"lr: {lr}")
print(f"eval_interval: {eval_interval}")
print(f"eval_batches: {eval_batches}")
print(f"device: {device}")

torch.manual_seed(42)

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
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
print(f"num_params: {sum(p.numel() for p in model.parameters()):,}")


@torch.inference_mode()
def estimate_loss():
    out = {}
    for split, loader in [("train", loader_train), ("val", loader_val)]:
        losses = torch.zeros(eval_batches, device=device)
        for i in range(eval_batches):
            X, Y = loader.next_batch()
            _, loss = model(X, Y)
            losses[i] = loss
        out[split] = losses.mean().item()
    return out


model.train()
t0 = time.time()
for step in range(steps):
    X, Y = loader_train.next_batch()
    _, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        t1 = time.time()
        model.eval()
        losses = estimate_loss()
        model.train()
        print(
            f"step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {t1 - t0:.2f}s"
        )
        t0 = time.time()
