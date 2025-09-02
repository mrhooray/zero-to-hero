import os
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
names_path = os.path.join(script_dir, "names.txt")

with open(names_path, "r") as f:
    names = f.read().splitlines()

chars = sorted(set("".join(names) + "."))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

block_size = 3
data = []
for name in names:
    chs = ["."] * block_size + list(name) + ["."]
    for i in range(len(chs) - block_size):
        prev = chs[i : i + block_size]
        next = chs[i + block_size]
        data.append((prev, next))

X = torch.tensor([[stoi[ch] for ch in prev] for prev, _ in data])
Y = torch.tensor([stoi[next] for _, next in data])

print("X shape:", X.shape)
print("Y shape:", Y.shape)
