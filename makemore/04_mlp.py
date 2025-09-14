import os
import torch
import torch.nn.functional as F

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
EMB = torch.randn((27, 2))

print(f"X shape {X.shape}")
print(f"Y shape {Y.shape}")
print(f"EMB shape {EMB.shape}")

X_onehot = F.one_hot(X, num_classes=27).float()
print(f"X_onehot shape {X_onehot.shape}")
print("X_onehot @ EMB equals to EMB[X]?", torch.allclose(X_onehot @ EMB, EMB[X]))

W1 = torch.randn((6, 100))
B1 = torch.randn((100))

emb = X_onehot @ EMB
h = emb.view(-1, 6) @ W1 + B1
print(f"h shape {h.shape}")

W2 = torch.randn((100, 27))
B2 = torch.randn((27))
logits = h @ W2 + B2
print(f"logits shape {logits.shape}")

cnts = logits.exp()
probs = cnts / cnts.sum(1, keepdim=True)
print(f"probs shape {probs.shape}")

probs_label = probs[torch.arange(X.shape[0]), Y]
print(f"probs_label shape {probs_label.shape}")
