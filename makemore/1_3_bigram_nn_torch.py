import os
import time
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
names_path = os.path.join(script_dir, "names.txt")

with open(names_path, "r") as f:
    names = f.read().splitlines()

print(names[:16])
print(len(names))
print(min(len(x) for x in names))
print(max(len(x) for x in names))

chars = sorted(set("".join(names) + "."))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
bigrams = []
for name in names:
    chs = ["."] + list(name) + ["."]
    bigrams.extend(zip(chs, chs[1:]))

X = torch.tensor([stoi[a] for a, _ in bigrams])
Y = torch.tensor([stoi[b] for _, b in bigrams])
W = torch.randn(vocab_size, vocab_size, requires_grad=True)

n_epochs = 256
learning_rate = 64

for epoch in range(n_epochs):
    start_time = time.time()
    xenc = F.one_hot(X, num_classes=vocab_size).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(len(probs)), Y].log().mean()
    print(f"Loss @ epoch {epoch + 1}: {loss.item():.4f}")

    W.grad = None
    loss.backward()

    W.data += -learning_rate * W.grad

    epoch_time = time.time() - start_time
    print(f"Time: {epoch_time:.2f}s")

generated = []
for _ in range(32):
    name = []
    ix = stoi["."]
    while True:
        logits = W[ix]
        probs = F.softmax(logits, dim=0)
        ix = torch.multinomial(probs, num_samples=1).item()
        if ix == stoi["."]:
            break
        name.append(itos[ix])
    generated.append("".join(name))

print("Generated names:")
for name in generated:
    print(name)
