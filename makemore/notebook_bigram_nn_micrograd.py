import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from micrograd.value import Value

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

X = []
Y = []
for name in names:
    chs = ["."] + list(name) + ["."]
    for a, b in zip(chs, chs[1:]):
        X.append(stoi[a])
        Y.append(stoi[b])

W = [
    [Value(np.random.uniform(0, 1)) for _ in range(vocab_size)]
    for _ in range(vocab_size)
]


def forward(i):
    logits = W[i]
    # to avoid exp overflow
    max_logit = max(logits, key=lambda x: x.data)
    exp_logits = [(x - max_logit).exp() for x in logits]
    sum_exp = sum(exp_logits)
    probs = [x / sum_exp for x in exp_logits]
    return probs


n_epochs = 10
learning_rate = 0.1

for epoch in range(n_epochs):
    start_time = time.time()
    total_loss = Value(0)
    for xi, yi in zip(X, Y):
        probs = forward(xi)
        # to avoid log(0)
        loss = -(probs[yi] + Value(1e-7)).log()
        total_loss = total_loss + loss

    total_loss.backward()

    for i in range(vocab_size):
        for j in range(vocab_size):
            W[i][j].data -= learning_rate * W[i][j].grad
            W[i][j].grad = 0.0

    epoch_time = time.time() - start_time
    print(
        f"Epoch {epoch}: loss = {total_loss.data / len(X):.4f}, time = {epoch_time:.2f}s"
    )

generated = []
for _ in range(32):
    name = []
    ix = stoi["."]
    while True:
        probs = forward(ix)
        probs_data = [x.data for x in probs]
        ix = np.random.choice(len(probs_data), p=probs_data)
        if ix == stoi["."]:
            break
        name.append(itos[ix])
    generated.append("".join(name))

print("Generated names:")
for name in generated:
    print(name)
