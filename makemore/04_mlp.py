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
print(f"X shape {X.shape}")
print(f"Y shape {Y.shape}")

g = torch.Generator().manual_seed(24)
emb_dim = 16
EMB = torch.randn((vocab_size, emb_dim), generator=g)
print(f"EMB shape {EMB.shape}")

X_onehot = F.one_hot(X, num_classes=vocab_size).float()
print(f"X_onehot shape {X_onehot.shape}")
print("X_onehot @ EMB equals to EMB[X]?", torch.allclose(X_onehot @ EMB, EMB[X]))

hidden_dim = 100
input_dim = block_size * emb_dim
W1 = torch.randn((input_dim, hidden_dim), generator=g)
B1 = torch.randn((hidden_dim), generator=g)

emb = X_onehot @ EMB
h = torch.tanh(emb.view(-1, input_dim) @ W1 + B1)
print(f"h shape {h.shape}")

W2 = torch.randn((hidden_dim, vocab_size), generator=g)
B2 = torch.randn((vocab_size), generator=g)
logits = h @ W2 + B2
print(f"logits shape {logits.shape}")

cnts = logits.exp()
probs = cnts / cnts.sum(1, keepdim=True)
print(f"probs shape {probs.shape}")

probs_label = probs[torch.arange(X.shape[0]), Y]
print(f"probs_label shape {probs_label.shape}")

params = [EMB, W1, B1, W2, B2]
params_count = sum(p.nelement() for p in params)
print(f"params_count {params_count}")

n = len(data)
shuffled_indices = torch.randperm(n, generator=g)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
train_idx = shuffled_indices[:n_train]
val_idx = shuffled_indices[n_train : n_train + n_val]
test_idx = shuffled_indices[n_train + n_val :]

data_tr = [data[i] for i in train_idx]
data_val = [data[i] for i in val_idx]
data_te = [data[i] for i in test_idx]

X_tr = torch.tensor([[stoi[ch] for ch in prev] for prev, _ in data_tr])
Y_tr = torch.tensor([stoi[next] for _, next in data_tr])
X_val = torch.tensor([[stoi[ch] for ch in prev] for prev, _ in data_val])
Y_val = torch.tensor([stoi[next] for _, next in data_val])
X_te = torch.tensor([[stoi[ch] for ch in prev] for prev, _ in data_te])
Y_te = torch.tensor([stoi[next] for _, next in data_te])

for p in params:
    p.requires_grad = True
    p.data = torch.randn(p.shape, generator=g)

batch_size = 128
n_epochs = 128 * 16
learning_rate = 0.16
for _ in range(n_epochs):
    batch = torch.randint(0, len(X_tr), (batch_size,), generator=g)
    emb = EMB[X_tr[batch]]
    h = torch.tanh(emb.view(-1, input_dim) @ W1 + B1)
    logits = h @ W2 + B2
    loss = F.cross_entropy(logits, Y_tr[batch])
    print(f"loss {loss.item()}")
    for p in params:
        p.grad = None

    loss.backward()

    for p in params:
        if p.grad is not None:
            p.data += -learning_rate * p.grad

# for hyperparameter tuning
with torch.no_grad():
    emb_val = EMB[X_val]
    h_val = torch.tanh(emb_val.view(-1, input_dim) @ W1 + B1)
    logits_val = h_val @ W2 + B2
    loss_val = F.cross_entropy(logits_val, Y_val)
    print(f"validation loss {loss_val.item()}")

# for model performance
# with torch.no_grad():
#     emb_te = EMB[X_te]
#     h_te = torch.tanh(emb_te.view(-1, input_dim) @ W1 + B1)
#     logits_te = h_te @ W2 + B2
#     loss_te = F.cross_entropy(logits_te, Y_te)
#     print(f"test loss {loss_te.item()}")
