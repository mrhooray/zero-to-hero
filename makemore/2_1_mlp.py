import os
import torch
import torch.nn.functional as F
import torch.nn.init as init

script_dir = os.path.dirname(os.path.abspath(__file__))
names_path = os.path.join(script_dir, "names.txt")

with open(names_path, "r") as f:
    names = f.read().splitlines()

chars = sorted(set("".join(names) + "."))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

block_size = 4
emb_dim = 16
hidden_dim = 256
batch_size = 256
epochs = 1024 * 128
learning_rate = 0.16
decay_points = [0.68, 0.84, 0.92]
decay_factors = [4, 4, 4]
bn_momentum = 0.1
bn_eps = 1e-5

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
print(f"block_size = {block_size}")
print(f"emb_dim = {emb_dim}")
print(f"hidden_dim = {hidden_dim}")
print(f"batch_size = {batch_size}")

g = torch.Generator().manual_seed(24)
EMB = torch.randn((vocab_size, emb_dim), generator=g)
print(f"EMB shape {EMB.shape}")

X_onehot = F.one_hot(X, num_classes=vocab_size).float()
print(f"X_onehot shape {X_onehot.shape}")
print("X_onehot @ EMB equals to EMB[X]?", torch.allclose(X_onehot @ EMB, EMB[X]))

input_dim = block_size * emb_dim

W1 = torch.empty((input_dim, hidden_dim))
init.kaiming_normal_(W1, mode="fan_in", nonlinearity="tanh", generator=g)
B1 = torch.zeros((hidden_dim))
BN_GAIN = torch.ones((1, hidden_dim))
BN_BIAS = torch.zeros((1, hidden_dim))
BN_RUNNING_MEAN = torch.zeros((1, hidden_dim))
BN_RUNNING_VAR = torch.ones((1, hidden_dim))

emb = X_onehot @ EMB
linear = emb.view(-1, input_dim) @ W1 + B1
norm = (linear - linear.mean(dim=0, keepdim=True)) / torch.sqrt(
    linear.var(dim=0, keepdim=True, unbiased=False) + bn_eps
)
h = torch.tanh(BN_GAIN * norm + BN_BIAS)
print(f"h shape {h.shape}")

W2 = torch.empty((hidden_dim, vocab_size))
init.kaiming_normal_(W2, mode="fan_in", nonlinearity="linear", generator=g)
B2 = torch.zeros((vocab_size))
logits = h @ W2 + B2
print(f"logits shape {logits.shape}")

cnts = logits.exp()
probs = cnts / cnts.sum(1, keepdim=True)
print(f"probs shape {probs.shape}")

probs_label = probs[torch.arange(X.shape[0]), Y]
print(f"probs_label shape {probs_label.shape}")

params = [EMB, W1, B1, BN_GAIN, BN_BIAS, W2, B2]
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

for epoch in range(epochs):
    current_decay = 1
    for point, factor in zip(decay_points, decay_factors):
        if epoch >= int(point * epochs):
            current_decay = factor
    lr = learning_rate / current_decay

    batch = torch.randint(0, len(X_tr), (batch_size,), generator=g)
    emb = EMB[X_tr[batch]]
    linear = emb.view(-1, input_dim) @ W1 + B1
    batch_mean = linear.mean(dim=0, keepdim=True)
    batch_var = linear.var(dim=0, keepdim=True, unbiased=False)
    norm = (linear - batch_mean) / torch.sqrt(batch_var + bn_eps)
    h = torch.tanh(BN_GAIN * norm + BN_BIAS)
    with torch.no_grad():
        BN_RUNNING_MEAN.mul_(1 - bn_momentum).add_(bn_momentum * batch_mean.detach())
        BN_RUNNING_VAR.mul_(1 - bn_momentum).add_(bn_momentum * batch_var.detach())
    logits = h @ W2 + B2
    loss = F.cross_entropy(logits, Y_tr[batch])
    print(f"epoch {epoch}, loss {loss.item()}")

    for p in params:
        p.grad = None
    loss.backward()

    for p in params:
        if p.grad is not None:
            p.data += -lr * p.grad

# for hyperparameter tuning
with torch.no_grad():
    emb_val = EMB[X_val]
    linear_val = emb_val.view(-1, input_dim) @ W1 + B1
    norm_val = (linear_val - BN_RUNNING_MEAN) / torch.sqrt(BN_RUNNING_VAR + bn_eps)
    h_val = torch.tanh(BN_GAIN * norm_val + BN_BIAS)
    logits_val = h_val @ W2 + B2
    loss_val = F.cross_entropy(logits_val, Y_val)
    print(f"validation loss {loss_val.item()}")

# for model performance
# with torch.no_grad():
#     emb_te = EMB[X_te]
#     linear_te = emb_te.view(-1, input_dim) @ W1 + B1
#     norm_te = (linear_te - BN_RUNNING_MEAN) / torch.sqrt(BN_RUNNING_VAR + bn_eps)
#     h_te = torch.tanh(BN_GAIN * norm_te + BN_BIAS)
#     logits_te = h_te @ W2 + B2
#     loss_te = F.cross_entropy(logits_te, Y_te)
#     print(f"test loss {loss_te.item()}")
