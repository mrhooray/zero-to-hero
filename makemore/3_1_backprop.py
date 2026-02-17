import os
import random
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
names_path = os.path.join(script_dir, "names.txt")

with open(names_path, "r") as f:
    names = f.read().splitlines()

chars = sorted(list(set("".join(names))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

block_size = 3


def build_dataset(names):
    X, Y = [], []
    for w in names:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


random.seed(42)
random.shuffle(names)
n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))

Xtr, Ytr = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[n1:n2])
Xte, Yte = build_dataset(names[n2:])


def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(
        f"{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}"
    )


n_embd = 10
n_hidden = 64

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = (
    torch.randn((n_embd * block_size, n_hidden), generator=g)
    * (5 / 3)
    / ((n_embd * block_size) ** 0.5)
)
b1 = torch.randn(n_hidden, generator=g) * 0.1
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1
bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden)) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

batch_size = 32
n = batch_size
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# embed
emb = C[Xb]
# flatten
embcat = emb.view(emb.shape[0], -1)
# linear layer
hprebn = embcat @ W1 + b1
# batch norm
bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1 / (n - 1) * (bndiff2).sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
# non-linearity
h = torch.tanh(hpreact)
# linear output layer
logits = h @ W2 + b2
# cross entropy loss
# same as F.cross_entropy(logits, Yb)
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

for p in parameters:
    p.grad = None
for t in [
    logprobs,
    probs,
    counts,
    counts_sum,
    counts_sum_inv,
    norm_logits,
    logit_maxes,
    logits,
    h,
    hpreact,
    bnraw,
    bnvar_inv,
    bnvar,
    bndiff2,
    bndiff,
    hprebn,
    bnmeani,
    embcat,
    emb,
]:
    t.retain_grad()
loss.backward()

# Exercise 1: backprop through the whole thing manually

# -----------------
# YOUR CODE HERE :)
# -----------------
# print(logprobs.shape, probs.shape)
dlogprobs = F.one_hot(Yb, logprobs.shape[1])  # from indexing
dlogprobs = -dlogprobs / n  # from mean, neg
dprobs = probs**-1 * dlogprobs
# print(probs.shape, counts_sum_inv.shape, counts_sum.shape, counts.shape)
dcounts_sum_inv = (counts * dprobs).sum(1, keepdims=True)  # from broadcast
dcounts_sum = -(counts_sum**-2) * dcounts_sum_inv
dcounts = counts_sum_inv * dprobs
dcounts += 1 * dcounts_sum
# print(norm_logits.shape, logit_maxes.shape, logits.shape)
dnorm_logits = counts * dcounts
dlogit_maxes = -1 * dnorm_logits.sum(1, keepdims=True)  # from broadcast
dlogits = 1 * dnorm_logits
dlogits += F.one_hot(logits.max(1).indices, logits.shape[1]) * dlogit_maxes  # from max

cmp("logprobs", dlogprobs, logprobs)
cmp("probs", dprobs, probs)
cmp("counts_sum_inv", dcounts_sum_inv, counts_sum_inv)
cmp("counts_sum", dcounts_sum, counts_sum)
cmp("counts", dcounts, counts)
cmp("norm_logits", dnorm_logits, norm_logits)
cmp("logit_maxes", dlogit_maxes, logit_maxes)
cmp("logits", dlogits, logits)

# print(logits.shape, h.shape, W2.shape, b2.shape)
dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)

cmp("h", dh, h)
cmp("W2", dW2, W2)
cmp("b2", db2, b2)

# print(h.shape, hpreact.shape)
dhpreact = (1 - h**2) * dh
cmp("hpreact", dhpreact, hpreact)

# print(hpreact.shape, bngain.shape, bnraw.shape, bnbias.shape)
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnraw = bngain * dhpreact
dbnbias = (1 * dhpreact).sum(0, keepdim=True)
cmp("bngain", dbngain, bngain)
cmp("bnraw", dbnraw, bnraw)
cmp("bnbias", dbnbias, bnbias)
# print(bnraw.shape, bndiff.shape, bnvar_inv.shape)
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
dbnvar = -0.5 * (bnvar + 1e-5) ** -1.5 * dbnvar_inv
# print(bnvar.shape, bndiff2.shape)
dbndiff2 = 1 / (n - 1) * torch.ones_like(bndiff2) * dbnvar
dbndiff = bnvar_inv * dbnraw
dbndiff += 2 * bndiff * dbndiff2
# print(bndiff.shape, hprebn.shape, bnmeani.shape)
dbnmeani = -dbndiff.sum(0, keepdim=True)
dhprebn = dbndiff.clone()  # due to +=
dhprebn += 1 / n * torch.ones_like(bnmeani) * dbnmeani
cmp("bnvar_inv", dbnvar_inv, bnvar_inv)
cmp("bnvar", dbnvar, bnvar)
cmp("bndiff2", dbndiff2, bndiff2)
cmp("bndiff", dbndiff, bndiff)
cmp("bnmeani", dbnmeani, bnmeani)
cmp("hprebn", dhprebn, hprebn)

# print(hprebn.shape, embcat.shape, W1.shape, b1.shape)
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
cmp("embcat", dembcat, embcat)
cmp("W1", dW1, W1)
cmp("b1", db1, b1)

demb = dembcat.view(emb.shape)
cmp("emb", demb, emb)

# print(emb.shape, C.shape, Xb.shape)
dC = torch.zeros_like(C)
# for i in range(Xb.shape[0]):
#     for j in range(Xb.shape[1]):
#         dC[Xb[i, j]] += demb[i, j]
dC.index_add_(0, Xb.view(-1), demb.view(-1, demb.shape[-1]))
cmp("C", dC, C)

# Exercise 2: backprop through cross_entropy in one go

loss_fast = F.cross_entropy(logits, Yb)

# -----------------
# YOUR CODE HERE :)
# -----------------

dlogits = F.softmax(logits, 1)  # probability
dlogits[range(n), Yb] -= 1
dlogits /= n
cmp("logits", dlogits, logits)
