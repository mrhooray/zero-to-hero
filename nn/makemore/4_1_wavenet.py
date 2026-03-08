import math
import os

import torch
import torch.nn.functional as F
import torch.nn.init as init
from typing import List, Tuple, Optional, Literal


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_modules", {})

    def parameters(self) -> List[torch.Tensor]:
        params = []
        for _, p in object.__getattribute__(self, "_parameters").items():
            params.append(p)
        for _, mod in object.__getattribute__(self, "_modules").items():
            params.extend(mod.parameters())
        return params

    def buffers(self) -> List[torch.Tensor]:
        bufs = []
        for _, b in object.__getattribute__(self, "_buffers").items():
            bufs.append(b)
        for _, mod in object.__getattribute__(self, "_modules").items():
            bufs.extend(mod.buffers())
        return bufs

    def get_modules(self):
        mods = [self]
        for _, mod in object.__getattribute__(self, "_modules").items():
            mods.extend(mod.get_modules())
        return mods

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            object.__getattribute__(self, "_modules")[name] = value
        elif isinstance(value, torch.Tensor):
            if value.requires_grad:
                object.__getattribute__(self, "_parameters")[name] = value
            else:
                object.__getattribute__(self, "_buffers")[name] = value
        super().__setattr__(name, value)


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        generator: Optional[torch.Generator] = None,
        nonlinearity: Literal[
            "linear", "tanh", "relu", "leaky_relu", "selu"
        ] = "linear",
    ):
        super().__init__()
        weight = torch.empty((out_features, in_features))
        weight.requires_grad = True
        init.kaiming_normal_(
            weight, mode="fan_in", nonlinearity=nonlinearity, generator=generator
        )
        self.weight = weight

        bias = torch.zeros(out_features)
        bias.requires_grad = True
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        weight = torch.randn((num_embeddings, embedding_dim), generator=generator)
        weight.requires_grad = True
        self.weight = weight

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.weight[indices]


class FlattenBy(Module):
    def __init__(
        self,
        seq: int,
    ):
        super().__init__()
        self.seq = seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        if s // self.seq == 1:
            return x.view(b, d * self.seq)
        return x.view(b, s // self.seq, d * self.seq)


class BatchNorm1d(Module):
    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        weight = torch.ones(dim)
        weight.requires_grad = True
        self.weight = weight

        bias = torch.zeros(dim)
        bias.requires_grad = True
        self.bias = bias

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        dim = tuple(range(x.ndim - 1))
        if training:
            batch_mean = x.mean(dim, keepdim=True)
            batch_var = x.var(dim, keepdim=True, unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean = (
                    self.running_mean * (1 - self.momentum)
                    + self.momentum * batch_mean.squeeze()
                )
                self.running_var = (
                    self.running_var * (1 - self.momentum)
                    + self.momentum * batch_var.squeeze()
                )

            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return self.weight * x_norm + self.bias


class Tanh(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, f"layer_{i}", layer)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        for layer in object.__getattribute__(self, "_modules").values():
            if isinstance(layer, BatchNorm1d):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


def load_data(
    names_path: str, block_size: int
) -> Tuple[List[Tuple[List[str], str]], List[str]]:
    with open(names_path, "r") as f:
        names = f.read().splitlines()

    chars = sorted(set("".join(names) + "."))

    data = []
    for name in names:
        chs = ["."] * block_size + list(name) + ["."]
        for i in range(len(chs) - block_size):
            prev = chs[i : i + block_size]
            nxt = chs[i + block_size]
            data.append((prev, nxt))

    return data, chars


def prepare_tensors(
    data: List[Tuple[List[str], str]], stoi: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor([[stoi[ch] for ch in prev] for prev, _ in data])
    Y = torch.tensor([stoi[nxt] for _, nxt in data])
    return X, Y


def train_step(
    model: Module,
    X_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    learning_rate: float,
) -> float:
    for p in model.parameters():
        p.grad = None

    logits = model(X_batch, training=True)
    loss = F.cross_entropy(logits, Y_batch)
    loss.backward()

    for p in model.parameters():
        if p.grad is not None:
            p.data += -learning_rate * p.grad

    return loss.item()


def evaluate(model: Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(X, training=False)
        loss = F.cross_entropy(logits, Y)
    return loss.item()


def main():
    # Configuration
    block_size = 8
    emb_dim = 16
    dilation = 2
    hidden_dim = 128
    batch_size = 128
    steps = 1024 * 128
    learning_rate = 0.1
    decay_points = [0.68, 0.84, 0.92]
    decay_factors = [4, 8, 16]
    seed = 24

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    names_path = os.path.join(script_dir, "names.txt")

    data, chars = load_data(names_path, block_size)
    stoi = {ch: i for i, ch in enumerate(chars)}
    # itos = {i: ch for i, ch in enumerate(chars)}  # for sampling
    vocab_size = len(chars)

    print(f"Dataset size: {len(data)}")
    print(f"Vocab size: {vocab_size}")
    print(f"block_size = {block_size}")
    print(f"emb_dim = {emb_dim}")
    print(f"dilation = {dilation}")
    print(f"hidden_dim = {hidden_dim}")
    print(f"batch_size = {batch_size}")

    # Split data
    g = torch.Generator().manual_seed(seed)
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

    X_tr, Y_tr = prepare_tensors(data_tr, stoi)
    X_val, Y_val = prepare_tensors(data_val, stoi)
    X_te, Y_te = prepare_tensors(data_te, stoi)

    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")

    # Create model
    g_model = torch.Generator().manual_seed(seed)
    layers = []
    for i in range(int(math.log(block_size) / math.log(dilation))):
        layers.append(FlattenBy(dilation))
        layers.append(
            Linear(
                (emb_dim if i == 0 else hidden_dim) * dilation,
                hidden_dim,
                generator=g_model,
                nonlinearity="tanh",
            ),
        )
        layers.append(BatchNorm1d(hidden_dim))
        layers.append(Tanh())

    model = Sequential(
        Embedding(vocab_size, emb_dim, generator=g_model),
        *layers,
        Linear(hidden_dim, vocab_size, generator=g_model, nonlinearity="linear"),
    )

    # Count parameters and buffers
    n_params = sum(p.nelement() for p in model.parameters())
    n_buffers = sum(b.nelement() for b in model.buffers())
    print(f"Total parameters: {n_params}")
    print(f"Total buffers: {n_buffers}")

    # Training loop
    for step in range(steps):
        current_decay = 1
        for point, factor in zip(decay_points, decay_factors):
            if step >= int(point * steps):
                current_decay = factor
        lr = learning_rate / current_decay

        batch_indices = torch.randint(0, len(X_tr), (batch_size,), generator=g)
        X_batch = X_tr[batch_indices]
        Y_batch = Y_tr[batch_indices]

        loss = train_step(model, X_batch, Y_batch, lr)

        if step % 1024 == 0:
            print(f"step {step}, loss {loss:.4f}, lr {lr:.6f}")

    # Validation
    val_loss = evaluate(model, X_val, Y_val)
    print(f"validation loss {val_loss:.4f}")

    # Final evaluation
    # test_loss = evaluate(model, X_te, Y_te)
    # print(f"test loss {test_loss:.4f}")


if __name__ == "__main__":
    main()
