from os.path import abspath, dirname, join

import torch
import torch.nn as nn
import torch.nn.functional as F


class Vocab:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.size = len(self.chars)

    def encode(self, seq):
        return [self.stoi[c] for c in seq]

    def decode(self, seq):
        return [self.itos[i] for i in seq]


class Head(nn.Module):
    def __init__(self, block_size, emb_dim, head_size):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("mask", torch.ones(block_size, block_size).tril())

    def forward(self, X):
        q = self.query(X)  # B, T, head_size
        k = self.key(X)
        v = self.value(X)
        _, T, head_size = q.shape
        w = q @ k.transpose(-2, -1)  # B, T, T
        w = w * head_size**-0.5  # scale by contracting dimension
        w = w.masked_fill(self.mask[:T, :T] == 0, -torch.inf)
        w = F.softmax(w, dim=-1)
        o = w @ v  # B, T, head_size
        return o


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, emb_dim, head_size):
        super().__init__()
        assert emb_dim % head_size == 0, "emb_dim must be divisible by head_size"
        num_heads = emb_dim // head_size
        self.heads = nn.ModuleList(
            [Head(block_size, emb_dim, head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, X):
        o = torch.cat([head(X) for head in self.heads], dim=-1)
        o = self.proj(o)
        return o


class FeedForward(nn.Module):
    def __init__(self, emb_dim, expansion=4):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * expansion),
            nn.ReLU(),
            nn.Linear(emb_dim * expansion, emb_dim),
        )

    def forward(self, X):
        return self.ff(X)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_mean = x.mean(-1, keepdim=True)
        batch_var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        return self.weight * x_norm + self.bias


class Block(nn.Module):
    def __init__(self, block_size, emb_dim, head_size):
        super().__init__()
        self.ln1 = LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(block_size, emb_dim, head_size)
        self.ln2 = LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim)

    def forward(self, X):
        X = X + self.attn(self.ln1(X))  # pre-norm
        X = X + self.ff(self.ln2(X))  # pre-norm
        return X


class GPTLM(nn.Module):
    def __init__(self, vocab_size, block_size, emb_size, num_blocks, head_size):
        super().__init__()
        self.tok_to_emb = nn.Embedding(vocab_size, emb_size)
        self.pos_to_emb = nn.Embedding(block_size, emb_size)
        self.blocks = nn.Sequential(
            *[Block(block_size, emb_size, head_size) for _ in range(num_blocks)]
        )
        self.ln = LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, vocab_size)

    def forward(self, X, Y=None):
        B, T = X.shape
        emb_tok = self.tok_to_emb(X)  # B, T, C
        emb_pos = self.pos_to_emb(torch.arange(T))  # T, C
        x = emb_tok + emb_pos
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)  # B, T, vocab_size

        if Y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), Y.view(B * T))

        return logits, loss

    @torch.inference_mode()
    def generate(self, X, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(X)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            X_next = torch.multinomial(probs, num_samples=1)
            X = torch.cat((X, X_next), dim=1)
        return X


def main():
    block_size = 8
    emb_dim = 32
    num_blocks = 4
    head_size = 8
    batch_size = 128
    steps = 1024 * 128
    eval_interval = 1024
    eval_size = 256
    lr = 1e-3
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        # device = "mps"
        device = "cpu"
    else:
        device = "cpu"

    print(f"block_size: {block_size}")
    print(f"emb_dim: {emb_dim}")
    print(f"num_blocks: {num_blocks}")
    print(f"head_size: {head_size}")
    print(f"batch_size: {batch_size}")
    print(f"steps: {steps}")
    print(f"eval_interval: {eval_interval}")
    print(f"eval_size: {eval_size}")
    print(f"lr: {lr}")
    print(f"device: {device}")

    torch.manual_seed(42)

    file_path = join(dirname(abspath(__file__)), "input.txt")

    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    vocab = Vocab(text)

    data = torch.tensor(vocab.encode(text), dtype=torch.long).to(device)
    n = int(0.9 * len(data))
    data_train = data[:n]
    data_val = data[n:]

    def get_batch(split):
        data = data_train if split == "train" else data_val
        ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
        offsets = torch.arange(block_size, device=device)
        X = data[ix.unsqueeze(1) + offsets]
        Y = data[ix.unsqueeze(1) + offsets + 1]
        return X, Y

    model = GPTLM(vocab.size, block_size, emb_dim, num_blocks, head_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params: {num_params:,}")

    @torch.inference_mode()
    def estimate_loss():
        out = {}
        # model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_size, device=device)
            for i in range(eval_size):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[i] = loss
            out[split] = losses.mean().item()
        # model.train()
        return out

    for step in range(steps):
        X, Y = get_batch("train")
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {step}:", f"losses: {losses}")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(
        "".join(vocab.decode(model.generate(context, max_new_tokens=1024)[0].tolist()))
    )


if __name__ == "__main__":
    main()
