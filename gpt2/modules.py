import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, block_size, emb_dim, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
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
        w = self.dropout(w)
        o = w @ v  # B, T, head_size
        return o


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, emb_dim, num_heads, dropout):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(block_size, emb_dim, head_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        o = torch.cat([head(X) for head in self.heads], dim=-1)
        o = self.proj(o)
        o = self.dropout(o)
        return o


class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout, expansion=4):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * expansion, emb_dim),
            nn.Dropout(dropout),
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
    def __init__(self, block_size, emb_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(block_size, emb_dim, num_heads, dropout)
        self.ln2 = LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim, dropout)

    def forward(self, X):
        X = X + self.attn(self.ln1(X))  # pre-norm
        X = X + self.ff(self.ln2(X))  # pre-norm
        return X


class GPT(nn.Module):
    def __init__(
        self, vocab_size, block_size, emb_size, num_blocks, num_heads, dropout
    ):
        super().__init__()
        self.tok_to_emb = nn.Embedding(vocab_size, emb_size)
        self.pos_to_emb = nn.Embedding(block_size, emb_size)
        self.blocks = nn.Sequential(
            *[
                Block(block_size, emb_size, num_heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.ln = LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, vocab_size)

    def forward(self, X, Y=None):
        B, T = X.shape
        emb_tok = self.tok_to_emb(X)  # B, T, C
        emb_pos = self.pos_to_emb(torch.arange(T, device=X.device))  # T, C
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
