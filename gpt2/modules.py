from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    block_size: int = 1024
    emb_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.0


class Head(nn.Module):
    def __init__(self, config: Config, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.emb_dim, head_size, bias=False)
        self.query = nn.Linear(config.emb_dim, head_size, bias=False)
        self.value = nn.Linear(config.emb_dim, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask", torch.ones(config.block_size, config.block_size).tril()
        )

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
    def __init__(self, config: Config):
        super().__init__()
        assert config.emb_dim % config.num_heads == 0, (
            "emb_dim must be divisible by num_heads"
        )
        head_size = config.emb_dim // config.num_heads
        self.heads = nn.ModuleList(
            [Head(config, head_size) for _ in range(config.num_heads)]
        )
        self.proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X):
        o = torch.cat([head(X) for head in self.heads], dim=-1)
        o = self.proj(o)
        o = self.dropout(o)
        return o


class FeedForward(nn.Module):
    def __init__(self, config: Config, expansion=4):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim * expansion),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.emb_dim * expansion, config.emb_dim),
            nn.Dropout(config.dropout),
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
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = LayerNorm(config.emb_dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = LayerNorm(config.emb_dim)
        self.ff = FeedForward(config)

    def forward(self, X):
        X = X + self.attn(self.ln1(X))  # pre-norm
        X = X + self.ff(self.ln2(X))  # pre-norm
        return X


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tok_to_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_to_emb = nn.Embedding(config.block_size, config.emb_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])
        self.ln = LayerNorm(config.emb_dim)
        self.head = nn.Linear(config.emb_dim, config.vocab_size)

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
