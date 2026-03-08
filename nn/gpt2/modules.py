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


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.num_heads == 0
        self.proj_heads = nn.Linear(config.emb_dim, config.emb_dim * 3)
        self.proj_output = nn.Linear(config.emb_dim, config.emb_dim)
        self.dropout_output = nn.Dropout(config.dropout)
        self.emb_dim = config.emb_dim
        self.num_heads = config.num_heads
        self.dropout = config.dropout

    def forward(self, X):
        B, T, C = X.shape

        q, k, v = self.proj_heads(X).split(self.emb_dim, dim=-1)
        q = q.view(B, T, self.num_heads, self.emb_dim // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.emb_dim // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.emb_dim // self.num_heads).transpose(1, 2)

        dropout = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_output(y)
        y = self.dropout_output(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: Config, expansion=4):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim * expansion),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.emb_dim * expansion, config.emb_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, X):
        return self.ff(X)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.emb_dim, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.emb_dim, bias=False)
        self.ff = FeedForward(config)

    def forward(self, X):
        X = X + self.attn(self.ln1(X))  # pre-norm
        X = X + self.ff(self.ln2(X))  # pre-norm
        return X


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_to_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_to_emb = nn.Embedding(config.block_size, config.emb_dim)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.emb_dim, bias=False)
        self.head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        # weight tying
        self.tok_to_emb.weight = self.head.weight

        # init params
        std = 0.02
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std_cond = std
                if name.endswith("proj_output") or name.endswith("ff.2"):
                    # each block adds 2 residual contributions (attn + ff); scale down
                    # so the residual stream variance stays ~1 after num_layers blocks
                    std_cond /= (2 * self.config.num_layers) ** 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std_cond)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, X, Y=None):
        B, T = X.shape
        emb_tok = self.tok_to_emb(X)  # B, T, C
        emb_pos = self.pos_to_emb(torch.arange(T, device=X.device))  # T, C
        x = emb_tok + emb_pos
        x = self.emb_dropout(x)
        x = self.blocks(x)
        x = self.ln(x)

        if Y is None:
            logits = self.head(x[:, [-1], :])
            loss = None
        else:
            logits = self.head(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), Y.view(B * T))

        return logits, loss

    @torch.inference_mode()
    def generate(self, X, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            X_ctx = (
                X
                if X.size(1) <= self.config.block_size
                else X[:, -self.config.block_size :]
            )
            logits, _ = self(X_ctx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            X_next = torch.multinomial(probs, num_samples=1)
            X = torch.cat((X, X_next), dim=1)
        return X
