"""
baselines/transformer.py
─────────────────────────
Transformer, Informer (ProbSparse attention), and PatchTST baselines.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding (shared)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# ---------------------------------------------------------------------------
# Vanilla Transformer
# ---------------------------------------------------------------------------
class TransformerModel(nn.Module):
    def __init__(self, seq_len: int = 96, pred_len: int = 24,
                 d_model: int = 128, nhead: int = 4, d_ff: int = 256,
                 n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj    = nn.Linear(d_model * seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.pos_enc(self.input_proj(x.unsqueeze(-1)))
        h = self.encoder(h)
        return self.proj(h.reshape(B, -1))


# ---------------------------------------------------------------------------
# Informer (ProbSparse attention approximation)
# ---------------------------------------------------------------------------
class ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention: samples top-u queries by their query sparsity
    score, runs standard attention only on those, fills the rest with the
    mean-value output.
    """

    def __init__(self, d_model: int, nhead: int, factor: int = 5,
                 dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.nhead    = nhead
        self.d_head   = d_model // nhead
        self.factor   = factor
        self.scale    = self.d_head ** -0.5
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        return x.reshape(B, L, self.nhead, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        Q = self._split(self.W_q(x))
        K = self._split(self.W_k(x))
        V = self._split(self.W_v(x))

        u = max(1, int(self.factor * math.log(L + 1)))
        u = min(u, L)

        # Sparsity scores: max-minus-mean of each query's log-prob row
        QK = (Q @ K.transpose(-2, -1)) * self.scale         # (B,h,L,L)
        scores = QK.max(-1).values - QK.mean(-1)            # (B,h,L)
        top_idx = scores.topk(u, dim=-1).indices            # (B,h,u)

        # Gather top-u queries and run full attention on them
        idx_exp = top_idx.unsqueeze(-1).expand(*top_idx.shape, L)
        Q_top = Q.gather(2, top_idx.unsqueeze(-1).expand(*top_idx.shape, self.d_head))
        attn = F.softmax(Q_top @ K.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.drop(attn)
        ctx_top = attn @ V                                   # (B,h,u,dh)

        # Fill output — start from mean-value context
        ctx = V.mean(2, keepdim=True).expand_as(V).clone()
        idx_exp2 = top_idx.unsqueeze(-1).expand(*top_idx.shape, self.d_head)
        ctx.scatter_(2, idx_exp2, ctx_top)

        ctx = ctx.transpose(1, 2).reshape(B, L, -1)
        return self.out(ctx)


class Informer(nn.Module):
    def __init__(self, seq_len: int = 96, pred_len: int = 24,
                 d_model: int = 128, nhead: int = 4,
                 d_ff: int = 256, n_layers: int = 2,
                 factor: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            nn.Sequential(
                ProbSparseAttention(d_model, nhead, factor, dropout),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff), nn.GELU(),
                nn.Linear(d_ff, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(n_layers)
        ])
        self.proj = nn.Linear(d_model * seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.pos_enc(self.input_proj(x.unsqueeze(-1)))
        for layer in self.layers:
            h = layer(h)
        return self.proj(h.reshape(B, -1))


# ---------------------------------------------------------------------------
# PatchTST
# ---------------------------------------------------------------------------
class PatchTST(nn.Module):
    """
    Patch Time-Series Transformer (Nie et al., 2023).
    Segments the input into non-overlapping patches, projects each,
    and runs a standard Transformer encoder over the patch tokens.
    """

    def __init__(self, seq_len: int = 96, pred_len: int = 24,
                 patch_len: int = 16, stride: int = 8,
                 d_model: int = 128, nhead: int = 4, d_ff: int = 256,
                 n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        n_patches = (seq_len - patch_len) // stride + 1
        self.n_patches = n_patches

        self.patch_emb = nn.Linear(patch_len, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb   = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
        self.drop      = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj    = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        # Extract patches
        patches = x.unfold(-1, self.patch_len, self.stride)  # (B, n_patches, patch_len)
        tok = self.patch_emb(patches)                         # (B, n_patches, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos_emb
        tok = self.drop(tok)
        tok = self.encoder(tok)
        return self.proj(tok[:, 0])                           # CLS token → forecast
