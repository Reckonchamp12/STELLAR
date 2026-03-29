"""
stellar/components.py
──────────────────────
Individual building blocks of the STELLAR architecture.

Classes
-------
RevIN                   — Reversible Instance Normalisation
AdaptiveSpectralFilter  — Hypernetwork-modulated per-sample FFT filter
KoopmanDynamics         — Nonlinear lifting + stable linear Koopman operator
MRTP                    — Multi-Resolution Temporal Patching
NPHead                  — Neural-Process probabilistic head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  RevIN – Reversible Instance Normalisation
# ---------------------------------------------------------------------------
class RevIN(nn.Module):
    """
    Reversible Instance Normalisation with learnable affine parameters.
    After norm() is called, denorm() reverts to the original scale.
    The stored statistics (_mu, _sig) are used by the parent model to
    rescale predictive variance back to the original space.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
        self._mu: torch.Tensor | None = None
        self._sig: torch.Tensor | None = None

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise (B, T) → (B, T); stores mean and std for denorm."""
        self._mu  = x.mean(-1, keepdim=True)
        self._sig = x.std(-1, keepdim=True).clamp(min=self.eps)
        return (x - self._mu) / self._sig * self.w + self.b

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Invert normalisation."""
        return (x - self.b) / (self.w + self.eps) * self._sig + self._mu


# ---------------------------------------------------------------------------
# 2.  Adaptive Spectral Filter
# ---------------------------------------------------------------------------
class AdaptiveSpectralFilter(nn.Module):
    """
    Hypernetwork-modulated complex bandpass filter in frequency space.

    For each sample the network outputs a per-sample complex filter applied
    on top of a shared base filter.  This allows the model to suppress noise
    and amplify the frequencies most informative for each individual series.

    Architecture
    ------------
    base_r, base_i  : shared learnable real/imag filter (n_freq,)
    hyper           : Linear(T, d_hid) → GELU → Linear(d_hid, 2*n_freq)
                      produces per-sample (Δr, Δi) corrections
    """

    def __init__(self, seq_len: int, d_hid: int = 64) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.nf = seq_len // 2 + 1

        self.base_r = nn.Parameter(torch.ones(self.nf))
        self.base_i = nn.Parameter(torch.zeros(self.nf))

        self.hyper = nn.Sequential(
            nn.Linear(seq_len, d_hid),
            nn.GELU(),
            nn.Linear(d_hid, self.nf * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T)
        delta = self.hyper(x)                             # (B, 2*nf)
        dr, di = delta[:, : self.nf], delta[:, self.nf :]

        fr = self.base_r + dr                              # (B, nf)
        fi = self.base_i + di

        xf = torch.fft.rfft(x, dim=-1)                   # (B, nf) complex
        yr = xf.real * fr - xf.imag * fi
        yi = xf.real * fi + xf.imag * fr
        return torch.fft.irfft(torch.complex(yr, yi), n=self.seq_len, dim=-1)


# ---------------------------------------------------------------------------
# 3.  Koopman Latent Dynamics
# ---------------------------------------------------------------------------
class KoopmanDynamics(nn.Module):
    """
    Learns φ: R^T → R^d (nonlinear lifting) and a linear Koopman operator
    K = U · diag(λ) · Vᵀ in that latent space.

    Eigenvalues λ = exp(ρ + iθ), ρ = −softplus(·) < 0  ⟹  |λ| < 1
    guarantees asymptotic stability by construction.

    Future states are computed as z_{t+k} = K^k z_0 and decoded to
    a point forecast of length pred_len.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d: int = 64,
        n_modes: int = 32,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.d = d

        # Nonlinear encoder  φ: R^T → R^d
        self.enc = nn.Sequential(
            nn.Linear(seq_len, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
        )

        # Spectral factorisation of K
        self.U = nn.Parameter(torch.randn(d, n_modes) * 0.02)
        self.V = nn.Parameter(torch.randn(d, n_modes) * 0.02)
        self._log_rho = nn.Parameter(torch.full((n_modes,), 0.5))
        self.theta    = nn.Parameter(torch.randn(n_modes) * 0.1)

        # Decoder: (pred_len, d) latent sequence → (pred_len,) forecast
        self.dec = nn.Sequential(
            nn.Linear(d * pred_len, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, pred_len),
        )

    def _build_koopman(self):
        rho   = -F.softplus(self._log_rho)          # (n_modes,) < 0
        lam_r = torch.exp(rho) * torch.cos(self.theta)
        lam_i = torch.exp(rho) * torch.sin(self.theta)
        Kr = self.U @ torch.diag(lam_r) @ self.V.T  # (d, d)
        Ki = self.U @ torch.diag(lam_i) @ self.V.T
        return Kr, Ki

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        z_r = self.enc(x)                            # (B, d)
        z_i = torch.zeros_like(z_r)
        Kr, Ki = self._build_koopman()

        states = []
        for _ in range(self.pred_len):
            nr = z_r @ Kr.T - z_i @ Ki.T
            ni = z_r @ Ki.T + z_i @ Kr.T
            z_r, z_i = nr, ni
            states.append(z_r)

        future = torch.stack(states, dim=1)          # (B, pred_len, d)
        return self.dec(future.reshape(B, -1))       # (B, pred_len)


# ---------------------------------------------------------------------------
# 4.  Multi-Resolution Temporal Patching (MRTP)
# ---------------------------------------------------------------------------
class MRTP(nn.Module):
    """
    Encodes patches at scales {4, 8, 16} independently with lightweight
    Transformers, then fuses via cross-scale multi-head attention.
    Returns a pooled context vector of dimension d.
    """

    def __init__(self, seq_len: int, d: int = 64) -> None:
        super().__init__()
        self.scales    = [4, 8, 16]
        self.n_patches = [seq_len // s for s in self.scales]

        self.emb = nn.ModuleList([nn.Linear(s, d) for s in self.scales])
        self.pos = nn.ParameterList([
            nn.Parameter(torch.randn(1, n, d) * 0.02)
            for n in self.n_patches
        ])

        def _enc_layer():
            return nn.TransformerEncoderLayer(
                d_model=d, nhead=4, dim_feedforward=d * 2,
                dropout=0.1, batch_first=True, norm_first=True,
            )

        self.local_tf = nn.ModuleList([
            nn.TransformerEncoder(_enc_layer(), num_layers=2)
            for _ in self.scales
        ])

        self.cross_attn = nn.MultiheadAttention(d, num_heads=4,
                                                 batch_first=True, dropout=0.1)
        self.ln   = nn.LayerNorm(d)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        all_tokens = []
        for s, n, emb, pos, tf in zip(
            self.scales, self.n_patches, self.emb, self.pos, self.local_tf
        ):
            patches = x[:, : n * s].reshape(B, n, s)   # (B, n, s)
            tok = emb(patches) + pos                    # (B, n, d)
            tok = tf(tok)
            all_tokens.append(tok)

        tokens = torch.cat(all_tokens, dim=1)           # (B, Σn, d)
        fused, _ = self.cross_attn(tokens, tokens, tokens)
        fused = self.ln(tokens + fused)

        pooled = self.pool(fused.transpose(1, 2)).squeeze(-1)  # (B, d)
        return self.proj(pooled)


# ---------------------------------------------------------------------------
# 5.  Neural-Process Probabilistic Head (NPPH)
# ---------------------------------------------------------------------------
class NPHead(nn.Module):
    """
    Encodes context into q(z|c) = N(μ_z, σ_z²) via reparametrisation.
    Decodes z ⊕ c into (mean, var) forecasts.

    Returns a KL term for inclusion in the ELBO loss:
        KL[q(z|c) ‖ p(z)] = KL[N(μ_z, σ_z²) ‖ N(0, I)]
    """

    def __init__(
        self,
        d_ctx: int,
        pred_len: int,
        d_lat: int = 32,
        d_hid: int = 128,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.d_lat    = d_lat       # stored for ablation subclasses

        self.q_mu = nn.Sequential(
            nn.Linear(d_ctx, d_hid), nn.GELU(), nn.Linear(d_hid, d_lat)
        )
        self.q_lv = nn.Sequential(
            nn.Linear(d_ctx, d_hid), nn.GELU(), nn.Linear(d_hid, d_lat)
        )
        self.dec   = nn.Sequential(
            nn.Linear(d_ctx + d_lat, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_hid),         nn.GELU(),
        )
        self.mu_h  = nn.Linear(d_hid, pred_len)
        self.lv_h  = nn.Linear(d_hid, pred_len)   # log-variance head

    def forward(self, ctx: torch.Tensor):
        mu_z = self.q_mu(ctx)
        lv_z = self.q_lv(ctx)

        if self.training:
            z = mu_z + torch.randn_like(mu_z) * torch.exp(0.5 * lv_z)
        else:
            z = mu_z

        h    = self.dec(torch.cat([ctx, z], dim=-1))
        mean = self.mu_h(h)                                 # (B, P)
        var  = F.softplus(self.lv_h(h)) + 1e-6             # (B, P)

        kl = -0.5 * (1 + lv_z - mu_z.pow(2) - lv_z.exp()).sum(-1).mean()
        return mean, var, kl
