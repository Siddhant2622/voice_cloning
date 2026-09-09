"""
src/models/mamba_ssm.py
=======================
Lightweight pure-PyTorch State Space Model (S4/Mamba-lite) for sequential
frame-level audio modeling.

Inspired by Resemble AI DETECT-2B's use of Mamba-SSM for efficient
temporal sequence modeling over per-frame SSL features.

This implementation uses ONLY standard torch.nn operations — no Mamba CUDA
extension required. Falls back gracefully on CPU.

Architecture:
  - S4-Lite: simplified selective state space model
  - Each block: LayerNorm → SSM scan → skip connection → FFN
  - Input/Output shape: [B, T, D] (batch, time-frames, feature-dim)

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
State Spaces" (2023), arXiv:2312.00752
"""

from __future__ import annotations

import math
import logging

logger = logging.getLogger(__name__)


def _torch():
    import torch
    return torch


def _nn():
    import torch.nn as nn
    return nn


class S4LiteBlock:
    """
    A single pure-PyTorch S4-lite (simplified SSM) block.
    Implements efficient recurrent state modeling without CUDA extensions.

    The selective state space operation is approximated via:
      1. Input projection to compute delta, B, C gates
      2. Selective scan via cumulative convolution (causal)
      3. Output projection

    Input:  [B, T, D]
    Output: [B, T, D]
    """

    def __new__(cls, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _S4Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.d_model = d_model
                self.d_inner = d_model * expand
                self.d_state = d_state

                # Input projection
                self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

                # Short depthwise conv for local context
                self.conv1d = nn.Conv1d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    kernel_size=d_conv,
                    padding=d_conv - 1,
                    groups=self.d_inner,
                    bias=True,
                )

                # SSM parameters
                self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
                self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

                # A matrix (log-initialized)
                A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
                self.A_log = nn.Parameter(torch.log(A))

                # D skip connection scalar
                self.D = nn.Parameter(torch.ones(self.d_inner))

                # Output projection
                self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

                # Normalization
                self.norm = nn.LayerNorm(d_model)

            def ssm_scan(self, x):
                """Simplified selective state space scan. x: [B, T, D_inner]"""
                B, T, D = x.shape
                A = -torch.exp(self.A_log.float())  # [D, N]

                # Project to get delta, B_ssm, C_ssm
                x_dbl = self.x_proj(x)  # [B, T, N*2 + D]
                dt_rank = D
                delta, B_ssm, C_ssm = x_dbl.split([dt_rank, self.d_state, self.d_state], dim=-1)

                delta = F.softplus(self.dt_proj(delta))  # [B, T, D]

                # Discretize A, B using ZOH
                # A_bar: [B, T, D, N]
                A_bar = torch.exp(
                    delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
                )  # [B, T, D, N]
                B_bar = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)  # [B, T, D, N]

                # Sequential scan via cumulative product (efficient on GPU)
                # h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]
                # y[t] = C[t] @ h[t] + D * x[t]
                h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
                ys = []
                for t in range(T):
                    h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)  # [B, D, N]
                    y = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)  # [B, D]
                    ys.append(y)
                y = torch.stack(ys, dim=1)  # [B, T, D]
                return y + x * self.D.unsqueeze(0).unsqueeze(0)

            def forward(self, x):
                # x: [B, T, D_model]
                residual = x
                x = self.norm(x)

                # Gated input projection
                xz = self.in_proj(x)  # [B, T, D_inner * 2]
                x_branch, z = xz.chunk(2, dim=-1)  # [B, T, D_inner] each

                # Depthwise conv for local context
                x_branch = x_branch.transpose(1, 2)  # [B, D_inner, T]
                x_branch = self.conv1d(x_branch)[:, :, :x.shape[1]]
                x_branch = x_branch.transpose(1, 2)  # [B, T, D_inner]
                x_branch = F.silu(x_branch)

                # SSM scan
                y = self.ssm_scan(x_branch)

                # Gating
                y = y * F.silu(z)

                # Output projection
                y = self.out_proj(y)
                return y + residual

        return _S4Block()


class MambaEncoder:
    """
    Stack of S4-lite blocks for full temporal encoding of SSL frame features.

    Usage:
        encoder = MambaEncoder(d_model=512, n_layers=4)
        out = encoder(x)  # [B, T, 512]
        pooled = out.mean(dim=1)  # [B, 512] global pool
    """

    def __new__(cls, d_model: int = 512, n_layers: int = 4, d_state: int = 16):
        import torch.nn as nn

        class _MambaEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    S4LiteBlock(d_model=d_model, d_state=d_state)
                    for _ in range(n_layers)
                ])
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x):
                # x: [B, T, d_model]
                for layer in self.layers:
                    x = layer(x)
                return self.norm(x)

        return _MambaEncoder()
