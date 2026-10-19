"""Toy models of superposition, after Elhage et al. (2022), plus the variant
described in the appendix of "Neural Networks Are Already Clean" (correlated
input groups, a bottleneck, and a hidden ReLU layer).

Everything here runs on CPU in seconds to a minute.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def sample_independent(batch: int, n: int, sparsity: float, gen: torch.Generator) -> torch.Tensor:
    """Each feature is present with probability (1 - sparsity), amplitude U[0,1]."""
    present = torch.rand(batch, n, generator=gen) >= sparsity
    amp = torch.rand(batch, n, generator=gen)
    return present.float() * amp


def sample_grouped(batch: int, n_groups: int, group_size: int, p_group: float,
                   gen: torch.Generator) -> torch.Tensor:
    """Correlated inputs: each group of `group_size` inputs turns on together
    with probability p_group; within an active group every member gets an
    independent U[0,1] amplitude."""
    on = (torch.rand(batch, n_groups, 1, generator=gen) < p_group).float()
    amp = torch.rand(batch, n_groups, group_size, generator=gen)
    return (on * amp).reshape(batch, n_groups * group_size)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class TMS(torch.nn.Module):
    """x -> h = W x -> x_hat = ReLU(W^T h + b).  W is (m, n)."""

    def __init__(self, n: int, m: int, gen: torch.Generator):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(m, n, generator=gen) / np.sqrt(m))
        self.b = torch.nn.Parameter(torch.zeros(n))

    def forward(self, x):
        h = x @ self.W.T
        return F.relu(h @ self.W + self.b)


class TMSHidden(torch.nn.Module):
    """The essay's variant: bottleneck, then a layer of hidden ReLU neurons.
    x -> h = W x (m dims) -> a = ReLU(V h + c) (k neurons) -> x_hat = U a + b."""

    def __init__(self, n: int, m: int, k: int, gen: torch.Generator):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(m, n, generator=gen) / np.sqrt(m))
        self.V = torch.nn.Parameter(torch.randn(k, m, generator=gen) / np.sqrt(m))
        self.c = torch.nn.Parameter(torch.zeros(k))
        self.U = torch.nn.Parameter(torch.randn(n, k, generator=gen) / np.sqrt(k))
        self.b = torch.nn.Parameter(torch.zeros(n))

    def hidden(self, x):
        h = x @ self.W.T
        return F.relu(h @ self.V.T + self.c)

    def forward(self, x):
        return self.hidden(x) @ self.U.T + self.b


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(model: torch.nn.Module, sampler, importance: torch.Tensor | None = None,
          steps: int = 5000, batch: int = 1024, lr: float = 1e-3, seed: int = 0):
    """Minimise sum_i I_i (x_i - x_hat_i)^2 with Adam and a cosine lr decay."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    for step in range(steps):
        x = sampler(batch, gen)
        xh = model(x)
        err = (x - xh) ** 2
        if importance is not None:
            err = err * importance
        loss = err.sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return np.array(losses)


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def feature_norms(W: torch.Tensor) -> np.ndarray:
    """||W_i|| for each feature column of W (m, n)."""
    return W.detach().norm(dim=0).numpy()


def superposition_index(W: torch.Tensor) -> np.ndarray:
    """sum_{j != i} (W_i_hat . W_j)^2: how much other features overlap with
    feature i's direction. Zero means feature i has a private direction."""
    W = W.detach()
    norms = W.norm(dim=0, keepdim=True).clamp_min(1e-8)
    What = W / norms
    G = (What.T @ W) ** 2            # (n, n): (W_i_hat . W_j)^2
    G.fill_diagonal_(0.0)
    return G.sum(1).numpy()


def feature_dimensionality(W: torch.Tensor) -> np.ndarray:
    """D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2, from the paper. 1 means a
    dedicated dimension; 1/2 an antipodal pair; 0 not represented."""
    W = W.detach()
    norms = W.norm(dim=0).clamp_min(1e-8)
    What = W / norms
    denom = ((What.T @ W) ** 2).sum(1)
    return (norms ** 2 / denom.clamp_min(1e-8)).numpy()


def cosine_matrix(W: torch.Tensor) -> np.ndarray:
    W = W.detach()
    What = W / W.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return (What.T @ What).numpy()
