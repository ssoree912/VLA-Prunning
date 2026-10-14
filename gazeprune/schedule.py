"""Per-token depth oracle (exact DP under a token-layer budget) and fixed per-layer schedules.

GazePrune keeps visual token i in layers 0 .. d_i - 1. Given per-layer teacher scores s[l, i] (attention the action
queries pay to visual token i in layer l), the oracle depth maximizes sum_i sum_{l < d_i} s[l, i] subject to
sum_i d_i = budget and d_i >= min_depth. A fixed schedule n_l (visual tokens kept in layer l) is the per-layer mean of
the oracle counts over training frames, rounded to an integer sum; at inference the ranker's top-n_l tokens are kept.
"""

from collections import defaultdict

import numpy as np
import torch


def oracle_depth_dp(scores: torch.Tensor, budget: int, min_depth: int = 0):
    """max sum_i sum_{l<d_i} scores[l, i] s.t. sum_i d_i = budget, d_i in [min_depth, L]. Returns (depth, objective)."""
    s = scores.detach().double().cpu()
    L, M = s.shape
    assert 0 <= min_depth <= L and min_depth * M <= budget <= L * M
    U = torch.cat([torch.zeros(M, 1, dtype=torch.float64), s.T.cumsum(dim=1)], dim=1)
    neg = float("-inf")
    U[:, :min_depth] = neg
    F = torch.full((budget + 1,), neg, dtype=torch.float64)
    F[0] = 0.0
    idx = torch.arange(budget + 1)[None, :] - torch.arange(L + 1)[:, None] + L
    choice = torch.empty(M, budget + 1, dtype=torch.long)
    pad = torch.full((L,), neg, dtype=torch.float64)
    for i in range(M):
        F, choice[i] = (torch.cat([pad, F])[idx] + U[i][:, None]).max(dim=0)
    depth = torch.empty(M, dtype=torch.long)
    b = budget
    for i in range(M - 1, -1, -1):
        depth[i] = choice[i, b]
        b -= int(depth[i])
    assert b == 0
    return depth, float(F[budget])


def round_schedule(mean_counts, budget):
    """Largest-remainder rounding to an integer sum, then sorted non-increasing (keeps the sum)."""
    n = np.floor(mean_counts).astype(int)
    for j in np.argsort(-(mean_counts - n), kind="stable")[: budget - n.sum()]:
        n[j] += 1
    return np.sort(n)[::-1]




def objective(s, depth):
    keep = torch.arange(s.shape[1], device=s.device)[None, :, None] < depth[:, None, :]
    return (s * keep).sum(dim=(1, 2))


def split(files, val_frac, seed):
    by_task = defaultdict(list)
    for f in files:
        by_task[str(np.load(f, mmap_mode="r")["instruction"])].append(f)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for _, fs in sorted(by_task.items()):
        fs = list(rng.permutation(fs))
        n_val = max(1, round(val_frac * len(fs)))
        val += fs[:n_val]
        train += fs[n_val:]
    return train, val


def load_labels(files, keys, pca_dim=None):
    """pca_dim: keep only the first pca_dim PCA coordinates of hv (stored sorted by explained variance)."""
    out = defaultdict(list)
    for f in files:
        d = np.load(f)
        for k in keys:
            out[k].append(d[k][..., :pca_dim] if (k == "hv" and pca_dim) else d[k])
    return {k: torch.from_numpy(np.concatenate(v)) for k, v in out.items()}


def mixed(s_act, s_instr, mix, m):
    """Per-layer mixed score (as the base OpenVLA mixed label): each source normalized over layers >= m."""
    sa = s_act / s_act[:, m:].sum((1, 2), keepdims=True)
    si = s_instr / s_instr[:, m:].sum((1, 2), keepdims=True)
    return (1 - mix) * sa + mix * si


def make_schedule(files, budget, min_depth, n_frames, seed, mix=None):
    rng = np.random.default_rng(seed)
    per_file = max(1, n_frames // len(files) + 1)
    counts = []
    for f in files:
        d = np.load(f)
        s = d["s_layer"] if mix is None else mixed(d["s_layer"].astype(np.float64), d["s_instr"].astype(np.float64),
                                                   mix, min_depth)
        for t in rng.choice(len(s), min(per_file, len(s)), replace=False):
            depth, _ = oracle_depth_dp(torch.from_numpy(s[t].astype(np.float64)), budget, min_depth)
            counts.append((depth.numpy()[None, :] > np.arange(s.shape[1])[:, None]).sum(1))
            if len(counts) >= n_frames:
                break
        if len(counts) >= n_frames:
            break
    counts = np.stack(counts)
    n = round_schedule(counts.mean(0), budget)
    assert n.sum() == budget and (np.diff(n) <= 0).all()
    return n, float(counts.std(0)[min_depth:].mean()), len(counts)


def budget_from_fastv(L, M, fastv_k, fastv_r):
    """VLA-Pruner-style token-layer budget: K+1 full layers, then k = round(M (1 - fastv_r)) tokens per layer."""
    k = int(round(M * (1 - fastv_r)))
    return (fastv_k + 1) * M + (L - fastv_k - 1) * k
