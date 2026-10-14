"""Token ranker: a per-token MLP that scores the visual tokens at the input of layer m (= 2).

Inputs per visual token i: its layer-(m-1) hidden state (standardized + PCA), a summary of the action queries (mean
hidden state), optionally the action-query attention to token i in layers < m (avis), and its position. One output
(action head) or two (action / instruction heads, mixed at inference with weight w).
"""

import torch
import torch.nn as nn

class DepthRanker(nn.Module):
    """Per-token MLP over [visual hidden (PCA) ; action-query summary ; query->token attention (layers < m) ; position]."""

    FEATURES = ("hv", "hq", "avis", "pos")

    def __init__(self, M=512, pca_dim=512, text_dim=4096, n_attn=64, d=256, d_attn=64, d_pos=64, hidden=512, dropout=0.1,
                 n_out=1, drop=()):
        super().__init__()
        assert set(drop) <= set(self.FEATURES) and len(drop) < len(self.FEATURES), drop
        self.M, self.n_out, self.drop = M, n_out, tuple(drop)  # drop: input features left out (ablation)
        self.v = nn.Linear(pca_dim, d) if "hv" not in drop else None
        self.t = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d)) if "hq" not in drop else None
        self.a = nn.Linear(n_attn, d_attn) if "avis" not in drop else None
        self.pos = nn.Embedding(M, d_pos) if "pos" not in drop else None
        d_in = sum(n for f, n in zip(self.FEATURES, (d, d, d_attn, d_pos)) if f not in drop)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),  # n_out = 2: action / instruction heads on a shared trunk (two-head ranker)
        )

    def forward(self, hv, hq, avis):
        """hv (B, M, P), hq (B, 4096), avis (B, n_attn, M) -> logits (B, M), or (B, M, n_out) for n_out > 1."""
        B, M = hv.shape[0], hv.shape[1]
        x = []
        if self.v is not None:
            x.append(self.v(hv))
        if self.t is not None:
            x.append(self.t(hq)[:, None].expand(-1, M, -1))
        if self.a is not None:
            x.append(self.a(torch.log(avis.transpose(1, 2) + 1e-6)))
        if self.pos is not None:
            x.append(self.pos.weight[None].expand(B, -1, -1))
        out = self.mlp(torch.cat(x, dim=-1))
        return out.squeeze(-1) if self.n_out == 1 else out


def combine_heads(logits, w):
    """Two-head ranker: (B, M, 2) action / instruction logits -> (1 - w) softmax(act) + w softmax(instr), as the label."""
    return (1 - w) * torch.softmax(logits[..., 0], dim=-1) + w * torch.softmax(logits[..., 1], dim=-1)


def depth_from_scores(score, n_l):
    """score (N, M) -> depth (N, M): rank j keeps #{l : n_l > j} layers (ties -> smaller index)."""
    order = torch.sort(-score, dim=1, stable=True).indices
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(score.shape[1], device=score.device).expand_as(order))
    return (n_l[None, None, :] > rank[..., None]).sum(-1)


def query_features(hidden_states, attn_rows, img_start, M, act_start, act_end):
    """Predictor inputs from the full sequence at the input of the predictor layer.
    hidden_states (1, T, 4096); attn_rows (1, n_attn, M) = action-query attention to visual tokens in earlier layers."""
    h = hidden_states[0].float()
    return h[img_start:img_start + M], h[act_start:act_end].mean(0), attn_rows[0].float()

