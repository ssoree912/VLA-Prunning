"""
oracle_predictor.py -- learned visual-token ranker for Oracle B-fixed depths (no teacher pass, no DP).

The ranker runs inside the prefill at the output of layer `layer` (= 2: layers 0 and 1 see all visual tokens). It
scores the M visual tokens from (i) their hidden states (standardized + PCA), (ii) the hidden state of the last prompt
token, (iii) that token's attention to each visual token in layers 0..layer-1, (iv) the patch position. The token of
rank j keeps depth #{l : n_l > j} under the fixed per-layer schedule n_l. Trained by oracle_scripts/train_predictor.py.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

M = 256


class DepthRanker(nn.Module):
    """
    Per-token MLP over [visual hidden (PCA) ; context summaries ; attention to token i (layers < m) ; position].
    use_last:  last prompt token hidden (hlast) + its attention to token i (avis)  -- the first action query
    use_instr: mean instruction-token hidden (hins) + instruction-token attention to token i (ains)
    Defaults (use_last=True, use_instr=False) are the original model, so older checkpoints load unchanged.
    drop: attention inputs left out ("avis", "ains"), as the OFT ranker (oft_oracle/core.py); the stored features may
    then lack them (the avis / ains arrays were removed from the feature files to save disk).
    """

    def __init__(self, pca_dim=512, text_dim=4096, n_attn=64, d=256, d_attn=64, d_pos=64, hidden=512, dropout=0.1,
                 use_last=True, use_instr=False, n_out=1, drop=()):
        super().__init__()
        assert use_last or use_instr
        assert set(drop) <= {"avis", "ains"}, drop
        self.use_last, self.use_instr = use_last, use_instr
        self.use_avis, self.use_ains = use_last and "avis" not in drop, use_instr and "ains" not in drop
        self.v = nn.Linear(pca_dim, d)
        if use_last:
            self.t = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
            if self.use_avis:
                self.a = nn.Linear(n_attn, d_attn)
        if use_instr:
            self.s = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
            if self.use_ains:
                self.b = nn.Linear(n_attn, d_attn)
        self.pos = nn.Embedding(M, d_pos)
        d_in = d + d_pos + d * (int(use_last) + int(use_instr)) + d_attn * (int(self.use_avis) + int(self.use_ains))
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )
        self.n_out = n_out

    def forward(self, hv, hlast=None, avis=None, hins=None, ains=None):
        """hv (B, M, P); hlast/hins (B, 4096); avis/ains (B, 64, M) -> logits (B, M)."""
        B = hv.shape[0]
        parts = [self.v(hv)]
        if self.use_last:
            parts.append(self.t(hlast)[:, None].expand(-1, M, -1))
            if self.use_avis:
                parts.append(self.a(torch.log(avis.transpose(1, 2) + 1e-6)))
        if self.use_instr:
            parts.append(self.s(hins)[:, None].expand(-1, M, -1))
            if self.use_ains:
                parts.append(self.b(torch.log(ains.transpose(1, 2) + 1e-6)))
        parts.append(self.pos.weight[None].expand(B, -1, -1))
        out = self.mlp(torch.cat(parts, dim=-1))
        return out.squeeze(-1) if self.n_out == 1 else out  # (B, M) or (B, M, n_out)


def combine_heads(logits, w):
    """Two-head ranker: (B, M, 2) action / instruction logits -> (1 - w) softmax(act) + w softmax(instr), as the label."""
    return (1 - w) * torch.softmax(logits[..., 0], dim=-1) + w * torch.softmax(logits[..., 1], dim=-1)


def depth_from_scores(score, n_l):
    """score (N, M) -> depth (N, M): rank j keeps #{l : n_l > j} layers (ties -> smaller index)."""
    order = torch.sort(-score, dim=1, stable=True).indices
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(score.shape[1], device=score.device).expand_as(order))
    return (n_l[None, None, :] > rank[..., None]).sum(-1)


class TokenDepthPredictor:
    """Callable used by the depth prefill: (hidden_states (1, T, 4096), attn_rows (1, 64, M)) -> depth (M,)."""

    def __init__(self, ckpt_path, device):
        ck = torch.load(ckpt_path, map_location="cpu")
        feat_dir = Path(f"/workspace/vla/data/oracle_features/{ck['args']['suite']}")
        pca = np.load(feat_dir / "pca.npz")
        self.mu, self.sd, self.comp = (torch.tensor(pca[k], device=device) for k in ("mean", "std", "components"))
        k = ck["args"].get("pca_dim")  # ranker trained on the first k PCA coordinates (stored sorted by variance)
        if k:
            self.comp = self.comp[:, :k].contiguous()
        self.two_head = bool(ck.get("two_head", False))
        self.head_weight = float(os.environ.get("ORACLE_HEAD_WEIGHT", ck.get("head_weight", 0.5)))
        self.model = DepthRanker(pca_dim=self.comp.shape[1], dropout=ck["args"]["dropout"],
                                 n_out=2 if self.two_head else 1, drop=ck["args"].get("drop") or ()).to(device).eval()
        self.uses_avis = self.model.use_avis  # False: the attention rows of the layers before the ranker are unused
        self.model.load_state_dict(ck["model"])
        self.schedule = ck["schedule"]
        if os.environ.get("ORACLE_SCHEDULE"):  # e.g. a schedule recomputed for the inference weight w
            override = json.load(open(os.environ["ORACLE_SCHEDULE"]))
            # the ranker is budget-agnostic: any budget works as long as the predictor layer (min depth) is the same
            assert override["min_depth"] == self.schedule["min_depth"] and sum(override["n_l"]) == override["budget"]
            self.schedule = override
        self.n_l = torch.tensor(self.schedule["n_l"], device=device)
        self.layer = int(self.schedule["min_depth"])  # predictor input = output of layer `layer - 1`

    @torch.no_grad()
    def __call__(self, hidden_states, attn_rows, img_start):
        h = hidden_states[0].float()
        hv = ((h[img_start:img_start + M] - self.mu) / self.sd) @ self.comp
        score = self.model(hv[None], h[-1][None], attn_rows.float() if self.uses_avis else None)
        if self.two_head:
            score = combine_heads(score, self.head_weight)
        return depth_from_scores(score, self.n_l)[0]
