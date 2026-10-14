"""TokenDepthPredictor: loads a trained ranker + schedule and maps (hidden states, attention rows) to per-token depths."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from gazeprune.ranker import DepthRanker, combine_heads, depth_from_scores, query_features

class TokenDepthPredictor:
    """(hidden_states, attn_rows, fastv_config) -> depth (M,), at the input of layer `self.layer`."""

    def __init__(self, ckpt_path, device):
        ck = torch.load(ckpt_path, map_location="cpu")
        pca = np.load(Path(ck["feature_dir"]) / "pca.npz")
        self.mu, self.sd, self.comp = (torch.tensor(pca[k], device=device) for k in ("mean", "std", "components"))
        k = ck["args"].get("pca_dim")  # ranker trained on the first k PCA coordinates
        if k:
            self.comp = self.comp[:, :k].contiguous()
        self.two_head = bool(ck.get("two_head", False))
        self.model = DepthRanker(M=ck["M"], pca_dim=self.comp.shape[1], dropout=ck["args"]["dropout"],
                                 n_out=2 if self.two_head else 1, drop=ck["args"].get("drop") or ()).to(device).eval()
        # instruction weight at inference (OFT_HEAD_WEIGHT; default: the label mix the schedule was built with)
        self.w = float(os.environ.get("OFT_HEAD_WEIGHT") or ck.get("mix") or 0.5)
        self.model.load_state_dict(ck["model"])
        self.uses_avis = self.model.a is not None  # False: no attention rows needed (layers < m can run on SDPA)
        # the ranker is budget-free: the schedule (budget) comes from OFT_SCHEDULE (make_schedule.py); older
        # checkpoints also carry the schedule they were selected with
        min_depth = ck.get("min_depth", ck.get("schedule", {}).get("min_depth"))
        if os.environ.get("OFT_SCHEDULE"):
            self.schedule = json.load(open(os.environ["OFT_SCHEDULE"]))
        else:
            assert "schedule" in ck, "budget-free checkpoint: set OFT_SCHEDULE (make_schedule.py)"
            self.schedule = ck["schedule"]
        assert self.schedule["min_depth"] == min_depth and sum(self.schedule["n_l"]) == self.schedule["budget"]
        self.n_l = torch.tensor(self.schedule["n_l"], device=device)
        self.layer = int(self.schedule["min_depth"])
        self.M = ck["M"]

    @torch.no_grad()
    def score(self, hidden_states, attn_rows, img_start, act_start, act_end):
        """(1, M) ranking score (higher = kept in more layers)."""
        if attn_rows is None:  # ranker trained without avis: placeholder (ignored by DepthRanker)
            attn_rows = hidden_states.new_zeros(1, 1, self.M)
        hv, hq, av = query_features(hidden_states, attn_rows, img_start, self.M, act_start, act_end)
        hv = ((hv - self.mu) / self.sd) @ self.comp
        score = self.model(hv[None], hq[None], av[None])
        return combine_heads(score, self.w) if self.two_head else score

    @torch.no_grad()
    def __call__(self, hidden_states, attn_rows, img_start, act_start, act_end):
        return depth_from_scores(self.score(hidden_states, attn_rows, img_start, act_start, act_end), self.n_l)[0]

