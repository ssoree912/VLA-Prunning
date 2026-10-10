"""
core.py -- OpenVLA-OFT version of the depth-predictor pipeline (see oracle_pruning/plan/Oracle_AB_labels.md).

OFT predicts the whole action chunk in ONE forward: the 56 (= 8 steps x 7 dims) zero-embedded action positions are
the action queries. With attentions requested (as in every OFT inference path) the mask is None and the eager
attention is bidirectional, so visual tokens also see the instruction. Teacher label: s[l, i] = mean over heads and the 56 action queries of the attention that layer l
gives to visual token i (M = 512 = 2 cameras x 256). No decode, no KV cache.

Contents
  oracle_depth_dp / round_schedule        -- exact DP (budget, min depth) and fixed-schedule rounding
  DepthRanker / depth_from_scores         -- the learned token ranker and rank -> depth under a fixed schedule
  TokenDepthPredictor                     -- callable used inside the forward at the input of layer `layer`
  install_depth_forward(vla, predictor)   -- per-instance override of LlamaModel.fastv_forward: warm-start queries
                                             (fastv_r == 0) use the official path, others the depth forward
"""

import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast


# ----------------------------------------------------------------------------------------------- DP and schedule
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


# ------------------------------------------------------------------------------------------------------ ranker
class DepthRanker(nn.Module):
    """Per-token MLP over [visual hidden (PCA) ; action-query summary ; query->token attention (layers < m) ; position]."""

    def __init__(self, M=512, pca_dim=512, text_dim=4096, n_attn=64, d=256, d_attn=64, d_pos=64, hidden=512, dropout=0.1):
        super().__init__()
        self.M = M
        self.v = nn.Linear(pca_dim, d)
        self.t = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
        self.a = nn.Linear(n_attn, d_attn)
        self.pos = nn.Embedding(M, d_pos)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d + d_attn + d_pos, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, hv, hq, avis):
        """hv (B, M, P), hq (B, 4096), avis (B, n_attn, M) -> logits (B, M)."""
        B, M = hv.shape[0], hv.shape[1]
        t = self.t(hq)[:, None].expand(-1, M, -1)
        a = self.a(torch.log(avis.transpose(1, 2) + 1e-6))
        p = self.pos.weight[None].expand(B, -1, -1)
        return self.mlp(torch.cat([self.v(hv), t, a, p], dim=-1)).squeeze(-1)


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


class TokenDepthPredictor:
    """(hidden_states, attn_rows, fastv_config) -> depth (M,), at the input of layer `self.layer`."""

    def __init__(self, ckpt_path, device):
        ck = torch.load(ckpt_path, map_location="cpu")
        pca = np.load(Path(ck["feature_dir"]) / "pca.npz")
        self.mu, self.sd, self.comp = (torch.tensor(pca[k], device=device) for k in ("mean", "std", "components"))
        self.model = DepthRanker(M=ck["M"], pca_dim=self.comp.shape[1], dropout=ck["args"]["dropout"]).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.schedule = ck["schedule"]
        self.n_l = torch.tensor(self.schedule["n_l"], device=device)
        self.layer = int(self.schedule["min_depth"])
        self.M = ck["M"]

    @torch.no_grad()
    def __call__(self, hidden_states, attn_rows, img_start, act_start, act_end):
        hv, hq, av = query_features(hidden_states, attn_rows, img_start, self.M, act_start, act_end)
        hv = ((hv - self.mu) / self.sd) @ self.comp
        score = self.model(hv[None], hq[None], av[None])
        return depth_from_scores(score, self.n_l)[0]


# --------------------------------------------------------------------------------------------- depth forward
def _depth_forward(self, attention_mask, inputs_embeds, output_hidden_states, fastv_config, predictor):
    """Single forward, no cache. Layers < predictor.layer see all tokens; then layer l keeps visual {i : depth_i > l}."""
    device = inputs_embeds.device
    img_start = int(fastv_config["image_token_start_index"])
    M = int(fastv_config["image_token_length"])
    act_start, act_end = int(fastv_config["action_token_start"]), int(fastv_config["action_token_end"])
    assert M == predictor.M and act_end - act_start > 0
    T = inputs_embeds.shape[1]
    cache_position = torch.arange(T, device=device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, 0, False)
    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    cur_index = torch.arange(T, device=device)
    depth, rows, n_visual, counts, first_layer = None, [], M, [], None
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if depth is None and layer_idx == predictor.layer:
            depth = predictor(hidden_states, torch.cat(rows, dim=1), img_start, act_start, act_end).to(device)
            assert int(depth.min()) >= predictor.layer
        if depth is not None:
            kept_visual = torch.nonzero(depth > layer_idx, as_tuple=False).squeeze(-1)
            if len(kept_visual) < n_visual:
                keep = torch.cat((torch.arange(img_start, device=device), kept_visual + img_start,
                                  torch.arange(img_start + M, T, device=device)))
                hidden_states = hidden_states[:, torch.isin(cur_index, keep)]
                cur_index = keep
                position_ids = keep.unsqueeze(0)
                cache_position = torch.arange(len(keep), device=device)
                causal_mask = self._update_causal_mask(None, hidden_states, cache_position, 0, False)
                n_visual = len(kept_visual)
                first_layer = layer_idx if first_layer is None else first_layer
        counts.append(n_visual)
        # output_attentions=True in every layer: in this fork the mask is dropped (None) and SDPA would then run
        # causally, while the eager path used by every OFT inference (vanilla and VLA-Pruner) is bidirectional.
        out = decoder_layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids, past_key_value=None,
                            output_attentions=True, use_cache=False, cache_position=cache_position)
        hidden_states = out[0]
        if depth is None:
            rows.append(out[1][:, :, act_start:act_end, img_start:img_start + M].float().mean(dim=2))
    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    all_idx = torch.arange(T, device=device)
    self.pruning_info = {
        "original_seq_length": T, "kept_indices": cur_index, "pruned_indices": all_idx[~torch.isin(all_idx, cur_index)],
        "pruning_layer": first_layer, "mode": "depth_predictor", "visual_depth": depth,
        "layer_visual_counts": counts,
    }
    output = BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None,
                                     hidden_states=all_hidden_states, attentions=None)
    output.pruning_info = self.pruning_info
    return output


def install_depth_forward(vla, predictor):
    """Route pruned queries of the OFT VLA-Pruner path (fastv_forward with fastv_r > 0) to the depth forward."""
    lm = vla.language_model.model
    official = lm.fastv_forward
    lm.depth_log = []

    def fastv_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                      inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                      return_dict=None, cache_position=None, fastv_config=None):
        if fastv_config is None or float(fastv_config.get("fastv_r", 0.0)) == 0.0:  # warm start: full tokens
            return official(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                            return_dict=return_dict, cache_position=cache_position, fastv_config=fastv_config)
        assert input_ids is None and inputs_embeds is not None and inputs_embeds.shape[0] == 1
        out = _depth_forward(self, attention_mask, inputs_embeds, output_hidden_states, fastv_config, predictor)
        self.depth_log.append(out.pruning_info["layer_visual_counts"])
        return out

    lm.fastv_forward = types.MethodType(fastv_forward, lm)
