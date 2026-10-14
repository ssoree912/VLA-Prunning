"""
oft.py -- GazePrune on the OFFICIAL OpenVLA-OFT code (moojink/openvla-oft + transformers fork
v4.40.1-openvla-oft, the stack SpecPrune / VLA-ADP run on), instead of the VLA-Pruner openvla-oft path (vlapruner_path.py).

install_official_depth_forward(vla, predictor) replaces vla.language_model.model.forward (LlamaModel) per instance:
  * layers < predictor.layer: all tokens, eager attention with mask None (bidirectional, as the official SDPA path with
    an all-ones mask) and output_attentions=True only there, to read the action-query -> visual attention rows;
  * predictor at the input of layer predictor.layer (same features as teacher/label_oft.py);
  * layer l keeps visual tokens {i : depth_i > l} = the n_l best ranked (nested), SDPA with mask None (bidirectional,
    no attention materialization), RoPE positions of the kept tokens unchanged. Implementation: right after the
    predictor the sequence is reordered once to [non-visual tokens; visual tokens by rank], so every pruned layer keeps
    a prefix whose length comes from the host-side schedule -- no per-layer nonzero / isin and no host-GPU sync
    (with mask None and explicit position ids, attention does not depend on the token order);
  * the final normed hidden states are scattered back to the full sequence length (pruned visual rows = 0), so the
    official predict_action slices the action hidden states at its usual indices (they are never pruned).
Token layout of the official OFT prompt (L1 regression, proprio):
    [BOS][512 visual = agentview 256 + wrist 256][proprio][prompt text (incl. the '' token 29871)][56 action][stop]
-> visual = [1, 513). Action queries = the 56 positions the official action head reads, [T - 58, T - 2) (the last prompt
token + the first 55 action tokens: the hidden state at p predicts token p + 1, as in finetune.py); labels and predictor
features use the same span (teacher/label_oft.py). Pruning starts at the first query (no warm start).
predictor=None keeps the official forward (full model) but still counts queries.

Run with the official OFT repo as cwd and PYTHONPATH head and the official transformers fork on PYTHONPATH
(scripts/common.sh: oft_env).
"""

import os
import types

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

N_VISUAL = 512
N_ACTION = 56  # NUM_ACTIONS_CHUNK (8) x ACTION_DIM (7) for LIBERO


def token_layout(T):
    """(img_start, M, act_start, act_end) of the official OFT LIBERO sequence of length T; act = action-head positions.
    OFT_QUERY_SPAN=tokens: the action-token positions [T - 57, T - 1) instead (predictors trained on the VLA-Pruner-path
    labels, label_rlds.py)."""
    if os.environ.get("OFT_QUERY_SPAN") == "tokens":
        return 1, N_VISUAL, T - 1 - N_ACTION, T - 1
    return 1, N_VISUAL, T - 2 - N_ACTION, T - 2


def depth_forward_official(self, inputs_embeds, predictor, output_hidden_states=False):
    """LlamaModel forward (batch 1, no cache) with the per-token depth schedule of `predictor`."""
    device = inputs_embeds.device
    T = inputs_embeds.shape[1]
    img_start, M, act_start, act_end = token_layout(T)
    assert M == predictor.M
    m, n_l = predictor.layer, predictor.schedule["n_l"]  # host-side schedule: n_l[l] visual tokens in layer l
    position_ids = torch.arange(T, device=device)[None]
    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    rows, n_fix, cur = [], T - M, M
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if layer_idx == m:
            score = predictor.score(hidden_states, torch.cat(rows, dim=1) if rows else None, img_start, act_start,
                                    act_end)[0]
            order = torch.sort(-score, stable=True).indices  # rank -> visual index (ties: smaller index first)
            perm = torch.cat((torch.arange(img_start, device=device), torch.arange(img_start + M, T, device=device),
                              order + img_start))
            hidden_states, position_ids = hidden_states[:, perm], perm[None]
        if layer_idx >= m and n_l[layer_idx] < cur:  # keep the n_l best ranked visual tokens: a prefix
            cur = n_l[layer_idx]
            hidden_states, position_ids = hidden_states[:, :n_fix + cur], position_ids[:, :n_fix + cur]
        # only the layers before the predictor materialize attention (eager), and only if the ranker uses avis
        need_attn = layer_idx < m and predictor.uses_avis
        out = decoder_layer(hidden_states, attention_mask=None, position_ids=position_ids, past_key_value=None,
                            output_attentions=need_attn, use_cache=False, cache_position=None)
        hidden_states = out[0]
        if need_attn:
            rows.append(out[1][:, :, act_start:act_end, img_start:img_start + M].float().mean(dim=2))
    hidden_states = self.norm(hidden_states)
    full = hidden_states.new_zeros(1, T, hidden_states.shape[-1])
    full[:, position_ids[0]] = hidden_states  # back to the original positions (pruned visual rows = 0)
    hidden_states = full
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(M, device=device)
    self.depth_info = {"visual_depth": (predictor.n_l[None, :] > rank[:, None]).sum(-1),  # == depth_from_scores
                       "layer_visual_counts": [min(M, n) if l >= m else M for l, n in enumerate(n_l)]}
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None,
                                   hidden_states=all_hidden_states, attentions=None)


def install_official_depth_forward(vla, predictor):
    """Per-instance override of the official LlamaModel.forward; predictor=None = official forward (full model)."""
    lm = vla.language_model.model
    official = lm.forward  # bound method of the class
    lm.depth_log, lm.n_queries = [], 0

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                cache_position=None):
        self.n_queries += 1
        if predictor is None or output_attentions:
            return official(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                            return_dict=return_dict, cache_position=cache_position)
        assert input_ids is None and inputs_embeds is not None and inputs_embeds.shape[0] == 1
        assert past_key_values is None and position_ids is None
        assert attention_mask is None or bool(attention_mask.all()), "padding is not supported"
        out = depth_forward_official(self, inputs_embeds, predictor, output_hidden_states=bool(output_hidden_states))
        self.depth_log.append(self.depth_info["layer_visual_counts"])
        return out

    lm.forward = types.MethodType(forward, lm)
