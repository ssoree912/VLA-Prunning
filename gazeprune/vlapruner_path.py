"""Depth forward for the VLA-Pruner openvla-oft code path (used only to reproduce the pre-port runs / the VLA-Pruner
baseline environment). The official OpenVLA-OFT path is gazeprune/oft.py."""

import os
import sys
import types

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast


def register_vla_cache_stub():
    """The VLA-Pruner OFT code imports experiments.robot.vla_cache_utils, which it does not ship (only used with
    use_vla_cache=True): register a stub module so the import succeeds."""
    name = "experiments.robot.vla_cache_utils"
    if name in sys.modules:
        return
    mod = types.ModuleType(name)

    def _missing(*args, **kwargs):
        raise NotImplementedError("experiments.robot.vla_cache_utils is not shipped; run with use_vla_cache=False")

    for fn in ("find_static_patches", "task_relevant_selection", "get_layer_mask_schedule", "draw_patches_overlay"):
        setattr(mod, fn, _missing)
    sys.modules[name] = mod

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
        # The ranker needs no action history, so by default it prunes from the first query; VLA-Pruner's warm start
        # (full tokens until its attention history fills, fastv_r = 0) is kept only with OFT_WARMUP=1. With OFT's
        # 8-action chunks an episode is ~10-20 queries, so 3 full warm-start queries would add a lot of visual tokens.
        warm = float(fastv_config.get("fastv_r", 0.0)) == 0.0 if fastv_config is not None else True
        if fastv_config is None or (warm and os.environ.get("OFT_WARMUP") == "1"):
            return official(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                            return_dict=return_dict, cache_position=cache_position, fastv_config=fastv_config)
        assert input_ids is None and inputs_embeds is not None and inputs_embeds.shape[0] == 1
        out = _depth_forward(self, attention_mask, inputs_embeds, output_hidden_states, fastv_config, predictor)
        self.depth_log.append(out.pruning_info["layer_visual_counts"])
        return out

    lm.fastv_forward = types.MethodType(fastv_forward, lm)

