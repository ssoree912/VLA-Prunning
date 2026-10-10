"""
oracle_llama.py -- oracle prefills for the VLA-Pruner LlamaModel.

The installed (shared) transformers is not modified. `install_oracle_prefill(llama_model)` overrides `fastv_forward`
(and wraps `forward`) on that one LlamaModel instance:
  * `fastv_config['oracle_keep_visual']` (Oracle A / Oracle-Act-TopK): keep exactly those visual tokens from layer K
    on (no semantic candidates, no MMDP).
  * `fastv_config['oracle_depth']` (Oracle B): visual token i enters layers 0..depth[i]-1 only, so the kept set can
    shrink before any layer and a removed token never comes back.
  * otherwise: the official implementation.
Pruning mechanics (hidden-state gather, original position ids, rebuilt causal mask, truncated cache_position,
`pruning_info`) mirror the official `use_cache` branch so the decode steps behave identically to VLA-Pruner.
"""

import types
from typing import Optional

import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast


def oracle_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k highest scores; ties broken by the smaller original index. Returned in ascending order."""
    order = torch.sort(-scores.float(), stable=True).indices
    return order[:k].sort().values


def oracle_depth_dp(scores: torch.Tensor, budget: int, min_depth: int = 0):
    """
    Oracle B depth allocation. scores: (L, M) teacher KV-attention score of visual token i at layer l.
    Returns (depth (M,) long in [min_depth, L], objective) maximizing sum_i sum_{l < depth_i} scores[l, i]
    subject to sum_i depth_i == budget, by exact DP over (token, budget used). Ties -> smaller depth.
    min_depth = m forbids removing any visual token before layer m.
    """
    s = scores.detach().double().cpu()
    L, M = s.shape
    assert 0 <= min_depth <= L and min_depth * M <= budget <= L * M, f"budget {budget} infeasible for min_depth {min_depth}"
    U = torch.cat([torch.zeros(M, 1, dtype=torch.float64), s.T.cumsum(dim=1)], dim=1)  # U[i, d] = sum_{l<d} s[l, i]
    neg = float("-inf")
    U[:, :min_depth] = neg
    F = torch.full((budget + 1,), neg, dtype=torch.float64)
    F[0] = 0.0
    # shifted[d, b] = F[b - d] (-inf when b < d), via a -inf left pad of width L
    idx = torch.arange(budget + 1)[None, :] - torch.arange(L + 1)[:, None] + L
    choice = torch.empty(M, budget + 1, dtype=torch.long)
    for i in range(M):
        cand = torch.cat([torch.full((L,), neg, dtype=torch.float64), F])[idx] + U[i][:, None]
        F, choice[i] = cand.max(dim=0)
    assert torch.isfinite(F[budget]), "infeasible budget"
    depth = torch.empty(M, dtype=torch.long)
    b = budget
    for i in range(M - 1, -1, -1):
        depth[i] = choice[i, b]
        b -= int(depth[i])
    assert b == 0 and int(depth.sum()) == budget
    return depth, float(F[budget])


def oracle_schedule_depth(scores: torch.Tensor, n_l: torch.Tensor):
    """
    Oracle B-fixed depths. n_l: (L,) non-increasing per-layer visual counts with n_l[0] <= M. With m = #{l : n_l = M}
    leading full layers, tokens are ranked by r_i = sum_{l >= m} scores[l, i] (ties -> smaller index) and the token of
    rank j keeps depth #{l : n_l > j}, so layer l keeps exactly the top n_l tokens (nested). Returns (depth, m).
    """
    s = scores.detach().double().cpu()
    L, M = s.shape
    n_l = n_l.cpu().long()
    assert n_l.shape == (L,) and int(n_l[0]) <= M and bool((n_l[1:] <= n_l[:-1]).all())
    m = int((n_l == M).sum())
    order = torch.sort(-s[m:].sum(dim=0), stable=True).indices
    rank = torch.empty(M, dtype=torch.long)
    rank[order] = torch.arange(M)
    depth = (n_l[None, :] > rank[:, None]).sum(dim=1)
    return depth, m


def _oracle_prefill(
    self,
    attention_mask: Optional[torch.Tensor],
    past_key_values,
    inputs_embeds: torch.Tensor,
    use_cache: bool,
    output_attentions: bool,
    output_hidden_states: bool,
    return_dict: bool,
    fastv_config: dict,
):
    K = fastv_config['fastv_k']
    img_start = fastv_config['image_token_start_index']
    img_len = fastv_config['image_token_length']
    keep_visual = fastv_config['oracle_keep_visual'].to(inputs_embeds.device)
    assert use_cache and inputs_embeds.shape[1] > 1, "oracle prefill expects the cached multimodal prefill call"
    assert keep_visual.dim() == 1 and len(torch.unique(keep_visual)) == len(keep_visual), "duplicate oracle indices"
    assert int(keep_visual.min()) >= 0 and int(keep_visual.max()) < img_len

    past_seen_tokens = 0
    if not isinstance(past_key_values, StaticCache):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_seen_tokens = past_key_values.get_seq_length()
    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, past_seen_tokens)
    seq_length_with_past = past_seen_tokens + inputs_embeds.shape[1]
    device = inputs_embeds.device

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    pruning_info = {
        'original_seq_length': seq_length_with_past, 'pruned_indices': None, 'kept_indices': None,
        'pruning_layer': None, 'rater_ids': None,
    }
    layer_mask = causal_mask
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if layer_idx == K:
            image_end = min(img_start + img_len, seq_length_with_past)
            keep_indexs = torch.cat((
                torch.arange(img_start, device=device),
                keep_visual.sort().values + img_start,
                torch.arange(image_end, seq_length_with_past, device=device),
            ))
            new_seq_length = keep_indexs.shape[0]
            all_idx = torch.arange(seq_length_with_past, device=device)
            pruning_info['pruned_indices'] = all_idx[~torch.isin(all_idx, keep_indexs)]
            pruning_info['kept_indices'] = keep_indexs
            pruning_info['pruning_layer'] = layer_idx
            hidden_states = hidden_states[:, keep_indexs, :]
            position_ids = keep_indexs.unsqueeze(0)
            layer_mask = self._update_causal_mask(None, hidden_states, 0)
            cache_position = cache_position[:new_seq_length]
            seq_length_with_past = new_seq_length

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=layer_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
    self.pruning_info = pruning_info
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, pruning_info] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states, past_key_values=next_cache,
        hidden_states=all_hidden_states, attentions=all_self_attns,
    )


def _oracle_depth_prefill(
    self,
    attention_mask: Optional[torch.Tensor],
    past_key_values,
    inputs_embeds: torch.Tensor,
    use_cache: bool,
    output_attentions: bool,
    output_hidden_states: bool,
    return_dict: bool,
    fastv_config: dict,
):
    """
    Oracle B prefill: before layer l keep the visual tokens with depth > l (nested sets, never restored).
    Depths come either from `oracle_depth` (given up front) or from `oracle_depth_fn` (learned predictor), which is
    called at the input of layer `oracle_depth_fn_layer` with the full hidden states and the last prompt token's
    attention to the visual tokens in the layers before it; no visual token is removed before that layer.
    """
    img_start = fastv_config['image_token_start_index']
    img_len = fastv_config['image_token_length']
    n_layers = len(self.layers)
    depth = fastv_config.get('oracle_depth')
    depth_fn = fastv_config.get('oracle_depth_fn')
    assert (depth is None) != (depth_fn is None), "give exactly one of oracle_depth / oracle_depth_fn"
    fn_layer = fastv_config.get('oracle_depth_fn_layer', 0)
    if depth is not None:
        depth = depth.to(inputs_embeds.device)
    else:
        assert output_attentions or fn_layer == 0, "the depth predictor needs the attentions of the layers before it"
    assert use_cache and inputs_embeds.shape[1] > 1, "oracle prefill expects the cached multimodal prefill call"
    pre_attn = []

    past_seen_tokens = 0
    if not isinstance(past_key_values, StaticCache):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_seen_tokens = past_key_values.get_seq_length()
    assert past_seen_tokens == 0, "oracle depth prefill expects an empty cache"
    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, past_seen_tokens)
    seq_length = inputs_embeds.shape[1]
    device = inputs_embeds.device
    image_end = min(img_start + img_len, seq_length)

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    cur_index = torch.arange(seq_length, device=device)  # original positions of the rows in hidden_states
    n_visual = img_len
    layer_counts = []
    first_layer = None
    layer_mask = causal_mask
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if depth is None and layer_idx == fn_layer:
            attn_rows = torch.cat(pre_attn, dim=1) if pre_attn else None
            depth = depth_fn(hidden_states, attn_rows, img_start).to(device)
            assert int(depth.min()) >= fn_layer, "predictor removed a token before its own layer"
        if depth is None:
            layer_counts.append(n_visual)
        else:
            assert depth.shape == (img_len,) and int(depth.min()) >= 0 and int(depth.max()) <= n_layers
            kept_visual = torch.nonzero(depth > layer_idx, as_tuple=False).squeeze(-1)
        if depth is not None and len(kept_visual) < n_visual:
            keep_indexs = torch.cat((
                torch.arange(img_start, device=device),
                kept_visual + img_start,
                torch.arange(image_end, seq_length, device=device),
            ))
            rows = torch.nonzero(torch.isin(cur_index, keep_indexs), as_tuple=False).squeeze(-1)
            hidden_states = hidden_states[:, rows, :]
            cur_index = keep_indexs
            position_ids = keep_indexs.unsqueeze(0)
            layer_mask = self._update_causal_mask(None, hidden_states, 0)
            cache_position = torch.arange(len(keep_indexs), device=device)
            n_visual = len(kept_visual)
            first_layer = layer_idx if first_layer is None else first_layer
        if depth is not None:
            layer_counts.append(n_visual)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=layer_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
            if depth is None:  # still before the predictor: keep the last token's attention to the visual tokens
                pre_attn.append(layer_outputs[1][:, :, -1, img_start:img_start + img_len])
        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

    # Decode steps derive their position from the layer-0 cache length; tokens dropped before layer 0 shorten it.
    self._oracle_pos_offset = int((depth == 0).sum())
    all_idx = torch.arange(seq_length, device=device)
    self.pruning_info = {
        'original_seq_length': seq_length,
        'pruned_indices': all_idx[~torch.isin(all_idx, cur_index)],
        'kept_indices': cur_index,  # kept through the last layer
        'pruning_layer': first_layer,
        'rater_ids': None,
        'visual_depth': depth,
        'layer_visual_counts': layer_counts,
    }

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, self.pruning_info] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states, past_key_values=next_cache,
        hidden_states=all_hidden_states, attentions=all_self_attns,
    )


def install_oracle_prefill(llama_model) -> None:
    """Route `fastv_forward` calls carrying an oracle selection to the oracle prefills on this instance only."""
    if getattr(llama_model, "_oracle_prefill_installed", False):
        return
    official = llama_model.fastv_forward  # bound method of the shared implementation
    official_forward = llama_model.forward
    llama_model._oracle_pos_offset = 0

    def fastv_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                      inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                      return_dict=None, cache_position=None, fastv_config=None):
        self._oracle_pos_offset = 0  # every prefill starts a new sequence
        use_depth = fastv_config is not None and (fastv_config.get('oracle_depth') is not None
                                                  or fastv_config.get('oracle_depth_fn') is not None)
        if fastv_config is None or (fastv_config.get('oracle_keep_visual') is None and not use_depth):
            return official(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                            return_dict=return_dict, cache_position=cache_position, fastv_config=fastv_config)
        assert input_ids is None and position_ids is None and cache_position is None
        cfg = self.config
        return (_oracle_depth_prefill if use_depth else _oracle_prefill)(
            self, attention_mask, past_key_values, inputs_embeds,
            use_cache=use_cache if use_cache is not None else cfg.use_cache,
            output_attentions=output_attentions if output_attentions is not None else cfg.output_attentions,
            output_hidden_states=output_hidden_states if output_hidden_states is not None else cfg.output_hidden_states,
            return_dict=return_dict if return_dict is not None else cfg.use_return_dict,
            fastv_config=fastv_config,
        )

    def forward(self, *args, **kwargs):
        # Cached decode after an Oracle B prefill that dropped tokens before layer 0: restore the original position.
        offset = self._oracle_pos_offset
        past = kwargs.get('past_key_values')
        if offset and not args and kwargs.get('cache_position') is None and past is not None:
            past_len = past.get_seq_length() if isinstance(past, Cache) else past[0][0].shape[-2]
            x = kwargs.get('input_ids') if kwargs.get('input_ids') is not None else kwargs.get('inputs_embeds')
            kwargs['cache_position'] = torch.arange(past_len + offset, past_len + offset + x.shape[1], device=x.device)
        return official_forward(*args, **kwargs)

    llama_model.fastv_forward = types.MethodType(fastv_forward, llama_model)
    llama_model.forward = types.MethodType(forward, llama_model)
    llama_model._oracle_prefill_installed = True
