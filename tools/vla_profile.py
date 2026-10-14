"""Per-query LLM FLOPs + latency profiler shared by every OpenVLA codebase (full / VLA-Pruner / ours / VLA-Cache / ...).

FLOPs: forward hooks on each LLaMA decoder layer read the real query length q (hidden_states) and the real KV length
kv (DynamicCache length of that layer after the call), so token pruning (fewer queries + shorter KV), per-layer depth
schedules and KV reuse (fewer queries, full KV) are all counted the same way. Per layer call (Llama-2-7B, MHA):
    2*q*4*d^2 (q,k,v,o proj) + 2*2*q*kv*d (QK^T, AV) + 2*3*q*d*m (gate/up/down)
LLM backbone only (vision encoder/projector are identical across methods; lm_head excluded). Each query is also
compared with the dense model on the same prompt: prefill q=kv=L_dense in all layers, then the same decode steps.

LLM time: CUDA events from the layer-0 pre-hook to the last-layer post-hook of each forward (prefill q>1 / decode q=1).
Latency: wall clock with torch.cuda.synchronize around `get_action` (method overhead such as patch similarity
included) and around `vla.predict_action` (model only).

Usage: python vla_profile.py <out.jsonl> <run_libero_eval.py> [draccus args...]   (cwd = that repo's src/openvla)
"""
import json
import os
import runpy
import sys
import time

import torch


class Profiler:
    def __init__(self, path):
        self.f = open(path, "w")
        self.q_idx = 0
        self.reset()

    def reset(self):
        self.calls = []  # (layer, q, kv)
        self.events = []  # (q, start, end) CUDA events around each full LLM forward (layer 0 .. last layer)

    def attach(self, vla):
        lm = vla.language_model
        cfg = lm.config
        self.d, self.m = cfg.hidden_size, cfg.intermediate_size
        layers = lm.model.layers
        self.n_layers = len(layers)
        for i, layer in enumerate(layers):
            layer.register_forward_pre_hook(self._pre(i), with_kwargs=True)
            layer.register_forward_hook(self._post(i), with_kwargs=True)
        orig = vla.predict_action

        def timed_predict_action(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(*a, **k)
            torch.cuda.synchronize()
            self.t_model = (time.perf_counter() - t0) * 1e3
            return out

        vla.predict_action = timed_predict_action

    def _pre(self, i):
        def hook(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            self._q = hs.shape[1]
            if i == 0:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self.events.append([hs.shape[1], ev, None])
        return hook

    def _post(self, i):
        def hook(mod, args, kwargs, out):
            cache = kwargs.get("past_key_value")
            kv = None
            if cache is not None and hasattr(cache, "key_cache") and len(cache.key_cache) > i:
                kv = cache.key_cache[i].shape[-2]
            if kv is None:
                mask = kwargs.get("attention_mask")
                kv = mask.shape[-1] if mask is not None and mask.dim() == 4 else self._q
            self.calls.append((i, self._q, kv))
            if i == self.n_layers - 1 and self.events and self.events[-1][2] is None:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                self.events[-1][2] = ev
        return hook

    def layer_flops(self, q, kv):
        d, m = self.d, self.m
        return 8 * q * d * d + 4 * q * kv * d + 6 * q * d * m

    def end_query(self, t_action):
        pre = [c for c in self.calls if c[1] > 1]
        dec = [c for c in self.calls if c[1] == 1]
        flops_pre = sum(self.layer_flops(q, kv) for _, q, kv in pre)
        flops_dec = sum(self.layer_flops(q, kv) for _, q, kv in dec)
        l_dense = max((kv for _, _, kv in pre), default=0)
        n_dec = len(dec) // max(self.n_layers, 1)
        dense = self.n_layers * (self.layer_flops(l_dense, l_dense)
                                 + sum(self.layer_flops(1, l_dense + t + 1) for t in range(n_dec)))
        prefill_q = [0] * self.n_layers
        for i, q, _ in pre:
            prefill_q[i] += q
        t_pre = sum(a.elapsed_time(b) for q, a, b in self.events if b is not None and q > 1)
        t_dec = sum(a.elapsed_time(b) for q, a, b in self.events if b is not None and q == 1)
        rec = dict(q=self.q_idx, flops=flops_pre + flops_dec, flops_prefill=flops_pre, flops_decode=flops_dec,
                   flops_dense=dense, ratio=(flops_pre + flops_dec) / dense if dense else None, l_dense=l_dense,
                   n_decode=n_dec, n_prefill_calls=len(pre) // max(self.n_layers, 1), prefill_q=prefill_q,
                   t_action_ms=t_action, t_model_ms=getattr(self, "t_model", None),
                   t_llm_prefill_ms=t_pre, t_llm_decode_ms=t_dec,
                   mem_peak_mb=torch.cuda.max_memory_allocated() / 2**20, mem_alloc_mb=torch.cuda.memory_allocated() / 2**20)
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.q_idx += 1
        self.reset()


def main():
    out, script, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    prof = Profiler(out)
    sys.path.insert(0, os.getcwd())
    import experiments.robot.robot_utils as ru

    orig_get_model, orig_get_action = ru.get_model, ru.get_action

    def get_model(*a, **k):
        model = orig_get_model(*a, **k)
        prof.attach(model)
        return model

    def get_action(*a, **k):
        prof.reset()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # mem_peak_mb: peak allocated GPU memory during this query
        t0 = time.perf_counter()
        res = orig_get_action(*a, **k)
        torch.cuda.synchronize()
        prof.end_query((time.perf_counter() - t0) * 1e3)
        return res

    ru.get_model, ru.get_action = get_model, get_action
    sys.argv = [script] + args
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
