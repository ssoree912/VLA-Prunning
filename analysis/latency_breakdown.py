"""
latency_breakdown.py -- where the time of our pruned LLM forward goes on the official OFT stack (one GPU, nothing else
running). Same real inputs (RLDS frames -> official predict_action -> LlamaModel inputs_embeds), CUDA-event timing:

  A  official full forward (SDPA, all tokens)
  B  official.depth_forward_official (current: layers 0-1 eager with attentions, per-layer nonzero/isin -> host syncs)
  C  B with layers 0-1 on SDPA + only the 56 action-query attention rows computed explicitly
  D  C + visual tokens reordered once by rank, every pruned layer is a slice (schedule sizes known on the host: no sync)
plus the parts: layers 0-1 eager vs SDPA+rows, predictor (features + PCA + MLP + ranking), and the per-layer pruning ops.
Also checks that C / D give the same kept tokens and ~the same action hidden states as B.

cwd = PYTHONPATH head = official repo (see run_ours_official.sh).
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # noqa: E402

from gazeprune import paths  # noqa: E402
from gazeprune.oft import depth_forward_official, token_layout  # noqa: E402
from gazeprune.predictor import TokenDepthPredictor  # noqa: E402

os.environ.setdefault("OFT_SCHEDULE", str(paths.SCHEDULE_DIR / "libero_spatial_official_mix0.35_m2_b3280.json"))
from verify_port import load_frames  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model  # noqa: E402
from experiments.robot.openvla_utils import get_vla_action, resize_image_for_policy  # noqa: E402

PRED = os.environ.get("OFT_PREDICTOR_CKPT",
                      str(paths.PREDICTOR_ROOT / "official/libero_spatial_twohead_no_avis_pca256/best.pt"))
REPS = 20


def layer_sdpa_rows(layer, h, position_ids, a0, a1, vs, M):
    """Decoder layer with SDPA (mask None = bidirectional) + the action-query attention rows (eager numerics)."""
    at = layer.self_attn
    B, T, _ = h.shape
    x = layer.input_layernorm(h)
    q = at.q_proj(x).view(B, T, at.num_heads, at.head_dim).transpose(1, 2)
    k = at.k_proj(x).view(B, T, at.num_key_value_heads, at.head_dim).transpose(1, 2)
    v = at.v_proj(x).view(B, T, at.num_key_value_heads, at.head_dim).transpose(1, 2)
    cos, sin = at.rotary_emb(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
    w = torch.matmul(q[:, :, a0:a1], k.transpose(2, 3)) / math.sqrt(at.head_dim)
    w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    rows = w[..., vs:vs + M].float().mean(dim=2)  # (B, H, M)
    h = h + at.o_proj(o.transpose(1, 2).reshape(B, T, -1))
    return h + layer.mlp(layer.post_attention_layernorm(h)), rows


def forward_fast(lm, emb, pred, ordered=True, ev=None):
    """C (ordered=False: same per-layer pruning as B) or D (ordered=True). Returns full-length normed hidden."""
    T = emb.shape[1]
    vs, M, a0, a1 = token_layout(T)
    dev = emb.device
    pos = torch.arange(T, device=dev)[None]
    h, rows = emb, []
    for l in range(pred.layer):
        h, r = layer_sdpa_rows(lm.layers[l], h, pos, a0, a1, vs, M)
        rows.append(r)
    mark(ev, "l01")
    score = pred.score(h, torch.cat(rows, 1), vs, a0, a1)[0]
    order = torch.sort(-score, stable=True).indices  # rank -> visual index (same tie rule as depth_from_scores)
    mark(ev, "pred")
    n_l = pred.schedule["n_l"]
    if ordered:  # [BOS, proprio, text, actions, stop] first, then visual tokens by rank: pruning = keeping a prefix
        rest = torch.cat((torch.arange(vs, device=dev), torch.arange(vs + M, T, device=dev)))
        perm = torch.cat((rest, order + vs))
        h, pos, n_fix, cur = h[:, perm], perm[None], len(rest), M
        for l in range(pred.layer, len(lm.layers)):
            if n_l[l] < cur:
                cur = n_l[l]
                h, pos = h[:, :n_fix + cur], pos[:, :n_fix + cur]
            h = lm.layers[l](h, attention_mask=None, position_ids=pos, past_key_value=None, output_attentions=False,
                             use_cache=False, cache_position=None)[0]
    else:  # B's per-layer selection (nonzero / isin -> one host sync per pruned layer)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(M, device=dev)
        depth = (torch.tensor(n_l, device=dev)[None, :] > rank[:, None]).sum(-1)
        cur_index, n_visual = torch.arange(T, device=dev), M
        for l in range(pred.layer, len(lm.layers)):
            kept = torch.nonzero(depth > l, as_tuple=False).squeeze(-1)
            if len(kept) < n_visual:
                keep = torch.cat((torch.arange(vs, device=dev), kept + vs, torch.arange(vs + M, T, device=dev)))
                h = h[:, torch.isin(cur_index, keep)]
                cur_index, pos, n_visual = keep, keep[None], len(kept)
            h = lm.layers[l](h, attention_mask=None, position_ids=pos, past_key_value=None, output_attentions=False,
                             use_cache=False, cache_position=None)[0]
    mark(ev, "layers")
    h = lm.norm(h)
    full = h.new_zeros(1, T, h.shape[-1])
    full[:, pos[0]] = h
    mark(ev, "end")
    return full, order


def mark(ev, name):
    if ev is not None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        ev.append((name, e))


def time_fn(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return ts


@torch.no_grad()
def main():
    cfg = GenerateConfig(pretrained_checkpoint=str(paths.oft_checkpoint("libero_spatial")),
                         task_suite_name="libero_spatial")
    vla, ah, pp, _, proc = initialize_model(cfg)
    lm = vla.language_model.model
    pred = TokenDepthPredictor(PRED, next(vla.parameters()).device)
    official = lm.forward
    embs = []

    def capture(*a, **k):
        embs.append(k["inputs_embeds"].detach().clone())
        return official(*a, **k)

    lm.forward = capture
    for f in load_frames():
        get_vla_action(cfg, vla, proc, {"full_image": resize_image_for_policy(f["image"], 224),
                                        "wrist_image": resize_image_for_policy(f["wrist"], 224),
                                        "state": f["state"].copy()}, f["instr"], ah, pp, None, use_film=False)
    lm.forward = official
    print(f"{len(embs)} frames, T = {sorted(set(e.shape[1] for e in embs))}, schedule sum {sum(pred.schedule['n_l'])}")

    variants = {
        "A full (SDPA)": lambda e: official(inputs_embeds=e, output_hidden_states=False, return_dict=True),
        "B current depth forward": lambda e: depth_forward_official(lm, e, pred),
        "C B + layers 0-1 SDPA+rows": lambda e: forward_fast(lm, e, pred, ordered=False),
        "D C + rank-ordered slicing": lambda e: forward_fast(lm, e, pred, ordered=True),
    }
    res = {}
    for name, fn in variants.items():
        fn(embs[0])
        res[name] = np.concatenate([time_fn(lambda: fn(e)) for e in embs])
    print("\nLLM forward (median ms over frames x reps):")
    for name, ts in res.items():
        print(f"  {name:32s} {np.median(ts):6.1f}   (p10 {np.percentile(ts, 10):.1f}, p90 {np.percentile(ts, 90):.1f})")

    # parts
    e = embs[0]
    T = e.shape[1]
    vs, M, a0, a1 = token_layout(T)
    pos = torch.arange(T, device=e.device)[None]

    def l01_eager():
        h = e
        for l in range(2):
            h = lm.layers[l](h, attention_mask=None, position_ids=pos, output_attentions=True)[0]

    def l01_sdpa():
        h = e
        for l in range(2):
            h = lm.layers[l](h, attention_mask=None, position_ids=pos, output_attentions=False)[0]

    def l01_rows():
        h = e
        for l in range(2):
            h, _ = layer_sdpa_rows(lm.layers[l], h, pos, a0, a1, vs, M)

    h2 = e
    rows = []
    for l in range(2):
        h2, r = layer_sdpa_rows(lm.layers[l], h2, pos, a0, a1, vs, M)
        rows.append(r)
    rows = torch.cat(rows, 1)
    parts = {
        "layers 0-1 eager + attentions": l01_eager,
        "layers 0-1 SDPA (no rows)": l01_sdpa,
        "layers 0-1 SDPA + 56 query rows": l01_rows,
        "predictor score (feat+PCA+MLP)": lambda: pred.score(h2, rows, vs, a0, a1),
        "predictor depth (B, with sort)": lambda: pred(h2, rows, vs, a0, a1),
    }
    print("\nparts (frame 0, median ms):")
    for name, fn in parts.items():
        fn()
        print(f"  {name:34s} {np.median(time_fn(fn)):6.2f}")

    # B breakdown with events inside (layers 0-1 / predictor / layers 2-31 incl. pruning ops)
    for name, ordered in (("C", False), ("D", True)):
        segs = []
        for _ in range(REPS):
            ev = []
            mark(ev, "start")
            forward_fast(lm, e, pred, ordered=ordered, ev=ev)
            torch.cuda.synchronize()
            segs.append([ev[i][1].elapsed_time(ev[i + 1][1]) for i in range(len(ev) - 1)])
        seg = np.median(np.array(segs), 0)
        print(f"  {name} segments: layers0-1 {seg[0]:.1f} | predictor {seg[1]:.1f} | layers2-31 {seg[2]:.1f} | "
              f"norm+scatter {seg[3]:.1f}")

    # equivalence: B (eager layers 0-1) vs C (SDPA layers 0-1): bf16 kernel differences can flip boundary tokens;
    # C vs D (same score): identical kept sets by construction, hidden differs only by the token order inside attention
    print("\nequivalence (all frames):")
    st = {"B~C depth equal": [], "C==D depth": [], "B~C act": [], "C~D act": [], "rows eager~sdpa": [], "score B~C": []}
    for e in embs:
        T = e.shape[1]
        vs, M, a0, a1 = token_layout(T)
        hb = depth_forward_official(lm, e, pred).last_hidden_state
        depth_b = lm.depth_info["visual_depth"]
        out = {}
        for name, ordered in (("C", False), ("D", True)):
            hx, order = forward_fast(lm, e, pred, ordered=ordered)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(len(order), device=order.device)
            out[name] = (hx, (pred.n_l[None, :] > rank[:, None]).sum(-1))
        rel = lambda x, y: float((x[0, T - 58:T - 2].float() - y[0, T - 58:T - 2].float()).abs().max()
                                 / y[0, T - 58:T - 2].float().abs().max())
        st["B~C depth equal"].append(float((out["C"][1] == depth_b).float().mean()))
        st["C==D depth"].append(float((out["C"][1] == out["D"][1]).all()))
        st["B~C act"].append(rel(out["C"][0], hb))
        st["C~D act"].append(rel(out["D"][0], out["C"][0]))
        pos = torch.arange(T, device=e.device)[None]
        h, hs = e, e
        for l in range(2):
            o = lm.layers[l](h, attention_mask=None, position_ids=pos, output_attentions=True)
            hs, r = layer_sdpa_rows(lm.layers[l], hs, pos, a0, a1, vs, M)
            st["rows eager~sdpa"].append(float((o[1][:, :, a0:a1, vs:vs + M].float().mean(2) - r).abs().max()
                                               / r.abs().max()))
            h = o[0]
    for k, v in st.items():
        print(f"  {k:18s} mean {np.mean(v):.4g}  min {np.min(v):.4g}  max {np.max(v):.4g}")

if __name__ == "__main__":
    main()
