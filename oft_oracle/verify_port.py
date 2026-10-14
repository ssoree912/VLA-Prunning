"""
verify_port.py -- checks the port of the depth predictor to the official OpenVLA-OFT code (official.py) against the
VLA-Pruner openvla-oft path (core.py) on a few LIBERO RLDS frames. Two stacks = two processes, then a comparison:

  --stack pruner    (cwd /workspace/vla/VLA-Pruner/src/openvla-oft, env transformers 4.47 fork)
  --stack official  (cwd + PYTHONPATH /workspace/vla/baselines/openvla-oft-official, transformers v4.40.1-openvla-oft)
  --compare         reads both npz files

Per frame and stack: full actions (official: SDPA and eager), teacher label s_layer (action-query -> visual attention,
mean over heads and queries; eager), layer-1 visual hidden + action-query mean hidden, predictor depth and pruned
actions; plus model latency (predict_action, median over repeats).
(a) official full == pruner full (actions, labels, layer-1 hidden) -> labels / predictor need no re-extraction
(b) ported pruned forward == core._depth_forward (same depth, ~same actions)
(c) latency: full vs pruned on each stack
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("/workspace/vla/eval_logs/oft_port")
PRED = "/workspace/vla/data/oft_predictor/libero_spatial_m2_s0/best.pt"
SCHED = "/workspace/vla/data/oft_predictor/schedules/libero_spatial_m2_b3280.json"
RLDS = "/workspace/vla/data/modified_libero_rlds/libero_spatial_no_noops/1.0.0"
EPISODES, STEPS = (0, 97, 211, 333), (0, 45, 90)


def load_frames():
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    ds = tfds.builder_from_directory(RLDS).as_dataset(split="train")
    frames = []
    for ep_idx, ep in enumerate(ds):
        if ep_idx > max(EPISODES):
            break
        if ep_idx not in EPISODES:
            continue
        steps = list(ep["steps"])
        for t in STEPS:
            s = steps[min(t, len(steps) - 1)]
            frames.append(dict(image=s["observation"]["image"].numpy(), wrist=s["observation"]["wrist_image"].numpy(),
                               state=s["observation"]["state"].numpy().astype(np.float64),
                               instr=s["language_instruction"].numpy().decode()))
    return frames


def timed(fn, reps):
    import torch
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def run_stack(stack, reps):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # /workspace/vla/oracle_pruning
    os.environ["OFT_SCHEDULE"] = SCHED
    frames = load_frames()
    import oft_oracle  # noqa: F401
    from oft_oracle.core import TokenDepthPredictor, query_features
    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
    from experiments.robot.openvla_utils import get_vla_action, resize_image_for_policy

    if stack == "pruner":
        ckpt = "/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-spatial"
        cfg = GenerateConfig(pretrained_checkpoint=ckpt, task_suite_name="libero_spatial", use_vla_cache=False,
                             use_fastv=False, use_vla_pruner=True, fastv_k=3, fastv_r=0.9140625, vla_pruner_layer=15,
                             vla_pruner_mode="semantic_action")
    else:
        ckpt = "/workspace/vla/models/oft-ours/openvla-7b-oft-finetuned-libero-spatial-ours"
        cfg = GenerateConfig(pretrained_checkpoint=ckpt, task_suite_name="libero_spatial")
    vla, ah, pp, _, proc = initialize_model(cfg)
    lm = vla.language_model.model
    pred = TokenDepthPredictor(PRED, next(vla.parameters()).device)
    print("attn impl", lm.config._attn_implementation, "| transformers", __import__("transformers").__version__,
          "| prismatic from", __import__("prismatic").__file__, flush=True)

    def obs(f):
        return {"full_image": resize_image_for_policy(f["image"], 224),
                "wrist_image": resize_image_for_policy(f["wrist"], 224), "state": f["state"].copy()}

    cap = {}
    lm.layers[pred.layer - 1].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0].detach().clone()))

    if stack == "pruner":
        from oft_oracle.core import install_depth_forward

        def act(f, mode):
            pruned = mode == "pruned"
            for o in (vla, vla.config):
                o.use_vla_pruner = o.use_fastv = pruned
                o.use_temporal = False
            with torch.no_grad():
                a, lc, _, _ = get_vla_action(cfg, vla, proc, obs(f), f["instr"], ah, pp, None, use_film=False,
                                             last_caches=None)
            if mode == "full_eager":
                att = lc["attentions"][:len(lm.layers)]
                return np.stack(a), att
            return np.stack(a), None

        install_depth_forward(vla, pred)
        get_depth = lambda: lm.pruning_info["visual_depth"]
        modes = ["full_eager", "pruned"]  # the VLA-Pruner vanilla path is eager with attentions in every layer
    else:
        from oft_oracle.official import install_official_depth_forward

        state = {"mode": "full_sdpa"}
        install_official_depth_forward(vla, pred)
        lm_depth = lm.forward
        lm_full = type(lm).forward.__get__(lm)

        def forward(*a, **k):
            if state["mode"] == "full_eager":
                k["output_attentions"] = True
                out = lm_full(*a, **k)
                cap["att"] = out.attentions
                return out
            if state["mode"] == "full_sdpa":
                return lm_full(*a, **k)
            return lm_depth(*a, **k)
        lm.forward = forward

        def act(f, mode):
            state["mode"] = mode
            a = get_vla_action(cfg, vla, proc, obs(f), f["instr"], ah, pp, None, use_film=False)
            return np.stack(a), cap.pop("att", None)

        get_depth = lambda: lm.depth_info["visual_depth"]
        modes = ["full_sdpa", "full_eager", "pruned"]

    rec = {k: [] for k in ["s_layer", "hv", "hq", "depth"] + [f"act_{m}" for m in modes]}
    for f in frames:
        for m in modes:
            a, att = act(f, m)
            rec[f"act_{m}"].append(a)
            if m == "full_eager":
                T = att[0].shape[-1]
                a0, a1 = T - 57, T - 1
                rec["s_layer"].append(torch.stack([x[0, :, a0:a1, 1:513].float().mean(dim=(0, 1)) for x in att]).cpu().numpy())
                rows = torch.cat([att[l][:, :, a0:a1, 1:513].float().mean(dim=2) for l in range(pred.layer)], 1)
                hv, hq, _ = query_features(cap["h"], rows, 1, 512, a0, a1)
                rec["hv"].append(hv.half().cpu().numpy())
                rec["hq"].append(hq.cpu().numpy())
            if m == "pruned":
                rec["depth"].append(get_depth().cpu().numpy())
    lat = {}
    for m in modes:
        if m == "full_eager" and stack == "official":
            continue
        act(frames[0], m)  # warm-up
        ts = []
        for f in frames:
            ts += timed(lambda: act(f, m), reps)
        lat[f"lat_{m}"] = np.array(ts)
        print(f"{stack} {m}: get_vla_action median {np.median(ts):.1f} ms", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"verify_{stack}.npz", **{k: np.stack(v) for k, v in rec.items()}, **lat)
    print("saved", OUT / f"verify_{stack}.npz", flush=True)


def compare():
    P, O = np.load(OUT / "verify_pruner.npz"), np.load(OUT / "verify_official.npz")
    rng = lambda a: np.abs(a).max()
    print(f"frames {len(P['depth'])}")
    print("(a) official vs VLA-Pruner path, full model")
    for mo in ("full_sdpa", "full_eager"):
        d = np.abs(O[f"act_{mo}"] - P["act_full_eager"])
        print(f"  actions official[{mo}] vs pruner[eager]: max|d| {d.max():.4f}  mean|d| {d.mean():.5f}  "
              f"(action |a| max {rng(P['act_full_eager']):.3f})")
    d = np.abs(O["act_full_sdpa"] - O["act_full_eager"])
    print(f"  official sdpa vs official eager: max|d| {d.max():.4f}")
    s_o, s_p = O["s_layer"].astype(np.float64), P["s_layer"].astype(np.float64)
    rel = np.abs(s_o - s_p).sum(-1) / s_p.sum(-1)
    corr = [np.corrcoef(s_o[i, l], s_p[i, l])[0, 1] for i in range(len(s_o)) for l in range(s_o.shape[1])]
    print(f"  s_layer (labels): L1 rel diff per (frame, layer) mean {rel.mean():.4f} max {rel.max():.4f}; "
          f"pearson min {min(corr):.4f} mean {np.mean(corr):.4f}")
    r_o, r_p = s_o[:, 2:].sum(1), s_p[:, 2:].sum(1)
    k = 103  # ~20% of 512
    ov = [len(set(np.argsort(-r_o[i])[:k]) & set(np.argsort(-r_p[i])[:k])) / k for i in range(len(r_o))]
    print(f"  ranking target r: top-{k} overlap mean {np.mean(ov):.3f} min {min(ov):.3f}")
    hv = np.abs(O["hv"].astype(np.float32) - P["hv"].astype(np.float32)).max() / np.abs(P["hv"].astype(np.float32)).max()
    hq = np.abs(O["hq"] - P["hq"]).max() / np.abs(P["hq"]).max()
    print(f"  layer-1 hidden: visual rel max diff {hv:.2e}, action-query mean rel max diff {hq:.2e}")
    print("(b) ported pruned forward vs core._depth_forward")
    same = (O["depth"] == P["depth"]).mean(1)
    print(f"  depth equal fraction per frame: {np.round(same, 4)}")
    d = np.abs(O["act_pruned"] - P["act_pruned"])
    print(f"  pruned actions: max|d| {d.max():.4f} mean|d| {d.mean():.5f}")
    d = np.abs(O["act_pruned"] - O["act_full_sdpa"])
    print(f"  official pruned vs official full: max|d| {d.max():.4f} mean|d| {d.mean():.5f}")
    d = np.abs(P["act_pruned"] - P["act_full_eager"])
    print(f"  pruner pruned vs pruner full: max|d| {d.max():.4f} mean|d| {d.mean():.5f}")
    print("(c) get_vla_action latency, median ms")
    for name, X in (("pruner", P), ("official", O)):
        print("  " + name + ": " + ", ".join(f"{k[4:]} {np.median(X[k]):.1f}" for k in X.files if k.startswith("lat_")))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stack", choices=["pruner", "official"])
    p.add_argument("--compare", action="store_true")
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()
    compare() if args.compare else run_stack(args.stack, args.reps)
