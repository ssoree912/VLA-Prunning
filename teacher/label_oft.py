"""
label_oft.py -- OpenVLA-OFT teacher labels + predictor features on the OFFICIAL OFT code (gazeprune/oft.py stack),
same outputs as label_rlds.py, with the action queries = the positions the official action head reads
(gazeprune.oft.token_layout: [T - 58, T - 2), i.e. the last prompt token + the first 55 action tokens; the hidden state at
p predicts token p + 1, as in finetune.py). label_rlds.py (VLA-Pruner path) used the action-token positions themselves.

Every frame goes through the official `get_vla_action` (two images, normalized proprio, center crop) with the official
LlamaModel forward and output_attentions=True (eager, mask None = bidirectional; the hidden states equal those of the
SDPA path up to bf16 kernels). Saved per episode (ep%04d.npz), as label_rlds.py:
    s_layer (T, L, M) f16   mean over heads and the 56 action queries of attention to visual token i
    s_instr (T, L, M) f16   mean over heads and the instruction tokens of attention to visual token i
    hv (T, M, P) f16, hq (T, 4096) f16, avis (T, 2H, M) f16, r (T, M) f32, teacher_actions, demo_action
PCA basis fitted by shard 0 on the first --pca_episodes episodes (pca.npz).

Run through scripts/label.sh (official OFT repo as cwd / PYTHONPATH head, official transformers fork):
    python teacher/label_oft.py --suite libero_spatial --num_shards 2 --shard 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import tensorflow_datasets as tfds
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from gazeprune import paths
from gazeprune.oft import N_VISUAL, token_layout
from gazeprune.ranker import query_features
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.openvla_utils import get_vla_action, resize_image_for_policy

N_PRE = 2  # predictor sits at the input of layer 2
PREFIX = "In: What action should the robot take to"
TEXT_START = 1 + N_VISUAL + 1  # [BOS][visual][proprio] then the prompt text (input_ids[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--rlds_dir", default=str(paths.RLDS_ROOT))
    parser.add_argument("--checkpoint", default=None, help="default: gazeprune.paths.oft_checkpoint(suite)")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--pca_dim", type=int, default=512)
    parser.add_argument("--pca_episodes", type=int, default=30)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1, help="episode ep_idx goes to shard ep_idx %% n")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--start_ep", type=int, default=0, help="skip episodes before this index (a second labeler can "
                        "start mid-dataset; existing ep*.npz files are skipped either way)")
    args = parser.parse_args()
    ck = args.checkpoint or str(paths.oft_checkpoint(args.suite))
    out_dir = Path(args.out_dir or paths.LABEL_ROOT / args.suite)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GenerateConfig(pretrained_checkpoint=ck, task_suite_name=args.suite)
    vla, action_head, proprio_projector, _, processor = initialize_model(cfg)
    stats = vla.norm_stats[cfg.unnorm_key]["action"]
    act_scale = np.maximum(np.array(stats["q99"]) - np.array(stats["q01"]), 1e-6)
    act_mask = np.array(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))

    lm = vla.language_model.model
    n_layers = len(lm.layers)
    official_forward = lm.forward
    cap = {}

    def forward(*a, **k):  # official forward, attentions kept (eager in every layer, bidirectional)
        k["output_attentions"] = True
        out = official_forward(*a, **k)
        cap["att"] = out.attentions
        return out

    lm.forward = forward
    lm.layers[N_PRE - 1].register_forward_hook(lambda mod, inp, out: cap.__setitem__("h", out[0].detach()))

    tok = processor.tokenizer
    instr_span = {}

    def instr_positions(instruction, T):
        if instruction not in instr_span:
            a = len(tok(PREFIX).input_ids)
            b = len(tok(f"{PREFIX} {instruction.lower()}").input_ids)
            instr_span[instruction] = (a, b)
        a, b = instr_span[instruction]
        q0, q1 = TEXT_START + a - 1, TEXT_START + b - 1  # input_ids[j] (j >= 1) sits at TEXT_START + j - 1
        assert TEXT_START <= q0 < q1 <= token_layout(T)[2], (instruction, T, a, b)
        return q0, q1

    def teacher(step, instruction):
        obs = {
            "full_image": resize_image_for_policy(step["observation"]["image"].numpy(), 224),
            "wrist_image": resize_image_for_policy(step["observation"]["wrist_image"].numpy(), 224),
            "state": step["observation"]["state"].numpy().astype(np.float64),
        }
        actions = get_vla_action(cfg, vla, processor, obs, instruction, action_head, proprio_projector, None,
                                 use_film=False)
        att = cap.pop("att")
        assert len(att) == n_layers and all(a.dim() == 4 for a in att)
        T = att[0].shape[-1]
        vs, M, a0, a1 = token_layout(T)
        assert a1 - a0 == 56
        s = torch.stack([a[0, :, a0:a1, vs:vs + M].float().mean(dim=(0, 1)) for a in att])  # (L, M)
        q0, q1 = instr_positions(instruction, T)
        s_instr = torch.stack([a[0, :, q0:q1, vs:vs + M].float().mean(dim=(0, 1)) for a in att])  # (L, M)
        rows = torch.cat([att[l][:, :, a0:a1, vs:vs + M].float().mean(dim=2) for l in range(N_PRE)], dim=1)
        hv, hq, av = query_features(cap["h"], rows, vs, M, a0, a1)
        return s, hv, hq, av, np.stack(actions), s_instr

    ds = tfds.builder_from_directory(f"{args.rlds_dir}/{args.suite}_no_noops/1.0.0").as_dataset(split="train")

    # ---- PCA basis of the layer-1 visual hidden states (running moments over the first episodes)
    pca_path = out_dir / "pca.npz"
    while args.shard > 0 and not pca_path.exists():  # shard 0 fits the PCA basis; the others wait for it
        time.sleep(30)
    if not pca_path.exists():
        S1, S2, n = 0.0, 0.0, 0
        for ep_idx, episode in enumerate(ds.take(args.pca_episodes)):
            for t, step in enumerate(episode["steps"]):
                if t % 3:
                    continue
                _, hv, _, _, _, _ = teacher(step, step["language_instruction"].numpy().decode())
                x = hv.double()
                S1, S2, n = S1 + x.sum(0), S2 + x.T @ x, n + len(x)
        mu = S1 / n
        cov = S2 / n - torch.outer(mu, mu)
        sd = cov.diagonal().clamp_min(0).sqrt() + 1e-6
        evals, evecs = torch.linalg.eigh(cov / torch.outer(sd, sd))
        comp = evecs[:, -args.pca_dim:].flip(1)
        explained = float(evals[-args.pca_dim:].sum() / evals.sum())
        np.savez(out_dir / "tmp_pca.npz", mean=mu.float().cpu().numpy(), std=sd.float().cpu().numpy(),
                 components=comp.float().cpu().numpy(), explained=explained, n_vectors=n)
        os.replace(out_dir / "tmp_pca.npz", pca_path)
        print(f"PCA {args.pca_dim} dims from {n} vectors: explained variance {explained:.4f}", flush=True)
    pca = np.load(pca_path)
    mu, sd, comp = (torch.tensor(pca[k], device="cuda") for k in ("mean", "std", "components"))

    # ---- labels + features
    t0, n_frames = time.time(), 0
    for ep_idx, episode in enumerate(ds):
        if args.max_episodes is not None and ep_idx >= args.max_episodes:
            break
        if ep_idx < args.start_ep or ep_idx % args.num_shards != args.shard:
            continue
        path = out_dir / f"ep{ep_idx:04d}.npz"
        if path.exists():
            continue
        rec = {k: [] for k in ("s_layer", "hv", "hq", "avis", "teacher_actions", "demo_action", "s_instr")}
        instruction = None
        for step in episode["steps"]:
            instruction = step["language_instruction"].numpy().decode()
            s, hv, hq, av, acts, s_instr = teacher(step, instruction)
            rec["s_instr"].append(s_instr.half().cpu().numpy())
            rec["s_layer"].append(s.half().cpu().numpy())
            rec["hv"].append((((hv - mu) / sd) @ comp).half().cpu().numpy())
            rec["hq"].append(hq.half().cpu().numpy())
            rec["avis"].append(av.half().cpu().numpy())
            rec["teacher_actions"].append(acts.astype(np.float32))
            rec["demo_action"].append(step["action"].numpy().astype(np.float32))
        out = {k: np.stack(v) for k, v in rec.items()}
        out["r"] = out["s_layer"][:, N_PRE:].astype(np.float32).sum(1)
        err = np.abs(out["teacher_actions"][:, 0] - out["demo_action"])[:, act_mask] / act_scale[act_mask]
        np.savez(path.with_name("tmp_" + path.name), **out, instruction=instruction, episode=ep_idx, stack="official")
        os.replace(path.with_name("tmp_" + path.name), path)  # atomic: never a truncated ep*.npz
        n_frames += len(out["r"])
        print(f"ep {ep_idx:4d}  T={len(out['r']):3d}  frames {n_frames}  {n_frames / (time.time() - t0):.2f} fr/s  "
              f"teacher-vs-demo |a|/range median {np.median(err):.3f}  '{instruction[:50]}'", flush=True)


if __name__ == "__main__":
    main()
