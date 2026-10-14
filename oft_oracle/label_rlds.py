"""
label_rlds.py -- OpenVLA-OFT teacher labels + depth-predictor features from the LIBERO RLDS training set.

Every frame goes through the real OFT inference function `get_vla_action` (third-person + wrist image, normalized
proprio, center crop; vanilla full-token forward with attentions = eager, bidirectional). From that single forward:
    s_layer (T, L, M) f16   teacher label: mean over heads and the 56 action queries of attention to visual token i
    hv      (T, M, P) f16   visual hidden states at the layer-1 output (input of layer 2), standardized + PCA-P
    hq      (T, 4096) f16   mean hidden state of the action queries at the layer-1 output
    avis    (T, 2H, M) f16  action-query attention to visual tokens in layers 0, 1 (head-wise)
    r       (T, M) f32      ranking target sum_{l >= 2} s[l, i]
    s_instr (T, L, M) f16   instruction label: mean over heads and the instruction tokens of attention to visual token i
                            (same forward; the instruction head of the two-head ranker, as in base OpenVLA)
    teacher_actions (T, 8, 7), demo_action (T, 7)
PCA basis: fitted first on --pca_episodes episodes (moments only), saved as pca.npz.
--add_instr: only add s_instr to existing label files that lack it (one more forward per frame, nothing else changes).
Instruction tokens: input_ids range [len(tok(prefix)), len(tok(prefix + " " + instruction))) of the prompt
"In: What action should the robot take to {instruction}?\nOut:"; text token j sits at text_token_start + j - 1 (BOS
is input_ids[0]; visual + proprio tokens come before the text).

Run (openvla-oft env, cwd = /workspace/vla/VLA-Pruner/src/openvla-oft):
    CUDA_VISIBLE_DEVICES=1 python /workspace/vla/oracle_pruning/oft_oracle/label_rlds.py --suite libero_spatial
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # /workspace/vla/oracle_pruning

import oft_oracle  # noqa: F401  (registers the vla_cache_utils stub before the OFT imports)
from oft_oracle.core import query_features
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.openvla_utils import get_vla_action, resize_image_for_policy

N_PRE = 2  # predictor sits at the input of layer 2
PREFIX = "In: What action should the robot take to"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--rlds_dir", default="/workspace/vla/data/modified_libero_rlds")
    parser.add_argument("--checkpoint", default="/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-spatial")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--pca_dim", type=int, default=512)
    parser.add_argument("--pca_episodes", type=int, default=30)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--add_instr", action="store_true", help="only add s_instr to existing label files")
    parser.add_argument("--num_shards", type=int, default=1, help="parallel labelers: episode ep_idx goes to shard ep_idx %% n")
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.out_dir or f"/workspace/vla/data/oft_labels/{args.suite}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=args.suite, use_vla_cache=False,
                         use_fastv=False, use_vla_pruner=False)
    vla, action_head, proprio_projector, _, processor = initialize_model(cfg)
    stats = vla.norm_stats[cfg.unnorm_key]["action"]
    act_scale = np.maximum(np.array(stats["q99"]) - np.array(stats["q01"]), 1e-6)
    act_mask = np.array(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))

    tok = processor.tokenizer
    instr_span = {}

    def instr_positions(instruction, text_start, text_end):
        if instruction not in instr_span:
            a = len(tok(PREFIX).input_ids)
            b = len(tok(f"{PREFIX} {instruction.lower()}").input_ids)
            instr_span[instruction] = (a, b)
        a, b = instr_span[instruction]
        q0, q1 = text_start + a - 1, text_start + b - 1
        assert text_start <= q0 < q1 <= text_end, (instruction, text_start, text_end, a, b)
        return q0, q1

    captured = {}
    vla.language_model.model.layers[N_PRE - 1].register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("h", out[0].detach()))

    def teacher(step, instruction):
        obs = {
            "full_image": resize_image_for_policy(step["observation"]["image"].numpy(), 224),
            "wrist_image": resize_image_for_policy(step["observation"]["wrist_image"].numpy(), 224),
            "state": step["observation"]["state"].numpy().astype(np.float64),
        }
        with torch.no_grad():
            actions, last_caches, _, _ = get_vla_action(cfg, vla, processor, obs, instruction, action_head,
                                                        proprio_projector, None, use_film=False, last_caches=None)
        tm = last_caches["token_metadata"]
        n_layers = len(vla.language_model.model.layers)
        att = last_caches["attentions"][:n_layers]  # the vanilla OFT path appends one extra entry after the layers
        assert all(a.dim() == 4 for a in att)
        vs, M = tm["visual_token_start"], tm["num_visual_tokens"]
        a0, a1 = tm["action_token_start"], tm["action_token_end"]
        assert att[0].shape[-1] == tm["sequence_length"] and a1 - a0 == 56
        s = torch.stack([a[0, :, a0:a1, vs:vs + M].float().mean(dim=(0, 1)) for a in att])  # (L, M)
        q0, q1 = instr_positions(instruction, tm["text_token_start"], tm["text_token_end"])
        s_instr = torch.stack([a[0, :, q0:q1, vs:vs + M].float().mean(dim=(0, 1)) for a in att])  # (L, M)
        rows = torch.cat([att[l][:, :, a0:a1, vs:vs + M].float().mean(dim=2) for l in range(N_PRE)], dim=1)
        hv, hq, av = query_features(captured["h"], rows, vs, M, a0, a1)
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
        if ep_idx % args.num_shards != args.shard:
            continue
        path = out_dir / f"ep{ep_idx:04d}.npz"
        if args.add_instr:  # fill s_instr into an existing label file (other arrays unchanged)
            if not path.exists():
                continue
            try:  # the main labeler may still be writing the newest file: skip it, a later pass picks it up
                old = dict(np.load(path))
            except Exception:
                continue
            if "s_instr" in old:
                continue
            si = [teacher(step, step["language_instruction"].numpy().decode())[5].half().cpu().numpy()
                  for step in episode["steps"]]
            assert len(si) == len(old["r"]), (path, len(si), len(old["r"]))
            tmp = path.with_name("tmp_" + path.name)  # not matched by the ep*.npz globs
            np.savez(tmp, **old, s_instr=np.stack(si))
            os.replace(tmp, path)  # atomic: a killed run never leaves a truncated label file
            n_frames += len(si)
            print(f"ep {ep_idx:4d}  +s_instr T={len(si):3d}  frames {n_frames}  {n_frames / (time.time() - t0):.2f} fr/s",
                  flush=True)
            continue
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
        np.savez(path.with_name("tmp_" + path.name), **out, instruction=instruction, episode=ep_idx)
        os.replace(path.with_name("tmp_" + path.name), path)  # atomic: never a truncated ep*.npz
        n_frames += len(out["r"])
        print(f"ep {ep_idx:4d}  T={len(out['r']):3d}  frames {n_frames}  {n_frames / (time.time() - t0):.2f} fr/s  "
              f"teacher-vs-demo |a|/range median {np.median(err):.3f}  '{instruction[:50]}'", flush=True)


if __name__ == "__main__":
    main()
