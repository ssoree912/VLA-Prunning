"""
diagnose_k.py -- offline diagnosis: is the problem the label (which 32 tokens) or the pruning position K?

For each saved observation (oracle_scripts/collect_obs_spatial.sh):
  1. Full generate -> action tokens a*, per-layer action->vision attention (mean over heads and the 7 predicting queries).
  2. Teacher-forced 7-step decode (prefill, then a* fed one token at a time -- the same path generate() takes, so the
     post-pruning cache/position behaviour is reproduced) for Full and for every (K, selection) pair.
  3. D_act = mean over the 7 dims of KL(p_Full || p_{K,M}) on the 256 action-bin logits (renormalised), plus greedy
     token agreement, |action - action_Full| (unnormalised) and gripper-sign flips.
Selections (all k tokens, fixed per observation across K):
  topk_late   TopK(S_act), S_act = mean attention of layers 16..31 (the closed-loop oracle label)
  topk_nosink same, excluding visual tokens {0,1,13,240} (near-always picked in the spatial oracle run; post-hoc list)
  topk_at_k   TopK of action attention at layer K-1 (the layer the official pruner scores at)
  random      uniform random k tokens (seeded per observation)
Latency: per-query wall time (prefill + 7 decode, no attention outputs -> SDPA kernels) for Full and each K.

    CUDA_VISIBLE_DEVICES=1 python oracle_scripts/diagnose_k.py --obs_glob 'experiments/logs/oracle/*obs_*_obs/*.npz'
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import DEVICE, get_processor, get_vla, process_image
from prismatic.extern.hf.oracle_llama import install_oracle_prefill, oracle_topk

SINKS = [0, 1, 13, 240]
SELECTIONS = ["topk_late", "topk_nosink", "topk_at_k", "random"]


def build_inputs(processor, img, task_description, center_crop=True):
    image = Image.fromarray(img).convert("RGB")
    if center_crop:
        image = process_image(image)
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):  # same as predict_action
        input_ids = torch.cat((input_ids, torch.tensor([[29871]], device=input_ids.device)), dim=1)
    return input_ids, inputs["attention_mask"], inputs["pixel_values"]


def set_pruning(vla, K, keep):
    """keep=None -> plain Full forward; else keep exactly `keep` visual tokens from layer K on (oracle prefill)."""
    vla.use_fastv = keep is not None
    if keep is not None:
        vla.fastv_config = vla._oracle_fastv_config(0.0, keep)
        vla.fastv_config["fastv_k"] = K


@torch.no_grad()
def forced_logits(vla, input_ids, attention_mask, pixel_values, forced_ids):
    """
    Logits (D, V) of the D action-predicting forwards with the given tokens fed back (teacher forcing).
    output_attentions=True keeps the eager attention math the closed-loop runs used (the LLM is loaded with the SDPA
    class, which falls back to eager only when attentions are requested).
    """
    out = vla(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, use_cache=True,
              output_attentions=True, return_dict=True)
    logits, pkv = [out.logits[0, -1].float()], out.past_key_values
    for r in range(len(forced_ids) - 1):
        tok = torch.tensor([[int(forced_ids[r])]], device=input_ids.device)
        out = vla(input_ids=tok, past_key_values=pkv, use_cache=True, output_attentions=True, return_dict=True)
        logits.append(out.logits[0, -1].float())
        pkv = out.past_key_values
    return torch.stack(logits)


def ids_to_action(vla, ids, unnorm_key):
    disc = np.clip(vla.vocab_size - np.asarray(ids) - 1, 0, vla.bin_centers.shape[0] - 1)
    norm = vla.bin_centers[disc]
    st = vla.get_action_stats(unnorm_key)
    mask = st.get("mask", np.ones_like(st["q01"], dtype=bool))
    hi, lo = np.array(st["q99"]), np.array(st["q01"])
    return np.where(mask, 0.5 * (norm + 1) * (hi - lo) + lo, norm)


def action_metrics(vla, logits, ref_logp, full_ids, unnorm_key):
    sl = slice(vla.vocab_size - 256, vla.vocab_size)  # action-bin tokens
    logp = torch.log_softmax(logits[:, sl], dim=-1)
    kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1)  # (D,)
    greedy = logits.argmax(-1).cpu().numpy()
    a, a_ref = ids_to_action(vla, greedy, unnorm_key), ids_to_action(vla, full_ids, unnorm_key)
    return {
        "kl": kl.cpu().numpy(),
        "agree": (greedy == full_ids).astype(np.float32),
        "abs_err": np.abs(a - a_ref).astype(np.float32),
        "grip_flip": float(np.sign(2 * a[-1] - 1) != np.sign(2 * a_ref[-1] - 1)),
    }


@torch.no_grad()
def time_query(vla, input_ids, attention_mask, pixel_values, D, reps):
    for _ in range(2):
        vla.generate(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, max_new_tokens=D, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        vla.generate(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, max_new_tokens=D, do_sample=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_glob", default="experiments/logs/oracle/*obs_*_obs/*.npz")
    parser.add_argument("--checkpoint", default="checkpoints/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--ks", default="3,8,16")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--max_obs", type=int, default=0, help="0 = all")
    parser.add_argument("--timing_obs", type=int, default=10)
    parser.add_argument("--timing_reps", type=int, default=10)
    parser.add_argument("--out", default="experiments/logs/oracle/diagnose_k")
    args = parser.parse_args()
    Ks = [int(x) for x in args.ks.split(",")]
    late = list(range(16, 32))

    cfg = GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=args.task_suite_name,
                         use_fastv=True, use_temporal=True, use_prefil_attention=True, fastv_r=0.0)
    cfg.unnorm_key = cfg.task_suite_name
    torch.manual_seed(0)
    vla = get_vla(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    unnorm_key = cfg.unnorm_key
    processor = get_processor(cfg)
    install_oracle_prefill(vla.language_model.model)
    D = vla.get_action_dim(unnorm_key)
    M = vla.fastv_image_token_length

    observations = []
    for f in sorted(glob.glob(args.obs_glob)):
        z = np.load(f)
        for j in range(len(z["images"])):
            observations.append((f, j, z["images"][j], str(z["task_description"]), int(z["task_id"]), int(z["env_step"][j])))
    if args.max_obs:
        observations = observations[: args.max_obs]
    print(f"{len(observations)} observations, K={Ks}, k={args.k}")
    os.makedirs(args.out, exist_ok=True)

    records, layer_attn = [], []
    for n, (f, j, img, task, task_id, env_step) in enumerate(observations):
        input_ids, attn_mask, pv = build_inputs(processor, img, task)

        # Full: tokens + per-layer action->vision attention (L, M)
        set_pruning(vla, None, None)
        full = vla.generate(input_ids, attention_mask=attn_mask, pixel_values=pv, max_new_tokens=D, do_sample=False,
                            return_dict_in_generate=True, output_attentions=True)
        full_ids = full.sequences[0, -D:].cpu().numpy()
        av = torch.stack([vla._full_action_vision_score(full.attentions, [l]) for l in range(len(full.attentions[0]))])
        del full
        ref = forced_logits(vla, input_ids, attn_mask, pv, full_ids)
        ref_logp = torch.log_softmax(ref[:, vla.vocab_size - 256: vla.vocab_size], dim=-1)
        if n == 0:  # teacher forcing must reproduce generate(); rho=1 through the oracle prefill must give KL 0
            assert np.array_equal(ref.argmax(-1).cpu().numpy(), full_ids), "forced Full != generate Full"
            set_pruning(vla, 3, torch.arange(M, device=DEVICE))
            m0 = action_metrics(vla, forced_logits(vla, input_ids, attn_mask, pv, full_ids), ref_logp, full_ids, unnorm_key)
            print(f"[sanity] rho=1 via oracle prefill: max KL {m0['kl'].max():.2e}, agree {m0['agree'].mean():.2f}")

        s_late = av[late].mean(0)
        sinkless = s_late.clone()
        sinkless[SINKS] = -1.0
        rng = np.random.default_rng(n)
        rand = torch.tensor(np.sort(rng.choice(M, args.k, replace=False)), device=DEVICE)
        for K in Ks:
            sels = {"topk_late": oracle_topk(s_late, args.k), "topk_nosink": oracle_topk(sinkless, args.k),
                    "topk_at_k": oracle_topk(av[K - 1], args.k), "random": rand}
            for name, keep in sels.items():
                set_pruning(vla, K, keep)
                logits = forced_logits(vla, input_ids, attn_mask, pv, full_ids)
                if n == 0 and name == "topk_late":  # forced decode == generate() on the pruned model too
                    gen = vla.generate(input_ids, attention_mask=attn_mask, pixel_values=pv, max_new_tokens=D, do_sample=False,
                                       return_dict_in_generate=True, output_attentions=True).sequences[0, -D:].cpu().numpy()
                    own = forced_logits(vla, input_ids, attn_mask, pv, gen)
                    assert np.array_equal(own.argmax(-1).cpu().numpy(), gen), "forced != generate (pruned)"
                m = action_metrics(vla, logits, ref_logp, full_ids, unnorm_key)
                records.append({"obs": n, "file": os.path.basename(f), "idx": j, "task_id": task_id, "env_step": env_step,
                                "K": K, "sel": name, "kl": float(m["kl"].mean()), "kl_dims": m["kl"].tolist(),
                                "agree": float(m["agree"].mean()), "agree_all": float(m["agree"].all()),
                                "abs_err": m["abs_err"].tolist(), "grip_flip": m["grip_flip"]})
        layer_attn.append(av.half().cpu().numpy())
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(observations)}")

    with open(os.path.join(args.out, "records.jsonl"), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in records)
    np.save(os.path.join(args.out, "layer_action_attention.npy"), np.stack(layer_attn))  # (N, L, M)

    # Latency (no attention outputs)
    timing = {}
    for name, K in [("full", None)] + [(f"K={K}", K) for K in Ks]:
        ts = []
        for f, j, img, task, *_ in observations[: args.timing_obs]:
            input_ids, attn_mask, pv = build_inputs(processor, img, task)
            set_pruning(vla, K, None if K is None else torch.arange(args.k, device=DEVICE))
            ts.append(time_query(vla, input_ids, attn_mask, pv, D, args.timing_reps))
        timing[name] = float(np.mean(ts))
    json.dump(timing, open(os.path.join(args.out, "timing.json"), "w"), indent=1)

    # Summary
    print(f"\n{'K':>3} {'selection':12s} {'KL':>8s} {'agree':>6s} {'all7':>6s} {'|dpos|':>7s} {'grip':>6s}")
    for K in Ks:
        for name in SELECTIONS:
            rs = [r for r in records if r["K"] == K and r["sel"] == name]
            print(f"{K:3d} {name:12s} {np.mean([r['kl'] for r in rs]):8.4f} {np.mean([r['agree'] for r in rs]):6.3f} "
                  f"{np.mean([r['agree_all'] for r in rs]):6.3f} {np.mean([np.mean(r['abs_err'][:3]) for r in rs]):7.4f} "
                  f"{np.mean([r['grip_flip'] for r in rs]):6.3f}")
    print("\nper-query latency (ms, SDPA, prefill + 7 decode):", {k: round(v, 1) for k, v in timing.items()})


if __name__ == "__main__":
    main()
