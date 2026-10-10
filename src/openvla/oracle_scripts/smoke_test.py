"""
smoke_test.py -- single-observation checks for Oracle-Act-TopK (plan section 5.1, items 1-6) and Oracle A / B
(per-layer scores, depth DP, depth prefill equivalences, per-layer KV lengths, decode positions).

Run from src/openvla:
    CUDA_VISIBLE_DEVICES=0 python oracle_scripts/smoke_test.py [--check_sdpa]

Closed-loop 2-episode checks for each method are in oracle_scripts/run_smoke_episodes.sh.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import DEVICE, get_processor, get_vla, process_image
from experiments.robot.robot_utils import set_seed_everywhere
from prismatic.extern.hf.oracle_llama import install_oracle_prefill, oracle_depth_dp, oracle_topk
from PIL import Image

W = 3  # temporal window = warm-start length


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def get_observation_image(cfg, task_id=0, init_state=0):
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = suite.get_task(task_id)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(task_id)[init_state])
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    img = get_libero_image(obs, 224)
    env.close()
    return img, task_description


def build_inputs(cfg, processor, img, task_description):
    image = Image.fromarray(img).convert("RGB")
    if cfg.center_crop:
        image = process_image(image)
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    return processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)


def set_mode(vla, mode, fastv_r):
    vla.use_fastv = mode in ("pruner", "oracle")
    vla.use_oracle = mode == "oracle"
    vla.fastv_r = fastv_r
    vla.reset_av_history()


def predict(vla, inputs, unnorm_key):
    action, _ = vla.predict_action(
        **inputs, unnorm_key=unnorm_key, do_sample=False, return_dict_in_generate=True,
        output_attentions=True, past_key_values=None,
    )
    return action, vla.last_query_info


def generate_with(vla, input_ids, gen_kwargs, D, fastv_config):
    """One generate() through fastv_forward with an explicit oracle config (bypasses the oracle query logic)."""
    vla.use_fastv, vla.use_oracle = True, False
    vla.fastv_config = fastv_config
    return vla.generate(input_ids, max_new_tokens=D, do_sample=False, return_dict_in_generate=True,
                        output_attentions=True, output_scores=True, **gen_kwargs)


def check_depth_dp(n_trials=30):
    """Exact DP vs brute force on small random instances."""
    g = torch.Generator().manual_seed(0)
    ok = True
    for _ in range(n_trials):
        L, M = 3, 4
        sc = torch.rand(L, M, generator=g)
        budget = int(torch.randint(0, L * M + 1, (1,), generator=g))
        U = torch.cat([torch.zeros(M, 1, dtype=torch.float64), sc.double().T.cumsum(1)], 1)
        best = max(sum(float(U[i, d[i]]) for i in range(M))
                   for d in itertools.product(range(L + 1), repeat=M) if sum(d) == budget)
        depth, obj = oracle_depth_dp(sc, budget)
        ok &= int(depth.sum()) == budget and abs(obj - best) < 1e-9
        ok &= abs(sum(float(U[i, depth[i]]) for i in range(M)) - obj) < 1e-9
    return report("depth DP == brute force (30 random 3x4 instances)", ok)


def check_oracle_ab(vla, inputs, unnorm_key, input_ids, gen_kwargs, D, M, k, full_ids, s_layer, fastv_r, results):
    L, K = s_layer.shape[0], vla.fastv_k
    install_oracle_prefill(vla.language_model.model)
    vla.fastv_r = fastv_r
    k_b, budget = vla._oracle_budget()
    results.append(report("budget B = K*M + (L-K)*k", k_b == k and budget == K * M + (L - K) * k, f"(k={k}, B={budget}, "
                          f"layer-avg {budget / (L * M):.6f})"))
    prompt_len = input_ids.shape[1] + M

    # depth = L for every token goes through the depth prefill but prunes nothing
    out = generate_with(vla, input_ids, gen_kwargs, D, vla._oracle_fastv_config(fastv_r, None, torch.full((M,), L)))
    results.append(report("depth prefill with all depth=L == Full", np.array_equal(out.sequences[0, -D:].cpu().numpy(), full_ids)))

    # Oracle A mask written as depths reproduces the Oracle A (single-cut) prefill bit for bit
    keep = oracle_topk(s_layer[K:].mean(dim=0), k)
    out_a = generate_with(vla, input_ids, gen_kwargs, D, vla._oracle_fastv_config(fastv_r, keep))
    depth_a = torch.full((M,), K)
    depth_a[keep.cpu()] = L
    out_ad = generate_with(vla, input_ids, gen_kwargs, D, vla._oracle_fastv_config(fastv_r, None, depth_a))
    same_ids = torch.equal(out_a.sequences, out_ad.sequences)
    max_diff = max(float((a.float() - b.float()).abs().max()) for a, b in zip(out_a.scores, out_ad.scores))
    results.append(report("depth prefill with Oracle A mask == Oracle A prefill", same_ids and max_diff == 0.0,
                          f"(max|logit diff|={max_diff:.2e})"))
    del out, out_a, out_ad

    # Oracle B at the target budget: nested sets, exact budget, per-layer KV cache lengths
    depth, obj = oracle_depth_dp(s_layer, budget)
    objective_a = float(s_layer[:K].sum() + s_layer[K:, keep.cpu()].sum())
    counts = (depth[None, :] > torch.arange(L)[:, None]).sum(dim=1)
    results.append(report("oracle B: sum_l n_l == B, n_l non-increasing", int(counts.sum()) == budget
                          and bool((counts[1:] <= counts[:-1]).all()), f"(n_l={counts.tolist()})"))
    results.append(report("oracle B: DP objective >= Oracle A objective", obj >= objective_a - 1e-9,
                          f"({obj:.4f} vs {objective_a:.4f})"))
    out_b = generate_with(vla, input_ids, gen_kwargs, D, vla._oracle_fastv_config(fastv_r, None, depth))
    pinfo = vla.language_model.model.pruning_info
    results.append(report("oracle B: prefill per-layer counts == DP", list(pinfo["layer_visual_counts"]) == counts.tolist()))
    pkv = getattr(out_b, "past_key_values", None)
    if pkv is not None:
        n_text = prompt_len - M
        kv_ok = all(pkv[l][0].shape[-2] == n_text + int(counts[l]) + D - 1 for l in range(L))
        results.append(report("oracle B: KV length of layer l == text + n_l + (D-1)", kv_ok,
                              f"(layer0 {pkv[0][0].shape[-2]}, layer{L - 1} {pkv[L - 1][0].shape[-2]})"))
    print(f"    oracle B ids {out_b.sequences[0, -D:].cpu().numpy()}  (full {full_ids})")
    del out_b

    # Tokens removed before layer 0: decode positions must stay at the original prompt_len + r - 1
    depth_z = torch.full((M,), L)
    depth_z[: M // 16] = 0
    seen = []
    hook = vla.language_model.model.layers[0].self_attn.register_forward_pre_hook(
        lambda mod, args, kw: seen.append(kw["position_ids"].flatten().tolist()) if kw["position_ids"].numel() == 1 else None,
        with_kwargs=True)
    out_z = generate_with(vla, input_ids, gen_kwargs, D, vla._oracle_fastv_config(fastv_r, None, depth_z))
    hook.remove()
    expect = [[prompt_len + r] for r in range(D - 1)]
    results.append(report("depth=0 tokens: decode position ids == original", seen == expect, f"({seen} vs {expect})"))
    del out_z

    # Closed-loop query path for both modes (warm start, full pass, oracle pass)
    vla.oracle_layers = list(range(K, L))
    for mode in ("topk", "depth"):
        vla.oracle_mode = mode
        action, info = run_query_sequence(vla, inputs, unnorm_key, "oracle", fastv_r)
        ok = info["warm"] is False and np.array_equal(info["full_action_ids"], full_ids)
        ok &= info["s_layer"].shape == (L, M)
        if mode == "topk":
            ok &= np.array_equal(np.sort(info["kept_visual"]), np.sort(keep.cpu().numpy()))
        else:
            ok &= int(info["visual_depth"].sum()) == budget and "kept_visual" not in info
        results.append(report(f"oracle_mode={mode}: query path", ok, f"action {action}"))
    vla.oracle_mode = "topk"


def run_query_sequence(vla, inputs, unnorm_key, mode, fastv_r):
    """W warm-start queries then one pruned query on the same observation."""
    set_mode(vla, mode, fastv_r)
    for _ in range(W):
        predict(vla, inputs, unnorm_key)
    return predict(vla, inputs, unnorm_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/openvla-7b-finetuned-libero-10")
    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--fastv_r", type=float, default=0.875)
    parser.add_argument("--check_sdpa", action="store_true", help="also load an SDPA model to compare Full actions")
    args = parser.parse_args()

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint, task_suite_name=args.task_suite_name,
        use_fastv=True, use_temporal=True, use_prefil_attention=True, use_text_vision_selection=False,
        fastv_r=args.fastv_r, use_oracle=True,
    )
    cfg.oracle_layer_ids = list(range(16, 32))
    cfg.unnorm_key = cfg.task_suite_name
    set_seed_everywhere(cfg.seed)
    vla = get_vla(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    processor = get_processor(cfg)
    img, task_description = get_observation_image(cfg)
    inputs = build_inputs(cfg, processor, img, task_description)
    unnorm_key = cfg.unnorm_key
    D = vla.get_action_dim(unnorm_key)
    M, s0 = vla.fastv_image_token_length, vla.fastv_image_token_start_index
    k = int(M * (1 - args.fastv_r) + 1e-6)
    print(f"checkpoint={args.checkpoint} unnorm_key={unnorm_key} D={D} M={M} k={k} task='{task_description}'")
    results = []

    # --- (2) attention rows belong to the query that PREDICTS action token r ---------------------------------------
    set_mode(vla, "full", 0.0)
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat((input_ids, torch.tensor([[29871]], device=input_ids.device)), dim=1)
    gen_kwargs = {k_: v for k_, v in inputs.items() if k_ != "input_ids"}
    out = vla.generate(input_ids, max_new_tokens=D, do_sample=False, return_dict_in_generate=True,
                       output_attentions=True, output_scores=True, **gen_kwargs)
    prompt_len = out.attentions[0][0].shape[-1]
    n_text = input_ids.shape[1]
    results.append(report("M visual tokens", prompt_len == n_text + M, f"(prompt_len={prompt_len}, text={n_text}, M={M})"))
    results.append(report("D attention steps", len(out.attentions) == D, f"({len(out.attentions)})"))
    shapes_ok = all(
        out.attentions[r][l].shape[-1] == prompt_len + r and (r == 0 or out.attentions[r][l].shape[-2] == 1)
        for r in range(D) for l in range(len(out.attentions[r]))
    )
    results.append(report("step r: 1 query over prompt_len+r keys", shapes_ok))
    pred_ok = all(int(out.scores[r][0].argmax()) == int(out.sequences[0, n_text + r]) for r in range(D))
    results.append(report("scores[r] (same forward as attentions[r]) emits action token r", pred_ok))

    layers = list(range(16, 32))
    s_act = vla._full_action_vision_score(out.attentions, layers)
    av, _, _ = vla._extract_action_modality_attentions(out.attentions, pruning_info=None)
    s_ref = av[layers].float().mean(dim=(0, 1, 2))
    results.append(report("S_act == official extractor (full pass)", torch.allclose(s_act, s_ref, atol=1e-6),
                          f"(max|diff|={float((s_act - s_ref).abs().max()):.2e})"))
    print(f"    S_act sum={float(s_act.sum()):.4f}  top-5={s_act.topk(5).indices.tolist()}")
    full_ids = out.sequences[0, -D:].cpu().numpy()
    s_layer = vla._action_vision_layer_scores(out.attentions)
    results.append(report("s_layer[16:].mean == S_act(late)", torch.allclose(s_layer[16:].mean(0), s_act, atol=1e-7)))
    del out, av

    # --- (1)/(4) Full vs rho=1 through the pruning code ----------------------------------------------------------
    set_mode(vla, "full", 0.0)
    a_full, _ = predict(vla, inputs, unnorm_key)
    set_mode(vla, "pruner", 0.0)
    a_fastv_r0, _ = predict(vla, inputs, unnorm_key)
    results.append(report("pruning code at rho=1 == Full", np.array_equal(a_full, a_fastv_r0), f"{a_full} vs {a_fastv_r0}"))
    a_oracle_r0, info = run_query_sequence(vla, inputs, unnorm_key, "oracle", 0.0)
    results.append(report("Oracle-Act-TopK at rho=1 == Full", np.array_equal(a_full, a_oracle_r0), f"{a_oracle_r0}"))
    if args.check_sdpa:
        set_seed_everywhere(cfg.seed)
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        sdpa = OpenVLAForActionPrediction.from_pretrained(
            args.checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True,
            attn_implementation="sdpa",
        ).to(DEVICE)
        # predict_action needs attentions, which SDPA cannot return, so compare raw action token ids
        sdpa_ids = sdpa.generate(input_ids, max_new_tokens=D, do_sample=False, **gen_kwargs)[0, -D:].cpu().numpy()
        n_diff = int((sdpa_ids != full_ids).sum())
        print(f"    eager ids {full_ids}\n    sdpa  ids {sdpa_ids}  ({n_diff}/{D} dims differ; informational)")
        del sdpa
        torch.cuda.empty_cache()

    # --- (3)/(5)/(6) Oracle-Act-TopK at the target budget ----------------------------------------------------------------
    a_oracle, info = run_query_sequence(vla, inputs, unnorm_key, "oracle", args.fastv_r)
    pinfo = vla.language_model.pruning_info
    kept = info["kept_visual"]
    results.append(report("oracle: warm start then pruned query", info["warm"] is False))
    results.append(report("oracle: full pass actions == Full", np.array_equal(info["full_action_ids"], full_ids)))
    results.append(report("oracle: kept visual tokens == k", len(kept) == k and len(np.unique(kept)) == k, f"({len(kept)})"))
    results.append(report("oracle: pruned at layer K", pinfo["pruning_layer"] == vla.fastv_k, f"({pinfo['pruning_layer']})"))
    non_visual = pinfo["kept_indices"][(pinfo["kept_indices"] < s0) | (pinfo["kept_indices"] >= s0 + M)]
    results.append(report("oracle: all non-visual tokens kept", len(non_visual) == prompt_len - M))
    exact = np.array_equal(np.sort(kept), np.sort(info["topk_act"]))
    results.append(report("oracle: kept == TopK_k(S_act) exactly (no semantic/MMDP)", exact))
    ref = np.sort(np.argsort(-info["s_act"], kind="stable")[:k])
    results.append(report("oracle: TopK recomputed from saved S_act matches", np.array_equal(ref, np.sort(kept))))
    if info["topk_ema"] is not None:
        print(f"    |TopK(S_act) & TopK(EMA)| = {len(set(kept.tolist()) & set(info['topk_ema'].tolist()))}/{k}")
    print(f"    oracle action {a_oracle}  (full {a_full})")

    # --- VLA-Pruner at the target budget (same observation repeated as history) ----------------------------------
    a_pruner, info_p = run_query_sequence(vla, inputs, unnorm_key, "pruner", args.fastv_r)
    kept_p = info_p["kept_visual"]
    results.append(report("vla-pruner: kept visual tokens == k", len(kept_p) == k, f"({len(kept_p)})"))
    print(f"    vla-pruner action {a_pruner}")

    # --- Oracle A / Oracle B ---------------------------------------------------------------------------------------
    results.append(check_depth_dp())
    check_oracle_ab(vla, inputs, unnorm_key, input_ids, gen_kwargs, D, M, k, full_ids, s_layer, args.fastv_r, results)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
