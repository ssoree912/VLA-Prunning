"""
analyze_label_sources.py -- does the action->vision label miss instruction grounding? (spatial vs goal, base OpenVLA)

For demo frames (run in order so VLA-Pruner's attention history is real) compute, per layer l and visual token i:
    act[l, i]   action-predicting queries -> visual KV (the current label, as in _action_vision_layer_scores)
    txt[l, i]   all prompt text tokens (prefill queries after the image, excluding the last = first action query)
    instr[l, i] instruction tokens only
    ind[l, i]   indirect action -> text -> vision: sum_j A(act -> text_j) * A(text_j -> vision_i), same layer
and the kept (token, layer) cells of
    predictor   learned ranker + fixed schedule (m = 2), from the dense pass features
    teacher     ranking by r = sum_{l>=2} act (Oracle B-fixed)
    pruner      the actual VLA-Pruner selection (layers 0-2 all, layer >= 3 its 32 kept tokens)
Reported: fraction of each score's mass kept by each method (sum over kept cells / total), and top-k overlap between
the act ranking and the txt / instr / ind rankings (sum over layers >= 2).

Run from src/openvla:
    CUDA_VISIBLE_DEVICES=3 python oracle_scripts/analyze_label_sources.py --suites libero_spatial libero_goal
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import DEVICE, get_processor, get_vla, process_image
from experiments.robot.robot_utils import set_seed_everywhere
from prismatic.extern.hf.oracle_llama import oracle_schedule_depth
from prismatic.extern.hf.oracle_predictor import TokenDepthPredictor

M, S0 = 256, 1


def mass(score, keep):
    return float((score * keep).sum() / score.sum())


def topk_overlap(a, b, k):
    return len(set(a.topk(k).indices.tolist()) & set(b.topk(k).indices.tolist())) / k


def analyze_suite(suite, n_demos_per_task, out_dir, max_files=None, max_frames=None):
    cfg = GenerateConfig(pretrained_checkpoint=f"checkpoints/openvla-7b-finetuned-{suite.replace('_', '-')}",
                         task_suite_name=suite, use_fastv=True, use_temporal=True, use_prefil_attention=True,
                         use_text_vision_selection=False, fastv_r=0.875, fastv_k=3)
    cfg.unnorm_key = suite
    set_seed_everywhere(7)
    vla = get_vla(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    processor = get_processor(cfg)
    tok = processor.tokenizer
    D = vla.get_action_dim(cfg.unnorm_key)
    pred = TokenDepthPredictor(f"/workspace/vla/data/oracle_predictor/{suite}_m2_tf1_s0/best.pt", DEVICE)
    n_l = pred.n_l.cpu()
    L = len(n_l)
    layer_idx = torch.arange(L)[:, None]

    files = sorted(glob.glob(f"/workspace/vla/data/oracle_labels/{suite}/ep*.npz"))
    by_task = {}
    for f in files:
        by_task.setdefault(str(np.load(f, mmap_mode="r")["instruction"]), []).append(f)
    rng = np.random.default_rng(0)
    chosen = [f for fs in by_task.values() for f in rng.choice(fs, n_demos_per_task, replace=False)][:max_files]

    recs = []
    for f in chosen:
        d = np.load(f)
        instr = str(d["instruction"])
        prompt = f"In: What action should the robot take to {instr.lower()}?\nOut:"
        n_prefix = len(tok("In: What action should the robot take to").input_ids)
        n_instr = len(tok(" " + instr.lower(), add_special_tokens=False).input_ids)
        vla.reset_av_history()
        for t, img in enumerate(d["images"][:max_frames]):
            image = process_image(Image.fromarray(img).convert("RGB"))
            inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
            ids = inputs["input_ids"]
            if not torch.all(ids[:, -1] == 29871):
                ids = torch.cat((ids, torch.tensor([[29871]], device=ids.device)), dim=1)
            kw = {k: v for k, v in inputs.items() if k != "input_ids"}
            # dense teacher pass
            vla.use_fastv = False
            with torch.no_grad():
                out = vla.generate(ids, max_new_tokens=D, do_sample=False, return_dict_in_generate=True,
                                   output_attentions=True, output_hidden_states=True, **kw)
            P = out.attentions[0][0].shape[-1]
            text_q = torch.arange(S0 + M, P - 1)                     # prompt text queries (after the image)
            instr_q = torch.arange(S0 + M + n_prefix - 1, S0 + M + n_prefix - 1 + n_instr)
            act = torch.stack([torch.stack([out.attentions[r][l][0, :, -1, S0:S0 + M].float().mean(0)
                                            for l in range(L)]) for r in range(D)]).mean(0).cpu()
            txt, ins, ind = [], [], []
            for l in range(L):
                A0 = out.attentions[0][l][0].float()                  # (H, P, P) prefill
                txt.append(A0[:, text_q, S0:S0 + M].mean(dim=(0, 1)))
                ins.append(A0[:, instr_q, S0:S0 + M].mean(dim=(0, 1)))
                t2v = A0[:, text_q, S0:S0 + M]                        # (H, T, M)
                rows = [out.attentions[r][l][0, :, -1, :P].float()[:, text_q] for r in range(D)]  # (H, T) each
                a2t = torch.stack(rows).mean(0)                       # (H, T)
                ind.append(torch.einsum("ht,htm->m", a2t, t2v) / a2t.shape[0])
            scores = {"act": act, "txt": torch.stack(txt).cpu(), "instr": torch.stack(ins).cpu(), "ind": torch.stack(ind).cpu()}
            h = out.hidden_states[0][pred.layer]
            attn_rows = torch.cat([out.attentions[0][l][:, :, -1, S0:S0 + M] for l in range(pred.layer)], dim=1)
            depth_pred = pred(h, attn_rows, S0).cpu()
            del out
            # VLA-Pruner pass (real history; first w queries are full)
            vla.use_fastv = True
            with torch.no_grad():
                vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False, return_dict_in_generate=True,
                                   output_attentions=True, past_key_values=None)
            kept = vla.language_model.pruning_info.get("kept_indices")
            warm = kept is None or len(kept) == P
            if warm:
                continue
            kept = kept.cpu()
            kv = kept[(kept >= S0) & (kept < S0 + M)] - S0
            keep_pr = torch.zeros(L, M)
            keep_pr[:vla.fastv_k] = 1
            keep_pr[vla.fastv_k:, kv] = 1
            depth_t, _ = oracle_schedule_depth(act, n_l)
            keeps = {"predictor": (layer_idx < depth_pred[None]).float(), "teacher": (layer_idx < depth_t[None]).float(),
                     "pruner": keep_pr}
            rec = {"suite": suite, "file": Path(f).name, "t": t, "task": instr}
            for m, keep in keeps.items():
                for sname, sc in scores.items():
                    rec[f"mass/{sname}/{m}"] = mass(sc[2:], keep[2:])  # layers >= 2 (where the methods differ)
            ra = act[2:].sum(0)
            for sname in ("txt", "instr", "ind"):
                rs = scores[sname][2:].sum(0)
                for k in (32, 45):
                    rec[f"top{k}/act~{sname}"] = topk_overlap(ra, rs, k)
            recs.append(rec)
        print(f"{suite} {Path(f).name} frames {len(d['images'])} records {len(recs)}", flush=True)
    del vla
    torch.cuda.empty_cache()
    json.dump(recs, open(out_dir / f"{suite}.json", "w"))
    return recs


def summarize(recs):
    keys = [k for k in recs[0] if "/" in k]
    return {k: float(np.mean([r[k] for r in recs])) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_goal"])
    p.add_argument("--demos_per_task", type=int, default=2)
    p.add_argument("--out_dir", default="/workspace/vla/data/label_source_analysis")
    p.add_argument("--max_files", type=int, default=None, help="smoke test")
    p.add_argument("--max_frames", type=int, default=None, help="smoke test")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for s in args.suites:
        recs = analyze_suite(s, args.demos_per_task, out_dir, args.max_files, args.max_frames)
        summary[s] = summarize(recs)
        summary[s]["n_frames"] = len(recs)
        # per task for the act~instr overlap and pruner/predictor instr mass
        per_task = {}
        for r in recs:
            per_task.setdefault(r["task"], []).append(r)
        summary[s]["per_task"] = {t: {k: float(np.mean([x[k] for x in v])) for k in
                                      ("top45/act~instr", "mass/instr/predictor", "mass/instr/pruner",
                                       "mass/act/predictor", "mass/act/pruner")} for t, v in per_task.items()}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    for s, v in summary.items():
        print(f"\n== {s} ({v['n_frames']} pruned frames)")
        for k, x in v.items():
            if k not in ("per_task", "n_frames"):
                print(f"  {k:28s} {x:.4f}")


if __name__ == "__main__":
    main()
