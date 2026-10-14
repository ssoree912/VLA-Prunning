"""
eval_predictor.py -- offline checks of the trained ranking predictor.

(1) RLDS validation demos (same split as training): top-k overlap with the teacher ranking r for several k.
(2) Closed-loop rollout states (images saved by run_libero_eval --save_observations, e.g. the obs_a/obs_b runs): the
    teacher label is computed on the fly (dense generate), then the same metrics. These are states the policy actually
    visits, not demo states.

Run from src/openvla:
    CUDA_VISIBLE_DEVICES=1 python oracle_scripts/eval_predictor.py --ckpt /workspace/vla/data/oracle_predictor/libero_spatial_m2_tf1_s0/best.pt
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import DEVICE, get_processor, get_vla
from predictor_features import M, N_PRE, build_batch, prefix_features
from train_predictor import DepthRanker, depth_from_scores, load, load_split, objective

KS = [16, 32, 45, 64, 128]


def metrics(scores, r, s, n_l):
    """Mean top-k overlap with the teacher ranking and fixed-schedule objective ratio vs teacher ranking."""
    out = {}
    for k in KS:
        top_t = torch.zeros_like(r, dtype=torch.bool).scatter_(1, r.topk(k, 1).indices, True)
        top_p = torch.zeros_like(top_t).scatter_(1, scores.topk(k, 1).indices, True)
        out[f"top{k}"] = float((top_t & top_p).sum(1).float().mean() / k)
    out["obj"] = float((objective(s, depth_from_scores(scores, n_l)) / objective(s, depth_from_scores(r, n_l))).mean())
    return out


def table(name, rows):
    print(f"\n[{name}]")
    print(f"{'method':14s} " + " ".join(f"{'top' + str(k):>7s}" for k in KS) + f" {'obj':>7s}")
    for m, d in rows.items():
        print(f"{m:14s} " + " ".join(f"{d['top' + str(k)]:7.3f}" for k in KS) + f" {d['obj']:7.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--obs_glob", default="experiments/logs/oracle/*obs_*_full_obs/*.npz")
    args = parser.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    targs, sched = ck["args"], ck["schedule"]
    suite = targs["suite"]
    n_l = torch.tensor(sched["n_l"], device=DEVICE)
    feat_dir = Path(f"/workspace/vla/data/oracle_features/{suite}")
    pca = np.load(feat_dir / "pca.npz")
    drop, kp = targs.get("drop") or (), targs.get("pca_dim")
    model = DepthRanker(pca_dim=kp or pca["components"].shape[1], dropout=targs["dropout"], drop=drop).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    def predict(hv, hlast, avis):
        with torch.no_grad():
            return torch.cat([model(hv[i:i + 256].float(), hlast[i:i + 256].float(),
                                    avis[i:i + 256].float() if model.use_avis else None)
                              for i in range(0, len(hv), 256)])

    def baselines(pred, r, avis):
        out = {"predictor": pred, "mean_target": (r / r.sum(1, keepdim=True)).mean(0, keepdim=True).expand_as(r)}
        if avis.shape[-1] == r.shape[1]:
            out["attn_l01"] = avis.float().mean(1)
        return out

    # (1) RLDS validation demos
    _, val_files = load_split(suite, targs["val_frac"], targs["seed"])
    va = {k: v.to(DEVICE) for k, v in load(val_files, suite, with_s=True, use_avis="avis" not in drop, pca_dim=kp).items()}
    pred = predict(va["hv"], va["hlast"], va["avis"])
    s, r = va["s"].float(), va["r"]
    table(f"RLDS val demos: {len(val_files)} demos, {len(r)} frames",
          {m: metrics(v, r, s, n_l) for m, v in baselines(pred, r, va["avis"]).items()})
    del va

    # (2) closed-loop rollout states
    obs_files = sorted(glob.glob(args.obs_glob))
    if not obs_files:
        return
    cfg = GenerateConfig(pretrained_checkpoint=f"checkpoints/openvla-7b-finetuned-{suite.replace('_', '-')}",
                         task_suite_name=suite, use_fastv=False)
    vla = get_vla(cfg)
    vla.fastv_image_token_start_index, vla.fastv_image_token_length = 1, M
    processor = get_processor(cfg)
    D = vla.get_action_dim(suite)
    mu, sd, comp = (torch.tensor(pca[k], device=DEVICE) for k in ("mean", "std", "components"))
    comp = comp[:, :kp] if kp else comp
    S, R, HV, HL, AV = [], [], [], [], []
    for f in obs_files:
        d = np.load(f)
        instr = str(d["task_description"])
        for img in d["images"]:
            ids, pv = build_batch(vla, processor, img[None], instr, cfg.center_crop)
            with torch.no_grad():
                out = vla.generate(ids, pixel_values=pv, attention_mask=None, max_new_tokens=D, do_sample=False,
                                   return_dict_in_generate=True, output_attentions=True)
            sl = vla._action_vision_layer_scores(out.attentions)
            del out
            h, a = prefix_features(vla, ids, pv)
            S.append(sl)
            R.append(sl[N_PRE:].sum(0))
            HV.append((((h[0, 1:1 + M].float() - mu) / sd) @ comp))
            HL.append(h[0, -1].float())
            AV.append(a[0].float())
    s, r = torch.stack(S).to(DEVICE), torch.stack(R).to(DEVICE)
    hv, hl, av = torch.stack(HV), torch.stack(HL), torch.stack(AV)
    pred = predict(hv, hl, av)
    table(f"closed-loop rollout states: {len(obs_files)} episodes, {len(r)} queries",
          {m: metrics(v, r, s, n_l) for m, v in baselines(pred, r, av).items()})


if __name__ == "__main__":
    main()
