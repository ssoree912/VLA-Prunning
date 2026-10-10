"""
train_predictor.py -- token-ranking predictor for Oracle B-fixed (no DP at inference).

The predictor sits at the layer-1 output. It scores each visual token; tokens are sorted and the token of rank j keeps
depth #{l : n_l > j} under the fixed schedule n_l (oracle_scripts/schedules/<suite>_m2.json), so layer l keeps exactly
the top n_l tokens. Target: the teacher's r_i = sum_{l >= 2} s[l, i] (oracle_scripts/predictor_features.py).

Model (per frame; MLP weights shared over the 256 visual tokens):
    v_i = Linear(512 -> 256)(hv_i)                 visual hidden state at layer 1 (PCA-512)
    t   = Linear(4096 -> 256)(LayerNorm(hlast))    last prompt token: the only position that has seen the instruction
    a_i = Linear(64 -> 64)(log(avis_i + 1e-6))     last-token attention to token i in layers 0, 1 (2 x 32 heads)
    p_i = Embedding(256, 64)[i]                    patch position (16 x 16)
    logit_i = MLP([v_i ; t ; a_i ; p_i]: 640 -> 512 -> 512 -> 1, GELU, dropout)
Loss: listwise KL( r / sum r || softmax(logit) ) over the 256 tokens of a frame.

Validation (held-out demos, ~10% per task): objective ratio sum_i sum_{l < d_i} s[l, i] under the true s for
    predictor depths / teacher-ranking depths (same schedule; the Oracle B-fixed policy) / DP optimum (m = 2),
plus recall of the layer-2 kept set (top n_2) and two no-training baselines (layer 0-1 attention, mean target).

Run from src/openvla:
    CUDA_VISIBLE_DEVICES=1 python oracle_scripts/train_predictor.py --suite libero_spatial
"""

import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prismatic.extern.hf.oracle_llama import oracle_depth_dp
from prismatic.extern.hf.oracle_predictor import DepthRanker, depth_from_scores

M = 256


def objective(s, depth):
    """sum_i sum_{l < d_i} s[l, i] for s (N, L, M), depth (N, M)."""
    L = s.shape[1]
    keep = torch.arange(L, device=s.device)[None, :, None] < depth[:, None, :]
    return (s * keep).sum(dim=(1, 2))


def load_split(suite, val_frac, seed):
    feat_files = sorted(glob.glob(f"/workspace/vla/data/oracle_features/{suite}/ep*.npz"))
    by_task = defaultdict(list)
    for f in feat_files:
        by_task[str(np.load(f)["instruction"])].append(f)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for task, fs in sorted(by_task.items()):
        fs = list(rng.permutation(fs))
        n_val = max(1, round(val_frac * len(fs)))
        val += fs[:n_val]
        train += fs[n_val:]
    return train, val


def load(files, suite, with_s):
    out = defaultdict(list)
    for f in files:
        d = np.load(f)
        for k in ("hv", "hlast", "avis", "r"):
            out[k].append(d[k])
        if with_s:
            out["s"].append(np.load(f"/workspace/vla/data/oracle_labels/{suite}/{Path(f).name}")["s_layer"])
    return {k: torch.from_numpy(np.concatenate(v)) for k, v in out.items()}


@torch.no_grad()
def evaluate(model, data, n_l, budget, min_depth, dev, dp_frames):
    model.eval()
    logits = torch.cat([model(data["hv"][i:i + 256].to(dev).float(), data["hlast"][i:i + 256].to(dev).float(),
                              data["avis"][i:i + 256].to(dev).float()) for i in range(0, len(data["hv"]), 256)])
    s = data["s"].to(dev).float()
    r = data["r"].to(dev)
    scores = {
        "predictor": logits,
        "teacher_rank": r,                                              # Oracle B-fixed (upper bound for ranking)
        "attn_l01": data["avis"].to(dev).float().mean(1),               # no training: layer 0-1 last-token attention
        "mean_target": (r / r.sum(1, keepdim=True)).mean(0, keepdim=True).expand_as(r),  # no image information
    }
    obj = {k: objective(s, depth_from_scores(v, n_l)) for k, v in scores.items()}
    n2 = int(n_l[min_depth])
    top_true = torch.zeros_like(r, dtype=torch.bool).scatter_(1, r.topk(n2, dim=1).indices, True)
    res = {"kl": float(F.kl_div(F.log_softmax(logits, 1), r / r.sum(1, keepdim=True), reduction="batchmean"))}
    for k, v in scores.items():
        top = torch.zeros_like(top_true).scatter_(1, v.topk(n2, dim=1).indices, True)
        res[f"recall@{n2}/{k}"] = float((top & top_true).sum(1).float().mean() / n2)
        res[f"obj/teacher_rank/{k}"] = float((obj[k] / obj["teacher_rank"]).mean())
    # DP optimum (free per-layer counts) on a fixed subset of frames (CPU, ~70 ms/frame)
    if dp_frames > 0:
        idx = torch.linspace(0, len(s) - 1, dp_frames).long()
        dp = torch.tensor([oracle_depth_dp(s[i].cpu(), budget, min_depth)[1] for i in idx], device=dev)
        for k in scores:
            res[f"obj/DP/{k}"] = float((obj[k][idx] / dp).mean())
    model.train()
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--schedule", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--train_frac", type=float, default=1.0, help="fraction of training demos (data scaling)")
    parser.add_argument("--dp_frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    sched = json.load(open(args.schedule or Path(__file__).parent / "schedules" / f"{args.suite}_m2.json"))
    n_l = torch.tensor(sched["n_l"], device=dev)
    out_dir = Path(args.out_dir or f"/workspace/vla/data/oracle_predictor/{args.suite}_m{sched['min_depth']}_"
                                   f"tf{args.train_frac:g}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files = load_split(args.suite, args.val_frac, args.seed)
    if args.train_frac < 1:
        rng = np.random.default_rng(args.seed + 1)
        train_files = sorted(rng.choice(train_files, round(args.train_frac * len(train_files)), replace=False))
    t0 = time.time()
    tr, va = load(train_files, args.suite, with_s=False), load(val_files, args.suite, with_s=True)
    print(f"train {len(train_files)} demos / {len(tr['r'])} frames, val {len(val_files)} demos / {len(va['r'])} "
          f"frames, loaded in {time.time() - t0:.0f}s", flush=True)

    model = DepthRanker(pca_dim=tr["hv"].shape[-1], dropout=args.dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(tr["r"]) // args.batch
    sched_lr = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                   pct_start=0.05)
    best, log = -1.0, []
    for epoch in range(args.epochs):
        perm = torch.randperm(len(tr["r"]))
        tot = 0.0
        for b in range(steps_per_epoch):
            i = perm[b * args.batch:(b + 1) * args.batch]
            r = tr["r"][i].to(dev)
            logits = model(tr["hv"][i].to(dev).float(), tr["hlast"][i].to(dev).float(), tr["avis"][i].to(dev).float())
            loss = F.kl_div(F.log_softmax(logits, 1), r / r.sum(1, keepdim=True), reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched_lr.step()
            tot += float(loss)
        use_dp = epoch in (0, args.epochs - 1) or epoch % 5 == 4
        res = evaluate(model, va, n_l, sched["budget"], sched["min_depth"], dev, args.dp_frames if use_dp else 0)
        res.update({"epoch": epoch + 1, "train_kl": tot / steps_per_epoch})
        log.append(res)
        key = res["obj/teacher_rank/predictor"]
        if key > best:
            best = key
            torch.save({"model": model.state_dict(), "args": vars(args), "schedule": sched, "epoch": epoch + 1,
                        "val": res}, out_dir / "best.pt")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    json.dump(log, open(out_dir / "log.json", "w"), indent=1)
    print(f"best obj/teacher_rank/predictor {best:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
