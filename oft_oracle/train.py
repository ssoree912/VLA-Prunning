"""
train.py -- OFT version: fixed schedule from the teacher labels + token-ranking predictor (layer-1 output, m = 2).

Budget: the OFT VLA-Pruner prunes AFTER layer K (layers 0..K see all M tokens, layers K+1..L-1 keep k), so
    B = (K+1) M + (L-K-1) k = 4*512 + 28*64 = 3840 token-layers (23.4% layer average).
Schedule: mean over --schedule_frames training frames of the DP (budget B, min depth m) per-layer counts, rounded.
Ranker: same as OpenVLA (oft_oracle/core.py DepthRanker, M = 512), target r_i = sum_{l >= m} s[l, i], listwise KL.

Run (openvla-oft env):
    CUDA_VISIBLE_DEVICES=1 python /workspace/vla/oracle_pruning/oft_oracle/train.py --suite libero_spatial
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oft_oracle.core import DepthRanker, depth_from_scores, oracle_depth_dp, round_schedule


def objective(s, depth):
    keep = torch.arange(s.shape[1], device=s.device)[None, :, None] < depth[:, None, :]
    return (s * keep).sum(dim=(1, 2))


def split(files, val_frac, seed):
    by_task = defaultdict(list)
    for f in files:
        by_task[str(np.load(f, mmap_mode="r")["instruction"])].append(f)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for _, fs in sorted(by_task.items()):
        fs = list(rng.permutation(fs))
        n_val = max(1, round(val_frac * len(fs)))
        val += fs[:n_val]
        train += fs[n_val:]
    return train, val


def load(files, keys):
    out = defaultdict(list)
    for f in files:
        d = np.load(f)
        for k in keys:
            out[k].append(d[k])
    return {k: torch.from_numpy(np.concatenate(v)) for k, v in out.items()}


def make_schedule(files, budget, min_depth, n_frames, seed):
    rng = np.random.default_rng(seed)
    per_file = max(1, n_frames // len(files) + 1)
    counts = []
    for f in files:
        s = np.load(f)["s_layer"]
        for t in rng.choice(len(s), min(per_file, len(s)), replace=False):
            depth, _ = oracle_depth_dp(torch.from_numpy(s[t].astype(np.float64)), budget, min_depth)
            counts.append((depth.numpy()[None, :] > np.arange(s.shape[1])[:, None]).sum(1))
            if len(counts) >= n_frames:
                break
        if len(counts) >= n_frames:
            break
    counts = np.stack(counts)
    n = round_schedule(counts.mean(0), budget)
    assert n.sum() == budget and (np.diff(n) <= 0).all()
    return n, float(counts.std(0)[min_depth:].mean()), len(counts)


@torch.no_grad()
def evaluate(model, va, n_l, budget, min_depth, dev, dp_frames):
    model.eval()
    logits = torch.cat([model(va["hv"][i:i + 128].to(dev).float(), va["hq"][i:i + 128].to(dev).float(),
                              va["avis"][i:i + 128].to(dev).float()) for i in range(0, len(va["r"]), 128)])
    model.train()
    r = va["r"].to(dev)
    s = va["s_layer"].to(dev).float()
    scores = {"predictor": logits, "teacher_rank": r, "attn_l01": va["avis"].to(dev).float().mean(1),
              "mean_target": (r / r.sum(1, keepdim=True)).mean(0, keepdim=True).expand_as(r)}
    obj = {k: objective(s, depth_from_scores(v, n_l)) for k, v in scores.items()}
    res = {"kl": float(F.kl_div(F.log_softmax(logits, 1), r / r.sum(1, keepdim=True), reduction="batchmean"))}
    for k, v in scores.items():
        for kk in (32, 64, int(n_l[min_depth]), 128, 256):
            tt = torch.zeros_like(r, dtype=torch.bool).scatter_(1, r.topk(kk, 1).indices, True)
            tp = torch.zeros_like(tt).scatter_(1, v.topk(kk, 1).indices, True)
            res[f"top{kk}/{k}"] = float((tt & tp).sum(1).float().mean() / kk)
        res[f"obj/teacher_rank/{k}"] = float((obj[k] / obj["teacher_rank"]).mean())
    if dp_frames:
        idx = torch.linspace(0, len(s) - 1, dp_frames).long()
        dp = torch.tensor([oracle_depth_dp(s[i].cpu(), budget, min_depth)[1] for i in idx], device=dev)
        for k in scores:
            res[f"obj/DP/{k}"] = float((obj[k][idx] / dp).mean())
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--label_dir", default=None)
    p.add_argument("--min_depth", type=int, default=2)
    p.add_argument("--fastv_k", type=int, default=3)
    p.add_argument("--fastv_r", type=float, default=0.875)
    p.add_argument("--schedule_frames", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--dp_frames", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    label_dir = Path(args.label_dir or f"/workspace/vla/data/oft_labels/{args.suite}")
    out_dir = Path(args.out_dir or f"/workspace/vla/data/oft_predictor/{args.suite}_m{args.min_depth}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(label_dir / "ep*.npz")))
    train_files, val_files = split(files, args.val_frac, args.seed)

    s0 = np.load(files[0])["s_layer"]
    L, M = s0.shape[1], s0.shape[2]
    k = int(round(M * (1 - args.fastv_r)))
    budget = (args.fastv_k + 1) * M + (L - args.fastv_k - 1) * k
    sched_path = out_dir / "schedule.json"
    if sched_path.exists():
        schedule = json.load(open(sched_path))
    else:
        t0 = time.time()
        n_l, std, n_used = make_schedule(train_files, budget, args.min_depth, args.schedule_frames, args.seed)
        schedule = {"n_l": n_l.tolist(), "budget": budget, "min_depth": args.min_depth, "fastv_k": args.fastv_k,
                    "fastv_r": args.fastv_r, "M": M, "L": L, "count_std_mean": std, "n_frames": n_used}
        json.dump(schedule, open(sched_path, "w"), indent=1)
        print(f"schedule ({time.time() - t0:.0f}s): {schedule}", flush=True)
    n_l = torch.tensor(schedule["n_l"], device=dev)

    t0 = time.time()
    tr = load(train_files, ("hv", "hq", "avis", "r"))
    va = load(val_files, ("hv", "hq", "avis", "r", "s_layer"))
    print(f"train {len(train_files)} demos / {len(tr['r'])} frames, val {len(val_files)} demos / {len(va['r'])} frames, "
          f"loaded in {time.time() - t0:.0f}s, budget {budget} ({budget / (L * M):.4f})", flush=True)

    model = DepthRanker(M=M, pca_dim=tr["hv"].shape[-1], n_attn=tr["avis"].shape[1], dropout=args.dropout).to(dev)
    print(f"params {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(tr["r"]) // args.batch
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.05)
    best, log = -1.0, []
    for epoch in range(args.epochs):
        perm = torch.randperm(len(tr["r"]))
        tot = 0.0
        for b in range(steps):
            i = perm[b * args.batch:(b + 1) * args.batch]
            r = tr["r"][i].to(dev)
            logits = model(tr["hv"][i].to(dev).float(), tr["hq"][i].to(dev).float(), tr["avis"][i].to(dev).float())
            loss = F.kl_div(F.log_softmax(logits, 1), r / r.sum(1, keepdim=True), reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lr_sched.step()
            tot += float(loss)
        use_dp = epoch in (0, args.epochs - 1) or epoch % 5 == 4
        res = evaluate(model, va, n_l, budget, args.min_depth, dev, args.dp_frames if use_dp else 0)
        res.update({"epoch": epoch + 1, "train_kl": tot / steps})
        log.append(res)
        if res["obj/teacher_rank/predictor"] > best:
            best = res["obj/teacher_rank/predictor"]
            torch.save({"model": model.state_dict(), "args": vars(args), "schedule": schedule, "M": M,
                        "feature_dir": str(label_dir), "epoch": epoch + 1, "val": res}, out_dir / "best.pt")
        print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in res.items()}), flush=True)
    json.dump(log, open(out_dir / "log.json", "w"), indent=1)
    print(f"best obj/teacher_rank/predictor {best:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
