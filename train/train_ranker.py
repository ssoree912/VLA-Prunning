"""
train_ranker.py -- GazePrune token ranker for OpenVLA-OFT (layer-1 output, m = 2). Budget-free: no DP, no schedule in training.

Ranker: gazeprune/ranker.py DepthRanker (M = 512), target r_i = sum_{l >= m} s[l, i], listwise KL.
--two_head (as base OpenVLA train_predictor_mix.py variant A --two_head): same inputs, two heads on a shared trunk,
    loss = KL(r_act / |r_act| || softmax(head0)) + KL(r_instr / |r_instr| || softmax(head1)), r_instr from s_instr.
Checkpoint selection: lowest validation KL (two-head: kl_act + kl_instr). The per-layer schedule for a budget is made
separately from the training labels (train/make_schedule.py, DP on the label; two-head: on the mixed label) and given at
inference (OFT_SCHEDULE); --report_schedule only adds schedule-based validation metrics to the log (not used to select).

Run (openvla-oft env, from the repo root):
    CUDA_VISIBLE_DEVICES=1 python train/train_ranker.py --suite libero_spatial --two_head --drop avis --pca_dim 256
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
from gazeprune import paths
from gazeprune.ranker import DepthRanker, combine_heads, depth_from_scores
from gazeprune.schedule import load_labels as load, mixed, objective, oracle_depth_dp, split


@torch.no_grad()
def evaluate(model, va, n_l, budget, min_depth, dev, dp_frames, mix=None, head_weights=()):
    """n_l = None: rank metrics only (KL, top-k overlaps); a schedule adds the objective vs teacher rank / DP."""
    model.eval()
    logits = torch.cat([model(va["hv"][i:i + 128].to(dev).float(), va["hq"][i:i + 128].to(dev).float(),
                              va["avis"][i:i + 128].to(dev).float()) for i in range(0, len(va["r"]), 128)])
    model.train()
    if mix is None:
        r = va["r"].to(dev)
        s = va["s_layer"].to(dev).float()
        pred = logits
        res = {"kl": float(F.kl_div(F.log_softmax(logits, 1), r / r.sum(1, keepdim=True), reduction="batchmean"))}
    else:  # two-head: evaluate on the mixed label, heads combined at w = mix
        r, s = va["r_mix"].to(dev), va["s_mix"].to(dev)
        pred = combine_heads(logits, mix)
        res = {"kl_act": float(F.kl_div(F.log_softmax(logits[..., 0], 1), va["ra_n"].to(dev), reduction="batchmean")),
               "kl_instr": float(F.kl_div(F.log_softmax(logits[..., 1], 1), va["ri_n"].to(dev), reduction="batchmean"))}
    scores = {"predictor": pred, "teacher_rank": r,
              "mean_target": (r / r.sum(1, keepdim=True)).mean(0, keepdim=True).expand_as(r)}
    if va["avis"].shape[-1] == r.shape[1]:  # avis stored in the labels (baseline: mean layer-0/1 attention)
        scores["attn_l01"] = va["avis"].to(dev).float().mean(1)
    if mix is not None:
        scores["action_head_only"] = logits[..., 0]
        for w in head_weights:
            scores[f"heads_w{w:g}"] = combine_heads(logits, w)
    ks = sorted({32, 64, 128, 256} | ({int(n_l[min_depth])} if n_l is not None else set()))
    for k, v in scores.items():
        for kk in ks:
            tt = torch.zeros_like(r, dtype=torch.bool).scatter_(1, r.topk(kk, 1).indices, True)
            tp = torch.zeros_like(tt).scatter_(1, v.topk(kk, 1).indices, True)
            res[f"top{kk}/{k}"] = float((tt & tp).sum(1).float().mean() / kk)
    if n_l is None:
        return res
    obj = {k: objective(s, depth_from_scores(v, n_l)) for k, v in scores.items()}
    for k in scores:
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
    p.add_argument("--pca_dim", type=int, default=None, help="use the first k PCA coordinates of hv (default: all)")
    p.add_argument("--report_schedule", default=None, help="schedule.json for extra validation metrics (reporting only)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--dp_frames", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--two_head", action="store_true", help="action / instruction heads (needs s_instr in the labels)")
    p.add_argument("--mix", type=float, default=0.5, help="two-head: label mix of the validation metrics (reporting)")
    p.add_argument("--head_weights", nargs="+", type=float, default=[0.0, 0.3, 0.4, 0.5])
    p.add_argument("--drop", nargs="*", default=[], choices=DepthRanker.FEATURES,
                   help="input features left out of the ranker (ablation), e.g. --drop avis")
    args = p.parse_args()
    mix = args.mix if args.two_head else None
    torch.manual_seed(args.seed)
    dev = "cuda"
    label_dir = Path(args.label_dir or paths.LABEL_ROOT / args.suite)
    out_dir = Path(args.out_dir or paths.PREDICTOR_ROOT / f"{args.suite}_m{args.min_depth}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(label_dir / "ep*.npz")))
    train_files, val_files = split(files, args.val_frac, args.seed)

    s0 = np.load(files[0])["s_layer"]
    L, M = s0.shape[1], s0.shape[2]
    n_l, budget = None, None
    if args.report_schedule:
        rs = json.load(open(args.report_schedule))
        assert rs["min_depth"] == args.min_depth and rs["M"] == M
        n_l, budget = torch.tensor(rs["n_l"], device=dev), rs["budget"]

    t0 = time.time()
    # avis is only read when the ranker uses it (labels may be stored without it); otherwise a dummy, ignored by
    # DepthRanker(drop=("avis",))
    av = ("avis",) if "avis" not in args.drop else ()
    tr = load(train_files, ("hv", "hq", "r") + av + (("s_instr",) if mix is not None else ()), args.pca_dim)
    va = load(val_files, ("hv", "hq", "r", "s_layer") + av + (("s_instr",) if mix is not None else ()), args.pca_dim)
    if not av:
        for d in (tr, va):
            d["avis"] = torch.zeros(len(d["r"]), 1, 1, dtype=torch.float16)
    if mix is not None:
        m = args.min_depth
        for d in (tr, va):
            d["ra_n"] = d["r"] / d["r"].sum(1, keepdim=True)                       # action half (sums to 1)
            ri = d.pop("s_instr")[:, m:].float().sum(1) if d is tr else d["s_instr"][:, m:].float().sum(1)
            d["ri_n"] = ri / ri.sum(1, keepdim=True)                                 # instruction half (sums to 1)
        va["s_mix"] = torch.from_numpy(mixed(va["s_layer"].double().numpy(), va.pop("s_instr").double().numpy(), mix, m)).float()
        va["r_mix"] = va["s_mix"][:, m:].sum(1)
    print(f"train {len(train_files)} demos / {len(tr['r'])} frames, val {len(val_files)} demos / {len(va['r'])} frames, "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    model = DepthRanker(M=M, pca_dim=tr["hv"].shape[-1], n_attn=tr["avis"].shape[1] if av else 64, dropout=args.dropout,
                        n_out=2 if mix is not None else 1, drop=args.drop).to(dev)
    print(f"params {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(tr["r"]) // args.batch
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.05)
    best, log = float("inf"), []
    for epoch in range(args.epochs):
        perm = torch.randperm(len(tr["r"]))
        tot = 0.0
        for b in range(steps):
            i = perm[b * args.batch:(b + 1) * args.batch]
            r = tr["r"][i].to(dev)
            logits = model(tr["hv"][i].to(dev).float(), tr["hq"][i].to(dev).float(), tr["avis"][i].to(dev).float())
            if mix is None:
                loss = F.kl_div(F.log_softmax(logits, 1), r / r.sum(1, keepdim=True), reduction="batchmean")
            else:  # each head learns its own half of the label
                loss = (F.kl_div(F.log_softmax(logits[..., 0], 1), tr["ra_n"][i].to(dev), reduction="batchmean")
                        + F.kl_div(F.log_softmax(logits[..., 1], 1), tr["ri_n"][i].to(dev), reduction="batchmean"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lr_sched.step()
            tot += float(loss)
        use_dp = n_l is not None and (epoch in (0, args.epochs - 1) or epoch % 5 == 4)
        res = evaluate(model, va, n_l, budget, args.min_depth, dev, args.dp_frames if use_dp else 0, mix, args.head_weights)
        res.update({"epoch": epoch + 1, "train_kl": tot / steps})
        log.append(res)
        val_kl = res["kl"] if mix is None else res["kl_act"] + res["kl_instr"]
        if val_kl < best:
            best = val_kl
            torch.save({"model": model.state_dict(), "args": vars(args), "min_depth": args.min_depth, "M": M,
                        "two_head": mix is not None, "mix": mix,
                        "feature_dir": str(label_dir), "epoch": epoch + 1, "val": res}, out_dir / "best.pt")
        print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in res.items()}), flush=True)
    json.dump(log, open(out_dir / "log.json", "w"), indent=1)
    print(f"best val KL {best:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
