"""
train_predictor_mix.py -- ranker for the mixed (action + instruction) label, base OpenVLA, with input ablations.

Target (per frame, layers >= m):  r_i = (1 - mix) * r_act_i / sum(r_act) + mix * r_instr_i / sum(r_instr)
    r_act from oracle_labels (s_layer), r_instr from oracle_instr_labels (s_instr); same as the mixed oracle.
Schedule: mean DP counts on the per-layer mixed score s_mix = (1-mix) s_act/|s_act| + mix s_instr/|s_instr| (layers>=m).
Variants (inputs; target, schedule and training identical):
    A  hv + last prompt token (hlast, avis)           C  A + instruction (hins, ains)
    B  A + ains                                       D  hv + instruction only (hins, ains)
Key metric: recall of the "instruction-only" tokens = in the mixed top-n2 but not in the action top-n2.
Boundary-aware loss (--pair_weights): KL + lambda * mean softplus(-(s_i - s_j)) over i in the teacher top-n2 and j in
the teacher ranks n2 .. n2+band plus the predictor's own top-n2 tokens outside the teacher top-n2 (hard negatives).

Run from src/openvla:
    CUDA_VISIBLE_DEVICES=1 python oracle_scripts/train_predictor_mix.py --suite libero_goal --variants A B C D
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
from prismatic.extern.hf.oracle_llama import oracle_depth_dp
from prismatic.extern.hf.oracle_predictor import DepthRanker, combine_heads, depth_from_scores

VARIANTS = {"A": (True, False, False), "B": (True, False, True), "C": (True, True, True), "D": (False, True, True)}
# (use_last, use_hins, use_ains); B uses ains only -> hins zeroed


def round_schedule(mean_counts, budget):
    n = np.floor(mean_counts).astype(int)
    for j in np.argsort(-(mean_counts - n), kind="stable")[: budget - n.sum()]:
        n[j] += 1
    return np.sort(n)[::-1]


def load(files, suite, drop=(), pca_dim=None):
    """drop: attention features not read ("avis", "ains"; they may be absent from the files) -> dummy (N, 1, 1).
    pca_dim: keep the first pca_dim PCA coordinates of hv (stored sorted by explained variance)."""
    out = defaultdict(list)
    for f in files:
        name = Path(f).name
        fe = np.load(f"/workspace/vla/data/oracle_features/{suite}/{name}")
        la = np.load(f"/workspace/vla/data/oracle_labels/{suite}/{name}")
        ins = np.load(f"/workspace/vla/data/oracle_instr_labels/{suite}/{name}")
        assert len(fe["r"]) == len(ins["hins"]) == len(la["s_layer"])
        out["hv"].append(fe["hv"][..., :pca_dim] if pca_dim else fe["hv"]); out["hlast"].append(fe["hlast"])
        if "avis" not in drop:
            out["avis"].append(fe["avis"])
        out["hins"].append(ins["hins"])
        if "ains" not in drop:
            out["ains"].append(ins["ains"])
        out["s_act"].append(la["s_layer"]); out["s_instr"].append(ins["s_instr"])
    res = {k: torch.from_numpy(np.concatenate(v)) for k, v in out.items()}
    for k in ("avis", "ains"):
        if k in drop:
            res[k] = torch.zeros(len(res["hv"]), 1, 1, dtype=torch.float16)
    return res


def mixed(s_act, s_ins, mix, m):
    """Per-layer mixed score (N, L, M) and its ranking target r (N, M)."""
    sa, si = s_act.float(), s_ins.float()
    sa = sa / sa[:, m:].sum(dim=(1, 2), keepdim=True)
    si = si / si[:, m:].sum(dim=(1, 2), keepdim=True)
    s = (1 - mix) * sa + mix * si
    return s, s[:, m:].sum(1)


def objective(s, depth):
    keep = torch.arange(s.shape[1], device=s.device)[None, :, None] < depth[:, None, :]
    return (s * keep).sum(dim=(1, 2))


def run_model(model, d, idx, dev):
    g = lambda k: d[k][idx].to(dev).float()
    return model(g("hv"), g("hlast"), g("avis"), g("hins"), g("ains"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--variants", nargs="+", default=["A", "B", "C", "D"])
    p.add_argument("--mix", type=float, default=0.5)
    p.add_argument("--min_depth", type=int, default=2)
    p.add_argument("--fastv_k", type=int, default=3)
    p.add_argument("--fastv_r", type=float, default=0.875)
    p.add_argument("--schedule_frames", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pair_weights", nargs="+", type=float, default=[0.0], help="lambda per run (variant x lambda)")
    p.add_argument("--neg_band", type=int, default=47)
    p.add_argument("--out_tag", default="", help="suffix of the output folder (keeps earlier checkpoints)")
    p.add_argument("--two_head", action="store_true", help="separate action / instruction heads, mixed at inference")
    p.add_argument("--head_weights", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6])
    p.add_argument("--drop", nargs="*", default=[], choices=["avis", "ains"], help="attention inputs left out (as OFT)")
    p.add_argument("--pca_dim", type=int, default=None, help="use the first k PCA coordinates of hv")
    args = p.parse_args()
    dev, m = "cuda", args.min_depth
    out_root = Path(f"/workspace/vla/data/oracle_predictor_mix/{args.suite}_m{m}_mix{args.mix:g}_s{args.seed}{args.out_tag}")
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"/workspace/vla/data/oracle_features/{args.suite}/ep*.npz"))
    by_task = defaultdict(list)
    for f in files:
        by_task[str(np.load(f)["instruction"])].append(f)
    rng = np.random.default_rng(args.seed)
    tr_f, va_f = [], []
    for _, fs in sorted(by_task.items()):
        fs = list(rng.permutation(fs))
        n_val = max(1, round(args.val_frac * len(fs)))
        va_f += fs[:n_val]; tr_f += fs[n_val:]
    t0 = time.time()
    tr, va = load(tr_f, args.suite, args.drop, args.pca_dim), load(va_f, args.suite, args.drop, args.pca_dim)
    for d in (tr, va):
        d["s_mix"], d["r"] = mixed(d["s_act"], d["s_instr"], args.mix, m)
        d["r_act"] = d["s_act"][:, m:].float().sum(1)
        d["ra_n"] = d["r_act"] / d["r_act"].sum(1, keepdim=True)                 # action half (sums to 1)
        ri = d["s_instr"][:, m:].float().sum(1)
        d["ri_n"] = ri / ri.sum(1, keepdim=True)                                 # instruction half (sums to 1)
    L, M = tr["s_act"].shape[1], tr["s_act"].shape[2]
    k = int(M * (1 - args.fastv_r) + 1e-6)
    budget = args.fastv_k * M + (L - args.fastv_k) * k
    print(f"train {len(tr_f)} demos / {len(tr['r'])} frames, val {len(va_f)} / {len(va['r'])}, {time.time() - t0:.0f}s, "
          f"budget {budget}", flush=True)

    sched_path = Path(__file__).parent / "schedules" / f"{args.suite}_m{m}_mix{args.mix:g}.json"
    if sched_path.exists():
        schedule = json.load(open(sched_path))
    else:
        idx = np.random.default_rng(args.seed).choice(len(tr["r"]), min(args.schedule_frames, len(tr["r"])), replace=False)
        counts = []
        for i in idx:
            depth, _ = oracle_depth_dp(tr["s_mix"][i].double(), budget, m)
            counts.append((depth.numpy()[None, :] > np.arange(L)[:, None]).sum(1))
        counts = np.stack(counts)
        n_l = round_schedule(counts.mean(0), budget)
        schedule = {"n_l": n_l.tolist(), "budget": budget, "min_depth": m, "mix": args.mix, "fastv_k": args.fastv_k,
                    "fastv_r": args.fastv_r, "suite": args.suite, "count_std_mean": float(counts.std(0)[m:].mean()),
                    "n_frames": int(len(idx))}
        json.dump(schedule, open(sched_path, "w"), indent=1)
    print(f"schedule: {schedule['n_l']}", flush=True)
    n_l = torch.tensor(schedule["n_l"], device=dev)
    n2 = int(n_l[m])

    # validation references
    s_va, r_va, ra_va = va["s_mix"].to(dev), va["r"].to(dev), va["r_act"].to(dev)
    top = lambda x, kk: torch.zeros_like(x, dtype=torch.bool).scatter_(1, x.topk(kk, 1).indices, True)
    t_mix, t_act = top(r_va, n2), top(ra_va, n2)
    instr_only = t_mix & ~t_act  # tokens the instruction term adds
    obj_teacher = objective(s_va, depth_from_scores(r_va, n_l))

    rank_va = torch.empty_like(r_va, dtype=torch.long)
    rank_va.scatter_(1, torch.sort(-r_va, dim=1, stable=True).indices, torch.arange(r_va.shape[1], device=dev).expand_as(rank_va))
    mid = (rank_va >= 20) & (rank_va < n2)  # the band missed in failed rollouts

    def metrics(score):
        tp = top(score, n2)
        return {f"top{n2}": float((tp & t_mix).sum(1).float().mean() / n2),
                "rank20_46_recall": float((tp & mid).sum() / mid.sum()),
                "instr_only_recall": float((tp & instr_only).sum() / instr_only.sum().clamp_min(1)),
                "obj_vs_teacher": float((objective(s_va, depth_from_scores(score, n_l)) / obj_teacher).mean())}

    base = {"action_rank (r_act)": metrics(ra_va)}
    if "avis" not in args.drop:
        base["attn_last (avis)"] = metrics(va["avis"].to(dev).float().mean(1))
    if "ains" not in args.drop:
        base["attn_instr (ains)"] = metrics(va["ains"].to(dev).float().mean(1))
    print(f"instruction-only tokens per frame: {float(instr_only.sum(1).float().mean()):.2f} of {n2}", flush=True)
    for kk, v in base.items():
        print(f"  baseline {kk:22s} {json.dumps({a: round(b, 4) for a, b in v.items()})}", flush=True)

    results = {"baselines": base, "schedule": schedule}
    def pair_loss(logits, r):
        """Boundary-aware pairwise loss: teacher top-n2 vs boundary band + predicted-in/teacher-out tokens."""
        order = torch.sort(-r, dim=1, stable=True).indices
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(r.shape[1], device=r.device).expand_as(order))
        prank = torch.empty_like(order)
        prank.scatter_(1, torch.sort(-logits.detach(), dim=1, stable=True).indices,
                       torch.arange(r.shape[1], device=r.device).expand_as(order))
        pos = torch.gather(logits, 1, order[:, :n2])                                  # (B, n2)
        neg_mask = (rank >= n2) & ((rank < n2 + args.neg_band) | (prank < n2))        # (B, M)
        diff = pos[:, :, None] - logits[:, None, :]                                    # (B, n2, M)
        lp = F.softplus(-diff) * neg_mask[:, None, :]
        return (lp.sum(dim=(1, 2)) / (n2 * neg_mask.sum(1).clamp_min(1))).mean()

    runs = [(v, lam) for v in args.variants for lam in args.pair_weights]
    for var, lam in runs:
        use_last, use_hins, use_ains = VARIANTS[var]
        torch.manual_seed(args.seed)
        model = DepthRanker(pca_dim=tr["hv"].shape[-1], dropout=args.dropout, use_last=use_last,
                            use_instr=use_hins or use_ains, n_out=2 if args.two_head else 1, drop=args.drop).to(dev)
        if var == "B":  # instruction attention only: zero the instruction hidden summary
            for d in (tr, va):
                d.setdefault("hins_backup", d["hins"]); d["hins"] = torch.zeros_like(d["hins_backup"])
        elif "hins_backup" in tr:
            for d in (tr, va):
                d["hins"] = d["hins_backup"]
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps = len(tr["r"]) // args.batch
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.05)
        best, best_res = -1.0, None
        for epoch in range(args.epochs):
            perm = torch.randperm(len(tr["r"]))
            model.train()
            for b in range(steps):
                i = perm[b * args.batch:(b + 1) * args.batch]
                r = tr["r"][i].to(dev)
                logits = run_model(model, tr, i, dev)
                if args.two_head:  # each head learns its own half of the label
                    loss = (F.kl_div(F.log_softmax(logits[..., 0], 1), tr["ra_n"][i].to(dev), reduction="batchmean")
                            + F.kl_div(F.log_softmax(logits[..., 1], 1), tr["ri_n"][i].to(dev), reduction="batchmean"))
                    logits = combine_heads(logits, args.mix).log()
                else:
                    loss = F.kl_div(F.log_softmax(logits, 1), r / r.sum(1, keepdim=True), reduction="batchmean")
                if lam > 0:
                    loss = loss + lam * pair_loss(logits, r)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            model.eval()
            with torch.no_grad():
                score = torch.cat([run_model(model, va, torch.arange(j, min(j + 256, len(va["r"]))), dev)
                                   for j in range(0, len(va["r"]), 256)])
            if args.two_head:
                per_w = {w: metrics(combine_heads(score, w)) for w in args.head_weights}
                res = dict(per_w[args.mix], per_w={f"{w:g}": v for w, v in per_w.items()})
            else:
                res = metrics(score)
            if res["obj_vs_teacher"] > best:
                best, best_res = res["obj_vs_teacher"], dict(res, epoch=epoch + 1)
                out = out_root / ((var if lam == 0 else f"{var}_pair{lam:g}") + ("_2head" if args.two_head else ""))
                out.mkdir(exist_ok=True)
                torch.save({"model": model.state_dict(), "args": vars(args), "schedule": schedule, "variant": var, "pair_weight": lam,
                            "two_head": args.two_head, "head_weight": args.mix,
                            "use_last": use_last, "use_instr": use_hins or use_ains, "zero_hins": var == "B",
                            "val": best_res}, out / "best.pt")
        print(f"variant {var}{' 2-head' if args.two_head else ''} pair_weight {lam:g} (last={use_last} hins={use_hins} ains={use_ains}, "
              f"{sum(q.numel() for q in model.parameters()) / 1e6:.2f}M): {json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in best_res.items()})}",
              flush=True)
        results[var if lam == 0 else f"{var}_pair{lam:g}"] = best_res
    json.dump(results, open(out_root / f"results_{'_'.join(args.variants)}_{'_'.join(f'{x:g}' for x in args.pair_weights)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
