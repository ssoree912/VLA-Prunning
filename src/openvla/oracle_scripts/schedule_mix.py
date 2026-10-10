"""
schedule_mix.py -- fixed per-layer schedule for a given action/instruction weight w (no retraining).

Same procedure as in train_predictor_mix.py, with the mixed score built at weight w instead of the label's 0.5:
    s_w = (1 - w) s_act / |s_act|_{l>=m} + w s_instr / |s_instr|_{l>=m}
DP (budget, min depth m) on --frames training frames (same demo split and seed as training), mean counts, rounded.

Run from src/openvla:
    python oracle_scripts/schedule_mix.py --suite libero_goal --w 0.4
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prismatic.extern.hf.oracle_llama import oracle_depth_dp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--w", type=float, required=True)
    p.add_argument("--min_depth", type=int, default=2)
    p.add_argument("--fastv_k", type=int, default=3)
    p.add_argument("--fastv_r", type=float, default=0.875)
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    m = args.min_depth
    files = sorted(glob.glob(f"/workspace/vla/data/oracle_features/{args.suite}/ep*.npz"))
    by_task = defaultdict(list)
    for f in files:
        by_task[str(np.load(f)["instruction"])].append(f)
    rng = np.random.default_rng(args.seed)
    train = []
    for _, fs in sorted(by_task.items()):
        fs = list(rng.permutation(fs))
        train += fs[max(1, round(args.val_frac * len(fs))):]
    sa, si = [], []
    for f in train:
        name = Path(f).name
        sa.append(np.load(f"/workspace/vla/data/oracle_labels/{args.suite}/{name}")["s_layer"])
        si.append(np.load(f"/workspace/vla/data/oracle_instr_labels/{args.suite}/{name}")["s_instr"])
    sa, si = np.concatenate(sa).astype(np.float64), np.concatenate(si).astype(np.float64)
    sa /= sa[:, m:].sum((1, 2), keepdims=True)
    si /= si[:, m:].sum((1, 2), keepdims=True)
    s = (1 - args.w) * sa + args.w * si
    N, L, M = s.shape
    k = int(M * (1 - args.fastv_r) + 1e-6)
    budget = args.fastv_k * M + (L - args.fastv_k) * k
    idx = np.random.default_rng(args.seed).choice(N, min(args.frames, N), replace=False)
    counts = []
    for i in idx:
        depth, _ = oracle_depth_dp(torch.from_numpy(s[i]), budget, m)
        counts.append((depth.numpy()[None, :] > np.arange(L)[:, None]).sum(1))
    counts = np.stack(counts)
    mean = counts.mean(0)
    n = np.floor(mean).astype(int)
    for j in np.argsort(-(mean - n), kind="stable")[: budget - n.sum()]:
        n[j] += 1
    n = np.sort(n)[::-1]
    assert n.sum() == budget and (n[:m] == M).all()
    out = Path(__file__).parent / "schedules" / f"{args.suite}_m{m}_w{args.w:g}.json"
    meta = {"n_l": n.tolist(), "budget": int(budget), "min_depth": m, "w": args.w, "suite": args.suite,
            "n_frames": int(len(idx)), "count_std_mean": float(counts.std(0)[m:].mean())}
    json.dump(meta, open(out, "w"), indent=1)
    print(out, json.dumps(meta))


if __name__ == "__main__":
    main()
