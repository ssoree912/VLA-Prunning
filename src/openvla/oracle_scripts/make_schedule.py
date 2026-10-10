"""
make_schedule.py -- fixed per-layer visual token counts n_l for Oracle B-fixed, from the RLDS teacher labels.

n_l = mean over training frames of the Oracle B (min_depth m) DP counts sum_i 1[l < d*_i], rounded to integers that
are non-increasing, equal M for l < m, and sum to the budget B = K*M + (L-K)*k exactly.

Run from src/openvla:
    python oracle_scripts/make_schedule.py --suite libero_spatial --min_depth 2
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prismatic.extern.hf.oracle_llama import oracle_depth_dp


def round_schedule(mean_counts, budget):
    """Largest-remainder rounding to an integer sum, then sorted non-increasing (keeps the sum)."""
    n = np.floor(mean_counts).astype(int)
    for j in np.argsort(-(mean_counts - n), kind="stable")[: budget - n.sum()]:
        n[j] += 1
    return np.sort(n)[::-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--label_dir", default=None)
    parser.add_argument("--min_depth", type=int, default=2)
    parser.add_argument("--fastv_k", type=int, default=3)
    parser.add_argument("--fastv_r", type=float, default=0.875)
    parser.add_argument("--n_frames", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    label_dir = Path(args.label_dir or f"/workspace/vla/data/oracle_labels/{args.suite}")
    files = sorted(glob.glob(str(label_dir / "ep*.npz")))
    s_all = np.concatenate([np.load(f)["s_layer"] for f in files]).astype(np.float64)  # (N, L, M)
    N, L, M = s_all.shape
    k = int(M * (1 - args.fastv_r) + 1e-6)
    budget = args.fastv_k * M + (L - args.fastv_k) * k
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, min(args.n_frames, N), replace=False)
    counts = []
    for q in idx:
        depth, _ = oracle_depth_dp(torch.from_numpy(s_all[q]), budget, args.min_depth)
        counts.append((depth.numpy()[None, :] > np.arange(L)[:, None]).sum(1))
    counts = np.stack(counts)
    n = round_schedule(counts.mean(0), budget)
    assert n.sum() == budget and (n[: args.min_depth] == M).all() and (np.diff(n) <= 0).all()
    out = Path(args.out or Path(__file__).parent / "schedules" / f"{args.suite}_m{args.min_depth}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"n_l": n.tolist(), "budget": int(budget), "min_depth": args.min_depth, "fastv_k": args.fastv_k,
            "fastv_r": args.fastv_r, "suite": args.suite, "source": str(label_dir), "n_label_files": len(files),
            "n_frames_used": int(len(idx)), "count_std_mean": float(counts.std(0)[args.min_depth:].mean())}
    json.dump(meta, open(out, "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
