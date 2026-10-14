"""make_schedule.py -- OFT fixed per-layer schedule for a given budget from the teacher labels: DP (budget, min depth) on
--frames training frames (same train/val split as train.py), mean per-layer counts, rounded. Given at inference via
OFT_SCHEDULE (the ranker is budget-free and trained without it). --mix: DP on the mixed action/instruction label
(two-head ranker).
Budget: (K+1)*M + (L-K-1)*k, k = round(M (1 - fastv_r)).
    python train/make_schedule.py --suite libero_spatial --fastv_r 0.9140625   # k=44 -> 3280 / 16384 = 20.02%
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gazeprune import paths  # noqa: E402
from gazeprune.schedule import budget_from_fastv, make_schedule, split  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--fastv_k", type=int, default=3)
    p.add_argument("--fastv_r", type=float, required=True)
    p.add_argument("--min_depth", type=int, default=2)
    p.add_argument("--frames", type=int, default=2000)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mix", type=float, default=None, help="two-head ranker: DP on the mixed label")
    p.add_argument("--label_dir", default=None, help="default <labels>/<suite> (gazeprune/paths.py)")
    p.add_argument("--tag", default="", help="name tag of the output file, e.g. _official")
    args = p.parse_args()
    label_dir = args.label_dir or str(paths.LABEL_ROOT / args.suite)
    files = sorted(glob.glob(f"{label_dir}/ep*.npz"))
    d = np.load(files[0])
    L, M = d["s_layer"].shape[1:]
    budget = budget_from_fastv(L, M, args.fastv_k, args.fastv_r)
    train_files, _ = split(files, args.val_frac, args.seed)  # same split as train.py
    n_l, std, n_used = make_schedule(train_files, budget, args.min_depth, args.frames, args.seed, args.mix)
    tag = args.tag + ("" if args.mix is None else f"_mix{args.mix:g}")
    out = paths.SCHEDULE_DIR / f"{args.suite}{tag}_m{args.min_depth}_b{budget}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    sched = {"n_l": n_l.tolist(), "budget": budget, "min_depth": args.min_depth, "fastv_k": args.fastv_k,
             "fastv_r": args.fastv_r, "M": int(M), "L": int(L), "retention": budget / (L * M), "count_std_mean": std,
             "n_frames": n_used, "mix": args.mix, "label_dir": label_dir}
    json.dump(sched, open(out, "w"), indent=1)
    print(out, json.dumps(sched))


if __name__ == "__main__":
    main()
