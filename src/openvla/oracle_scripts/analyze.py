"""
analyze.py -- paired comparison of Full / VLA-Pruner / Oracle-Act-TopK (plan section 5.2).

    python oracle_scripts/analyze.py --log_dir ./experiments/logs/oracle --tag pilot

Success rate = task-equal-weighted mean. Delta CIs: paired bootstrap that resamples initial states within each task.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

METHODS = ["full", "pruner", "oracle", "oracle_a", "oracle_b", "oracle_b_m2", "oracle_bfix", "oracle_bfix_mix", "predictor"]


def load(log_dir, tag, method):
    paths = sorted(glob.glob(os.path.join(log_dir, f"*--{tag}_{method}.jsonl")))
    if not paths:
        return None, None
    path = paths[-1]  # latest run
    rows = [json.loads(line) for line in open(path)]
    return {(r["task_id"], r["init_state"]): r for r in rows}, path


def task_sr(res, keys):
    by_task = defaultdict(list)
    for key in keys:
        by_task[key[0]].append(res[key]["success"])
    return {t: float(np.mean(v)) for t, v in sorted(by_task.items())}


def paired_delta(a, b, keys, n_boot, rng):
    by_task = defaultdict(list)
    for key in keys:
        by_task[key[0]].append(float(a[key]["success"]) - float(b[key]["success"]))
    diffs = [np.array(v) for _, v in sorted(by_task.items())]
    point = np.mean([d.mean() for d in diffs])
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diffs])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    a_only = sum(a[k]["success"] and not b[k]["success"] for k in keys)
    b_only = sum(b[k]["success"] and not a[k]["success"] for k in keys)
    return point, lo, hi, a_only, b_only


def oracle_diagnostics(jsonl_path):
    diag_dir = jsonl_path[: -len(".jsonl")] + "_diag"
    files = sorted(glob.glob(os.path.join(diag_dir, "*.npz")))
    if not files:
        return
    ov_ema, kept_act = [], []
    for f in files:
        d = np.load(f)
        if "topk_act" not in d:
            continue
        live = ~d["warm"]
        for i in np.nonzero(live)[0]:
            act, kept = set(d["topk_act"][i]), set(d["kept_visual"][i]) - {-1}
            k = len(act)
            kept_act.append(len(kept & act) / k)  # must be 1.0: oracle keeps exactly TopK_k(S_act)
            if "topk_ema" in d and d["topk_ema"][i][0] >= 0:
                ov_ema.append(len(act & set(d["topk_ema"][i])) / k)
    print(f"\nOracle diagnostics over {len(kept_act)} pruned queries ({len(files)} episodes):")
    print(f"  |TopK(S_act current) & TopK(EMA past)| / k : mean {np.mean(ov_ema):.3f}  median {np.median(ov_ema):.3f}")
    print(f"  |kept & TopK(S_act)| / k (sanity, expect 1) : min {np.min(kept_act):.3f}")


def depth_diagnostics(jsonl_path):
    """Oracle B: per-layer visual token counts, budget check, and DP objective vs the Oracle A mask on the same scores."""
    files = sorted(glob.glob(os.path.join(jsonl_path[: -len(".jsonl")] + "_diag", "*.npz")))
    counts, objs = [], []
    for f in files:
        d = np.load(f)
        if "visual_depth" not in d:
            continue
        live = ~d["warm"]
        counts.append(d["layer_visual_counts"][live])
        objs.append(d["objective"][live])
    if not counts:
        return
    counts, objs = np.concatenate(counts), np.concatenate(objs)
    L, M = counts.shape[1], 256
    print(f"\nOracle B diagnostics over {len(counts)} pruned queries ({len(files)} episodes):")
    print(f"  layer-avg visual retention: {counts.sum(1).mean() / (L * M):.6f} (min {counts.sum(1).min()}, max {counts.sum(1).max()} token-layers)")
    print("  mean visual tokens per layer: " + " ".join(f"{c:.0f}" for c in counts.mean(0)))
    label = "fixed schedule / free-count DP" if "bfix" in jsonl_path else "B / A"
    print(f"  objective {label} (same teacher scores): mean {np.mean(objs[:, 0] / objs[:, 1]):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./experiments/logs/oracle")
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    res, paths = {}, {}
    for m in METHODS:
        r, p = load(args.log_dir, args.tag, m)
        if r is not None:
            res[m], paths[m] = r, p
            print(f"{m:7s}: {p} ({len(r)} episodes)")
    if not res:
        raise SystemExit(f"no *--{args.tag}_<method>.jsonl in {args.log_dir}")
    keys = sorted(set.intersection(*[set(r) for r in res.values()]))
    print(f"paired (task, init_state) keys present in all methods: {len(keys)}\n")

    srs = {m: task_sr(r, keys) for m, r in res.items()}
    tasks = sorted(next(iter(srs.values())))
    print("task  " + "  ".join(f"{m:>7s}" for m in res))
    for t in tasks:
        print(f"{t:4d}  " + "  ".join(f"{srs[m][t]:7.2f}" for m in res))
    print("SR    " + "  ".join(f"{np.mean(list(srs[m].values())):7.3f}" for m in res))

    print()
    pairs = [("oracle", "pruner"), ("oracle", "full"), ("pruner", "full"),
             ("oracle_a", "pruner"), ("oracle_b", "pruner"), ("oracle_b", "oracle_a"), ("oracle_a", "full"), ("oracle_b", "full"),
             ("oracle_b_m2", "oracle_b"), ("oracle_b_m2", "oracle_a"), ("oracle_b_m2", "pruner"),
             ("oracle_bfix", "oracle_b_m2"), ("oracle_bfix", "pruner"), ("oracle_bfix", "full"),
             ("oracle_bfix_mix", "oracle_bfix"), ("oracle_bfix_mix", "pruner"), ("oracle_bfix_mix", "full"),
             ("predictor", "pruner"), ("predictor", "oracle_bfix"), ("predictor", "oracle_b_m2"), ("predictor", "full")]
    for a, b in pairs:
        if a in res and b in res:
            d, lo, hi, a_only, b_only = paired_delta(res[a], res[b], keys, args.n_boot, rng)
            print(f"SR({a}) - SR({b}) = {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"discordant: {a}-only {a_only}, {b}-only {b_only}")

    for m in ("oracle", "oracle_a"):
        if m in paths:
            oracle_diagnostics(paths[m])
    for m in ("oracle_b", "oracle_b_m2", "oracle_bfix"):
        if m in paths:
            print(f"\n[{m}]", end="")
            depth_diagnostics(paths[m])


if __name__ == "__main__":
    main()
