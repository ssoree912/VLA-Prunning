# GazePrune

Per-token depth pruning of the visual tokens of vision-language-action models, driven by **where the action queries
look**. A small ranker reads the hidden states at the input of layer 2 and orders the visual tokens; a fixed per-layer
schedule — obtained from an exact depth oracle on teacher attention labels — decides how many tokens each layer keeps.
One forward per query, no teacher pass at inference, no per-layer host–GPU synchronization.

Models: [OpenVLA-OFT](https://github.com/moojink/openvla-oft) (LIBERO checkpoints) and base OpenVLA.
Built on the [VLA-Pruner](https://github.com/MINT-SJTU/VLA-Pruner) code base (kept under `src/`).

## Layout

```text
VLA-Prunning/
├── gazeprune/       library: ranker, oracle depth DP + schedules, predictor, official-OFT depth forward, paths
├── teacher/         teacher labels + ranker features from the LIBERO RLDS demos (OFT)
├── train/           ranker training (budget-free) and per-budget schedules
├── eval/            eval entry points (ours / Full on the official OFT stack, baseline wrappers) + shared profiler
├── analysis/        latency breakdown, port verification
├── tools/           label maintenance (drop arrays, truncate PCA features)
├── scripts/         common.sh, label.sh, train.sh, eval.sh, latency.sh, gpu_slots.sh
├── patches/         local edits to the baseline repos (official OFT, SpecPrune, VLA-Cache)
├── docs/            method.md (labels / features / training / inference), benchmark.md (measurement protocol)
└── src/             VLA-Pruner code: openvla/ (base OpenVLA + GazePrune integration, oracle_scripts/), openvla-oft/
```

Checkpoints, labels, predictors and logs are not tracked. `scripts/common.sh` and `gazeprune/paths.py` read
`GAZEPRUNE_ROOT` (default `/workspace/vla`), `GAZEPRUNE_DATA`, `GAZEPRUNE_MODELS`, `GAZEPRUNE_LOGS` and
`GAZEPRUNE_BASELINES`.

## Setup (OpenVLA-OFT)

- Environment: the OpenVLA-OFT conda env (`openvla-oft`) with LIBERO.
- Official stack, added through `PYTHONPATH` only: the official OFT repo (`$GAZEPRUNE_BASELINES/openvla-oft-official`)
  and the transformers fork `v4.40.1-openvla-oft` + tokenizers 0.19 (`_transformers_oft/src`, `_pydeps_oft`).
- Baselines (same location): `SpecPrune-VLA`, `vla-cache` (+ its transformers branch `vla-cache-openvla-oft`),
  VLA-Pruner's `src/openvla-oft`. Apply `patches/*.patch` to them.
- One checkpoint copy per method (`$GAZEPRUNE_MODELS/oft-<method>/<ckpt>-<method>`, weights symlinked): `get_vla`
  rewrites files inside the checkpoint dir and HF caches remote code by the dir name.

## Pipeline (OpenVLA-OFT)

```bash
bash scripts/label.sh 0 libero_spatial            # teacher labels + features (official OFT stack)
bash scripts/train.sh 0 libero_spatial 0.35       # two-head ranker (no avis, PCA 256) + w=0.35 schedule at 20%
bash scripts/eval.sh 0 ours libero_spatial        # 500 episodes (50 init states x 10 tasks), seed 7
bash scripts/eval.sh 0 full libero_spatial        # baselines: full | specprune | specprune_nodyn | vlapruner | vlacache
bash scripts/latency.sh 0                         # clean per-query latency / FLOPs / memory of every method
```

Default GazePrune setting: two-head ranker (action / instruction heads), inputs = layer-1 visual hidden (PCA 256),
action-query summary, patch position; head weight w = 0.35; 20 % visual token-layer budget (3280 of 32 x 512).
See `docs/method.md` for the formulas and `docs/benchmark.md` for the measurement protocol.

## Base OpenVLA

The base-OpenVLA integration lives in `src/openvla` (`prismatic/extern/hf/oracle_predictor.py`, `oracle_llama.py`,
`experiments/robot/libero/run_libero_eval.py`) with its scripts in `src/openvla/oracle_scripts/` (labels, features,
two-head training with `--drop avis ains --pca_dim 256`, schedules, 500-episode launcher `std500_p20.sh`).
