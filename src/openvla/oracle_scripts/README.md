# Oracle-Act-TopK (base OpenVLA) — Phase 0

Plan: `/workspace/vla/oracle_pruning/plan/Oracle.md`. Base OpenVLA LIBERO checkpoints, no OFT.

## What changed vs upstream VLA-Pruner (`src/openvla`)

- `prismatic/extern/hf/modeling_prismatic.py` — `_generate_oracle()`: after the same w=3 full-token warm start,
  each query runs (1) a full-token `generate()` → `S_act` = mean softmax attention of the D action-predicting queries
  to the 256 visual tokens over layers 16..31 (0-based) and all heads; (2) a second `generate()` that keeps exactly
  `TopK_k(S_act)` (ties → smaller index) from layer K=3 on. No semantic score, no MMDP. Only the action of (2) is executed.
- `prismatic/extern/hf/oracle_llama.py` — the direct-TopK prefill. Patched onto the oracle model instance only; the
  shared `transformers` in site-packages (used by running VLA-Pruner jobs) is **not** modified. Pruning mechanics
  (position ids, causal mask, KV cache, decode) mirror the official `fastv_forward`.
- `experiments/robot/libero/run_libero_eval.py` — flags `--use_oracle`, `--oracle_layers {late|pruner|a,b,c}`,
  `--init_state_offset`, `--task_ids`, `--save_video`, `--save_diagnostics`; writes `<run_id>.jsonl`
  (one line per episode) and `<run_id>_diag/*.npz` (per query: S_act, TopK(S_act), past-EMA TopK, kept indices).
- VLA-Pruner baseline = official path unchanged (EMA of layer 15 + prefill semantic + MMDP).

## Order of runs (from `src/openvla`)

```bash
CUDA_VISIBLE_DEVICES=0 python oracle_scripts/smoke_test.py --check_sdpa   # single-observation checks (plan 5.1)
bash oracle_scripts/run_smoke_episodes.sh 0                              # 2 episodes x 3 methods, task 0
bash oracle_scripts/run_pilot.sh 0 1 2                                   # init states 0-9, 100 ep/method
bash oracle_scripts/run_confirm.sh 0 1 2                                 # init states 10-49, 400 ep/method
python oracle_scripts/analyze.py --tag pilot                             # paired SR deltas + bootstrap CI
```

Budget: `RHO_R=0.75 bash oracle_scripts/run_pilot.sh` for ρ=0.25 (k=64).

## Oracle A / Oracle B (same teacher scores, same token-layer budget)

Both take the per-layer teacher score `s[l, i]` (`_action_vision_layer_scores`): softmax attention from the D
action-predicting queries (last prompt position + decode steps) to visual token i's KV at layer l, averaged over
heads and queries, not renormalized to the image. Budget `B = K*M + (L-K)*k` = 1696 token-layers at K=3, k=32
(layer-avg retention 20.703%).

- `--oracle_mode topk --oracle_layers post_k` (**Oracle A**, method `oracle_a`): TopK_k of `s[K:].mean(0)`, kept from
  layer K to the end. Same single-cut prefill as Oracle-Act-TopK v0 (`oracle`, layers 16..31).
- `--oracle_mode depth` (**Oracle B**, method `oracle_b`): `oracle_depth_dp` picks a depth `d_i ∈ [0, L]` per visual
  token maximizing `Σ_i Σ_{l<d_i} s[l, i]` with `Σ_i d_i = B` (exact DP); `_oracle_depth_prefill` drops token i before
  layer `d_i`, so the kept sets are nested and KV lengths differ per layer. Decode positions are corrected when tokens
  are dropped before layer 0.
- Diagnostics per query: `s_layer` (L×M, fp16) for both; A adds `s_act`, `topk_act`; B adds `visual_depth`,
  `layer_visual_counts`, `objective` = [B objective, Oracle A objective on the same scores].

```bash
bash -c 'source oracle_scripts/common.sh; run_method 0 oracle_a pilot --num_trials_per_task 10 --save_video False'
bash -c 'source oracle_scripts/common.sh; run_method 1 oracle_b pilot --num_trials_per_task 10 --save_video False'
python oracle_scripts/analyze.py --tag pilot   # adds oracle_b-oracle_a deltas and Oracle B layer counts
```
