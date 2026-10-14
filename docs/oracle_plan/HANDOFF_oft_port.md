# Handoff: OFT experiments + porting our predictor to the official OFT code (2026-10-12)

User communication: Korean; report remaining time as "hours from now" (server clock differs); keep CPU/GPU usage sane,
don't leave GPUs idle. Commit only as ssoree912 (see ~/.claude/CLAUDE.md), push to remote `ssoree`.

## NEXT TASK (asked by the user): port our depth predictor to the official OpenVLA-OFT code on GPU 3 and verify

Why: our OFT runs use the VLA-Pruner openvla-oft code path (`/workspace/vla/VLA-Pruner/src/openvla-oft`, env
`openvla-oft` with VLA-Pruner's patched transformers 4.47). In that fork the SDPA path drops the attention mask (would be
causal), so `oft_oracle/core.py::_depth_forward` runs `output_attentions=True` (eager) in ALL 32 layers -> LLM 73 ms vs
SpecPrune 56 ms (median, same FLOPs). Baselines (SpecPrune, VLA-ADP) run on the official OFT fork.

Plan:
1. Official stack = what SpecPrune/ADP use: transformers fork `v4.40.1-openvla-oft` at
   `/workspace/vla/baselines/_transformers_oft/src` + tokenizers 0.19 at `/workspace/vla/baselines/_pydeps_oft`, added via
   PYTHONPATH only (shared env untouched). An official OFT code base is needed (no pruning): e.g. clone
   https://github.com/moojink/openvla-oft into `/workspace/vla/baselines/openvla-oft-official`, run with cwd/PYTHONPATH =
   that repo (else `prismatic` resolves to VLA-Pruner's editable install). Use a private checkpoint dir with symlinked
   weights (get_vla rewrites modeling_prismatic.py/config.json inside the checkpoint dir; HF caches remote code by dir
   basename, so give it a unique basename, e.g. `/workspace/vla/models/oft-ours/openvla-7b-oft-finetuned-libero-spatial-ours`).
2. Port: replace LlamaModel.forward (language_model.model) with our depth forward: layers 0-1 full tokens (eager +
   output_attentions only there, to get action-query->visual attention rows), predictor at layer-2 input
   (`TokenDepthPredictor` in `oft_oracle/core.py`: inputs visual hidden (PCA), mean action-query hidden, layer0-1 attention
   rows), then per-layer nested pruning by the fixed schedule; layers >= 2 with SDPA + bidirectional mask (no attention
   materialization). Token layout (official OFT): [BOS][512 visual (agentview 256 + wrist 256)][proprio][prompt text]
   [56 action tokens][stop]. Prune from the FIRST query (no warm start).
3. Verify on a few frames (e.g. RLDS demo frames or `VLA-ADP/experiments/robot/libero/sample_libero_spatial_observation.pkl`):
   (a) official full vs VLA-Pruner-env full: actions + per-layer attention (labels) + layer-1 hidden ~equal (if they match,
   labels/predictor need NOT be re-extracted); (b) ported pruned forward vs current `_depth_forward`: same kept tokens /
   depths and ~same actions; (c) latency (clean, one process on GPU 3).
4. Then rerun OFT spatial 200 eps with the port (3 task shards, profiler) and compare with the current result.

## Current results (OFT LIBERO-Spatial, 200 eps = init 0-19 x 10 tasks, seed 7, osmesa)
| method | success | visual retention (layer avg) | LLM FLOPs vs dense | model ms (median) |
|---|---|---|---|---|
| ours (action-only ranker, 20% schedule b3280, no warm start) | 198/200 (99.0%) | 20.0% | 0.316 | 128 |
| SpecPrune (official default) | 188/200 (94.0%) | 20.7% | 0.321 | 103 |
Paired: ours +5.0%p [95% CI +2.0, +8.5], ours-only 12 / SpecPrune-only 2. get_action latency: ours 157 ms, SpecPrune 177.
SpecPrune OFT goal (official default): 190/200 (95.0%), retention 21.5%. OFT full spatial (old run, init 0-9): 100/100.
Logs: `/workspace/vla/eval_logs/oft_spatial200/`, `/workspace/vla/eval_logs/oft_goal200/` (status.txt, *.jsonl,
*.prof.jsonl per query: prefill_q per layer, flops, t_model_ms...). Retention per query = sum_l(q_l - (q_0-512))/(32*512).

## Running jobs at handoff (do not restart)
- GPU 0: OFT goal labels (action + instruction s_instr), 2 shards (`label_rlds.py --num_shards 2 --shard 0/1`), ~2 h left
  at 05:20 UTC; completion is logged to `/workspace/vla/eval_logs/oft_goal200/status.txt` ("END ours labels").
  User said: labels only; train later on request (goal -> two-head, `train.py --two_head --mix 0.5`, eval w=0.4 at the
  SpecPrune-matched budget 21.5%: fastv_r 0.896484375 -> k=53 -> b3532).
- GPU 1: adding s_instr to the OFT spatial labels (`add_instr_loop.sh`), ~30 min left.
- GPU 3: free (for the port).
- Base OpenVLA: VLA-Cache spatial 500 done (415/500, 83.0%); full object stopped at 288/500 (next task5 init38);
  libero_10 ours stopped at 164/500. See `/workspace/vla/oracle_pruning/VLA-Pruner/src/openvla/experiments/logs/oracle/std500_p20_status.txt`.

## Code (OFT pipeline, `/workspace/vla/oracle_pruning/oft_oracle/`)
- `label_rlds.py`: teacher labels from RLDS demos via the real OFT `get_vla_action`; saves s_layer (action->visual attn),
  s_instr (instruction->visual), hv (layer-1 visual hidden, PCA 512), hq, avis, r. Options: --add_instr, --num_shards/--shard.
  Writes atomically (tmp_ file + os.replace). Run with cwd `/workspace/vla/VLA-Pruner/src/openvla-oft`.
- `train.py`: budget-free ranker (listwise KL); budget only sets the stored schedule + checkpoint selection.
  `--two_head --mix 0.5` = base-OpenVLA two-head (action/instruction heads). Predictor ckpt (spatial, action-only):
  `/workspace/vla/data/oft_predictor/libero_spatial_m2_s0/best.pt`.
- `make_schedule.py`: schedule for another budget (inference-time swap via env OFT_SCHEDULE). Schedules in
  `/workspace/vla/data/oft_predictor/schedules/` (spatial b3280 = 20.02%, b3616 = 22.07%).
- `core.py`: DepthRanker (n_out 1/2), combine_heads, TokenDepthPredictor (env OFT_SCHEDULE, OFT_HEAD_WEIGHT),
  install_depth_forward (prunes from query 0; OFT_WARMUP=1 restores VLA-Pruner warm start), _depth_forward.
- `run_eval.py` (env OFT_TASK_IDS, OFT_INIT_OFFSET), `run_ours_eval.sh <gpu> <ckpt> <schedule|""> <tag> [w]` (3 shards),
  `profile_eval.py` (shared profiler wrapper for this code path).
- Baselines launcher: `/workspace/vla/baselines/tools/run_oft200.sh` (SUITE env, SHARD + SPEC_TASK_IDS); profiler
  `/workspace/vla/baselines/tools/vla_profile.py`. SpecPrune edits: GPU env respected, SPEC_TASK_IDS sharding.
- Labels are big because of hv (PCA-512 per visual token per frame): OFT spatial ~33 GB.
