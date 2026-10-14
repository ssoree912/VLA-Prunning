"""
run_vlapruner.py -- the upstream OpenVLA-OFT LIBERO eval (experiments/robot/libero/run_libero_eval.py, unmodified) plus
  * per-episode jsonl (task_id, init_state, success, queries) for paired comparisons,
  * optional learned depth predictor (env OFT_PREDICTOR_CKPT): rides on the VLA-Pruner path, so the warm start
    (first w queries with full tokens) and the token metadata are identical to VLA-Pruner; every pruned query runs
    the depth forward instead of the VLA-Pruner single cut.

Env:  OFT_RESULTS_JSONL (required), OFT_METHOD (label), OFT_PREDICTOR_CKPT (optional),
      OFT_TASK_IDS / OFT_INIT_OFFSET (optional task subset and first initial state)
Args: forwarded to the upstream draccus config, e.g.
    CUDA_VISIBLE_DEVICES=3 OFT_RESULTS_JSONL=... OFT_METHOD=pruner python eval/run_vlapruner.py \
        --pretrained_checkpoint <ckpt> --task_suite_name libero_spatial --use_vla_cache False --use_vla_pruner True \
        --fastv_k 3 --fastv_r 0.875 --vla_pruner_layer 15 --vla_pruner_mode semantic_action --num_trials_per_task 10
(cwd = src/openvla-oft of this repo or of VLA-Pruner)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from gazeprune.vlapruner_path import install_depth_forward, register_vla_cache_stub  # noqa: E402

register_vla_cache_stub()
import experiments.robot.libero.run_libero_eval as R
from gazeprune.predictor import TokenDepthPredictor  # noqa: E402

RESULTS = os.environ["OFT_RESULTS_JSONL"]
METHOD = os.environ.get("OFT_METHOD", "unknown")
PRED_CKPT = os.environ.get("OFT_PREDICTOR_CKPT")
state = {"task_id": None, "episode": 0, "cfg": None}

_initialize_model = R.initialize_model


def initialize_model(cfg):
    out = _initialize_model(cfg)
    state["cfg"] = cfg
    if PRED_CKPT:
        assert cfg.use_vla_pruner, "the predictor uses the VLA-Pruner path (warm start + token metadata)"
        model = out[0]
        install_depth_forward(model, TokenDepthPredictor(PRED_CKPT, next(model.parameters()).device))
    return out


_run_task = R.run_task


TASK_IDS = os.environ.get("OFT_TASK_IDS")  # e.g. "0,1,2,3,4": run only these tasks (sharding)
INIT_OFFSET = int(os.environ.get("OFT_INIT_OFFSET", "0"))  # first LIBERO initial state (episodes use offset + k)


class _OffsetSuite:
    """Task suite view whose init states start at INIT_OFFSET (so episode k uses initial state INIT_OFFSET + k)."""

    def __init__(self, suite):
        self._suite = suite

    def __getattr__(self, name):
        return getattr(self._suite, name)

    def get_task_init_states(self, task_id):
        return self._suite.get_task_init_states(task_id)[INIT_OFFSET:]


def run_task(cfg, task_suite, task_id, *args, **kwargs):
    if TASK_IDS and str(task_id) not in TASK_IDS.split(","):
        # eval_libero passes (model, resize_size, processor, action_head, proprio_projector, noisy_action_projector,
        # total_episodes, total_successes, log_file) positionally: hand the running totals back unchanged
        return (args[6], args[7]) if len(args) >= 8 else (kwargs.get("total_episodes", 0), kwargs.get("total_successes", 0))
    state["task_id"], state["episode"] = task_id, INIT_OFFSET
    return _run_task(cfg, _OffsetSuite(task_suite) if INIT_OFFSET else task_suite, task_id, *args, **kwargs)


_run_episode = R.run_episode


def run_episode(cfg, env, task_description, model, *args, **kwargs):
    lm = model.language_model.model
    if hasattr(lm, "depth_log"):
        lm.depth_log.clear()
    result = _run_episode(cfg, env, task_description, model, *args, **kwargs)
    success, _, _, metrics = result
    rec = {
        "task_id": state["task_id"], "init_state": state["episode"], "task": task_description, "success": bool(success),
        "method": METHOD, "queries": int(metrics["episode_step"]), "policy_time_s": float(metrics["episode_time"]),
        "seed": cfg.seed, "fastv_k": cfg.fastv_k, "fastv_r": cfg.fastv_r if (cfg.use_vla_pruner or cfg.use_fastv) else 0.0,
        "checkpoint": str(cfg.pretrained_checkpoint), "predictor_ckpt": PRED_CKPT,
    }
    if getattr(lm, "depth_log", None):
        counts = np.array(lm.depth_log)
        rec["pruned_queries"] = len(counts)
        rec["visual_token_layers_per_query"] = sorted(set(counts.sum(1).tolist()))
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    state["episode"] += 1
    return result


R.initialize_model, R.run_task, R.run_episode = initialize_model, run_task, run_episode

if __name__ == "__main__":
    R.eval_libero()
