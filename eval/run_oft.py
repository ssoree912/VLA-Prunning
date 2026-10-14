"""
run_oft.py -- the OFFICIAL OpenVLA-OFT LIBERO eval (moojink/openvla-oft experiments/robot/libero/
run_libero_eval.py, unmodified) plus
  * per-episode jsonl (task_id, init_state, success, queries) for paired comparisons,
  * optional learned depth predictor (env OFT_PREDICTOR_CKPT, OFT_SCHEDULE, OFT_HEAD_WEIGHT): gazeprune/oft.py replaces the
    LlamaModel forward; every query (from the first one) is pruned.

Env:  OFT_RESULTS_JSONL (required), OFT_METHOD (label), OFT_PREDICTOR_CKPT (optional),
      OFT_TASK_IDS / OFT_INIT_OFFSET (optional task subset and first initial state)
Run through scripts/eval.sh (official OFT repo as cwd / PYTHONPATH head, official transformers fork).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import experiments.robot.libero.run_libero_eval as R
from gazeprune.oft import install_official_depth_forward
from gazeprune.predictor import TokenDepthPredictor

RESULTS = os.environ["OFT_RESULTS_JSONL"]
METHOD = os.environ.get("OFT_METHOD", "unknown")
PRED_CKPT = os.environ.get("OFT_PREDICTOR_CKPT") or None
TASK_IDS = os.environ.get("OFT_TASK_IDS")  # e.g. "0,1,2,3": run only these tasks (sharding)
INIT_OFFSET = int(os.environ.get("OFT_INIT_OFFSET", "0"))  # first LIBERO initial state (episodes use offset + k)
state = {"task_id": None, "episode": 0}

_initialize_model = R.initialize_model


def initialize_model(cfg):
    out = _initialize_model(cfg)
    model = out[0]
    pred = TokenDepthPredictor(PRED_CKPT, next(model.parameters()).device) if PRED_CKPT else None
    install_official_depth_forward(model, pred)
    return out


class _OffsetSuite:
    """Task suite view whose init states start at INIT_OFFSET (so episode k uses initial state INIT_OFFSET + k)."""

    def __init__(self, suite):
        self._suite = suite

    def __getattr__(self, name):
        return getattr(self._suite, name)

    def get_task_init_states(self, task_id):
        return self._suite.get_task_init_states(task_id)[INIT_OFFSET:]


_run_task = R.run_task


def run_task(cfg, task_suite, task_id, model, resize_size, processor=None, action_head=None, proprio_projector=None,
             noisy_action_projector=None, total_episodes=0, total_successes=0, log_file=None):
    if TASK_IDS and str(task_id) not in TASK_IDS.split(","):
        return total_episodes, total_successes
    state["task_id"], state["episode"] = task_id, INIT_OFFSET
    return _run_task(cfg, _OffsetSuite(task_suite) if INIT_OFFSET else task_suite, task_id, model, resize_size,
                     processor, action_head, proprio_projector, noisy_action_projector, total_episodes, total_successes,
                     log_file)


_run_episode = R.run_episode


def run_episode(cfg, env, task_description, model, *args, **kwargs):
    lm = model.language_model.model
    lm.depth_log.clear()
    lm.n_queries = 0
    result = _run_episode(cfg, env, task_description, model, *args, **kwargs)
    rec = {
        "task_id": state["task_id"], "init_state": state["episode"], "task": task_description,
        "success": bool(result[0]), "method": METHOD, "queries": int(lm.n_queries), "seed": cfg.seed,
        "checkpoint": str(cfg.pretrained_checkpoint), "predictor_ckpt": PRED_CKPT,
        "schedule": os.environ.get("OFT_SCHEDULE"), "stack": "official",
    }
    if lm.depth_log:
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
