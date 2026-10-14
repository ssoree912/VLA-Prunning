"""
shard_eval.py -- run an UNMODIFIED OFT-style LIBERO eval script (experiments/robot/libero/run_libero_eval.py of the
official OFT repo or of a baseline built on it: SpecPrune, VLA-Cache, ...) on a task subset and an initial-state
offset, with a per-episode jsonl, without editing the baseline's code:
  * run_task: tasks not in OFT_TASK_IDS are skipped (running totals handed back unchanged); the task suite is wrapped so
    that episode k uses initial state OFT_INIT_OFFSET + k;
  * run_episode: one jsonl record per episode (task_id, init_state, success).
Env: OFT_RESULTS_JSONL (required), OFT_METHOD, OFT_TASK_IDS, OFT_INIT_OFFSET.
Usage (cwd = the baseline repo, which must be first on PYTHONPATH; usually under the profiler):
    python eval/profiler.py out.prof.jsonl eval/shard_eval.py [draccus args...]
"""

import inspect
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import experiments.robot.libero.run_libero_eval as R  # noqa: E402

RESULTS = os.environ["OFT_RESULTS_JSONL"]
METHOD = os.environ.get("OFT_METHOD", "unknown")
TASK_IDS = os.environ.get("OFT_TASK_IDS")
INIT_OFFSET = int(os.environ.get("OFT_INIT_OFFSET", "0"))
state = {"task_id": None, "episode": 0}


class _OffsetSuite:
    def __init__(self, suite):
        self._suite = suite

    def __getattr__(self, name):
        return getattr(self._suite, name)

    def get_task_init_states(self, task_id):
        return self._suite.get_task_init_states(task_id)[INIT_OFFSET:]


_run_task, _run_episode = R.run_task, R.run_episode
_task_sig = inspect.signature(_run_task)


def run_task(*args, **kwargs):
    b = _task_sig.bind(*args, **kwargs)
    b.apply_defaults()
    a = b.arguments
    if TASK_IDS and str(a["task_id"]) not in TASK_IDS.split(","):
        return a["total_episodes"], a["total_successes"]
    state["task_id"], state["episode"] = a["task_id"], INIT_OFFSET
    if INIT_OFFSET:
        a["task_suite"] = _OffsetSuite(a["task_suite"])
    return _run_task(*b.args, **b.kwargs)


def run_episode(cfg, *args, **kwargs):
    result = _run_episode(cfg, *args, **kwargs)
    rec = {"task_id": state["task_id"], "init_state": state["episode"], "success": bool(result[0]), "method": METHOD,
           "seed": cfg.seed, "suite": str(cfg.task_suite_name), "checkpoint": str(cfg.pretrained_checkpoint)}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    state["episode"] += 1
    return result


R.run_task, R.run_episode = run_task, run_episode

if __name__ == "__main__":
    R.eval_libero()
