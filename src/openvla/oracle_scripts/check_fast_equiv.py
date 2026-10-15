"""check_fast_equiv.py -- runs the base OpenVLA LIBERO eval with the learned predictor and, at every query, runs
predict_action twice on the same input (ORACLE_FAST off, then on, model state restored in between) and compares the
action tokens / actions, kept-visual counts and depths. The rollout continues with the fast-path actions.
Usage (cwd src/openvla): python check_fast_equiv.py <out.json> run_libero_eval.py [draccus args...]"""
import copy
import json
import runpy
import sys
import time

import numpy as np
import torch

out_path, script, args = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, ".")
import experiments.robot.robot_utils as ru  # noqa: E402
import prismatic.extern.hf.oracle_llama as OL  # noqa: E402

stats = {"queries": 0, "action_equal": 0, "depth_equal": 0, "counts_equal": 0, "max_abs_diff": 0.0,
         "t_slow_ms": [], "t_fast_ms": []}
orig_get_model = ru.get_model


def get_model(*a, **k):
    vla = orig_get_model(*a, **k)
    orig = vla.predict_action
    lm = vla.language_model.model

    def timed(fast, *aa, **kk):
        OL.FAST = fast
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig(*aa, **kk)
        torch.cuda.synchronize()
        info = copy.deepcopy(getattr(vla, "last_query_info", None))
        return out, info, (time.perf_counter() - t0) * 1e3

    def predict_action(*aa, **kk):
        saved = {n: copy.copy(getattr(vla, n)) for n in ("av_hist", "last_query_info") if hasattr(vla, n)}
        saved_off = getattr(lm, "_oracle_pos_offset", 0)
        a_s, i_s, t_s = timed(False, *aa, **kk)
        for n, v in saved.items():
            setattr(vla, n, v)
        lm._oracle_pos_offset = saved_off
        a_f, i_f, t_f = timed(True, *aa, **kk)
        stats["queries"] += 1
        out_f = a_f
        x_s = np.asarray(a_s[0] if isinstance(a_s, tuple) else a_s, dtype=np.float64)
        x_f = np.asarray(a_f[0] if isinstance(a_f, tuple) else a_f, dtype=np.float64)
        stats["action_equal"] += int(np.array_equal(x_s, x_f))
        stats["max_abs_diff"] = max(stats["max_abs_diff"], float(np.abs(x_s - x_f).max()))
        d_s = None if not i_s else i_s.get("visual_depth"); d_f = None if not i_f else i_f.get("visual_depth")
        stats["depth_equal"] += int((d_s is None and d_f is None) or (d_s is not None and d_f is not None and np.array_equal(d_s, d_f)))
        c_s = None if not i_s else i_s.get("layer_visual_counts"); c_f = None if not i_f else i_f.get("layer_visual_counts")
        stats["counts_equal"] += int((c_s is None and c_f is None) or (c_s is not None and c_f is not None and np.array_equal(c_s, c_f)))
        stats["t_slow_ms"].append(t_s); stats["t_fast_ms"].append(t_f)
        if stats["queries"] % 20 == 0:
            json.dump({k: (v if not isinstance(v, list) else float(np.median(v))) for k, v in stats.items()}, open(out_path, "w"))
        return out_f

    vla.predict_action = predict_action
    return vla


ru.get_model = get_model
sys.argv = [script] + args
try:
    runpy.run_path(script, run_name="__main__")
finally:
    json.dump({k: (v if not isinstance(v, list) else float(np.median(v))) for k, v in stats.items()}, open(out_path, "w"), indent=1)
