"""OpenVLA-OFT depth-predictor pipeline. Importing this package first registers a stub for the VLA-Cache helper module
that the upstream VLA-Pruner OFT code imports but does not ship (only used when use_vla_cache=True, never here)."""

import sys
import types


def _vla_cache_stub():
    name = "experiments.robot.vla_cache_utils"
    if name in sys.modules:
        return
    mod = types.ModuleType(name)

    def _missing(*args, **kwargs):
        raise NotImplementedError("experiments.robot.vla_cache_utils is not shipped; run with use_vla_cache=False")

    for fn in ("find_static_patches", "task_relevant_selection", "get_layer_mask_schedule", "draw_patches_overlay"):
        setattr(mod, fn, _missing)
    sys.modules[name] = mod


_vla_cache_stub()
