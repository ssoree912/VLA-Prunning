"""specprune_flags.py -- SpecPrune with flag overrides: shard_eval.py with SpecPrune's GenerateConfig class flags overridden from env
SPEC_FLAGS, e.g. "token_prune=0,dynamic_prune=0" (these flags are class attributes, not draccus fields)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shard_eval  # noqa: E402  (patches run_task / run_episode)

R = shard_eval.R
for kv in filter(None, os.environ.get("SPEC_FLAGS", "").split(",")):
    k, v = kv.split("=")
    setattr(R.GenerateConfig, k, bool(int(v)))
    print("SPEC_FLAG", k, getattr(R.GenerateConfig, k), flush=True)

if __name__ == "__main__":
    R.eval_libero()
