"""Shared FLOPs/latency profiler (baselines/tools/vla_profile.py) for the VLA-Pruner openvla-oft code path: registers the
vla_cache_utils stub (oft_oracle/__init__.py) before the profiler imports experiments.robot.robot_utils.
Usage: python profile_eval.py <out.prof.jsonl> <script.py> [args...]   (cwd = /workspace/vla/VLA-Pruner/src/openvla-oft)"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # /workspace/vla/oracle_pruning
sys.path.insert(0, os.getcwd())
import oft_oracle  # noqa: F401,E402

PROFILER = "/workspace/vla/baselines/tools/vla_profile.py"
sys.argv = [PROFILER] + sys.argv[1:]
runpy.run_path(PROFILER, run_name="__main__")
