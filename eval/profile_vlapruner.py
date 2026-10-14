"""Shared FLOPs/latency profiler (eval/profiler.py) for the VLA-Pruner openvla-oft code path: registers the
vla_cache_utils stub before the profiler imports experiments.robot.robot_utils.
Usage: python eval/profile_vlapruner.py <out.prof.jsonl> <script.py> [args...]   (cwd = VLA-Pruner src/openvla-oft)"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.getcwd())
from gazeprune.vlapruner_path import register_vla_cache_stub  # noqa: E402

register_vla_cache_stub()

PROFILER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiler.py")
sys.argv = [PROFILER] + sys.argv[1:]
runpy.run_path(PROFILER, run_name="__main__")
