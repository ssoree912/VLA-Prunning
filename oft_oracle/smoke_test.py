"""
smoke_test.py -- checks for the OFT depth forward before closed-loop evaluation (one RLDS frame).

  1. depth = L for every token through the depth forward == vanilla OFT forward (actions and action hidden states)
  2. predictor inputs inside the forward == the features stored by label_rlds.py for the same frame
  3. per-layer visual counts == schedule, sum == budget; the action chunk is finite

Run (openvla-oft env, cwd = /workspace/vla/VLA-Pruner/src/openvla-oft):
    CUDA_VISIBLE_DEVICES=1 python /workspace/vla/oracle_pruning/oft_oracle/smoke_test.py --ckpt <best.pt>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import oft_oracle  # noqa: F401
from oft_oracle.core import TokenDepthPredictor, install_depth_forward, query_features
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.openvla_utils import get_vla_action, resize_image_for_policy

CKPT = "/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-spatial"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()
    cfg = GenerateConfig(pretrained_checkpoint=CKPT, task_suite_name="libero_spatial", use_vla_cache=False,
                         use_fastv=False, use_vla_pruner=True, fastv_k=3, fastv_r=0.875, vla_pruner_layer=15,
                         vla_pruner_mode="semantic_action")
    vla, ah, pp, _, proc = initialize_model(cfg)
    pred = TokenDepthPredictor(args.ckpt, next(vla.parameters()).device)
    L = len(vla.language_model.model.layers)

    ds = tfds.builder_from_directory("/workspace/vla/data/modified_libero_rlds/libero_spatial_no_noops/1.0.0").as_dataset(split="train")
    step = next(iter(next(iter(ds))["steps"]))
    instr = step["language_instruction"].numpy().decode()

    def obs():
        return {"full_image": resize_image_for_policy(step["observation"]["image"].numpy(), 224),
                "wrist_image": resize_image_for_policy(step["observation"]["wrist_image"].numpy(), 224),
                "state": step["observation"]["state"].numpy().astype(np.float64)}

    def act(pruned):
        """pruned=False: vanilla OFT forward; True: fastv path with fastv_r > 0 (-> installed depth forward)."""
        for o in (vla, vla.config):
            o.use_vla_pruner = o.use_fastv = pruned
            o.use_temporal = False  # no warm start inside the smoke test
        with torch.no_grad():
            a, lc, _, _ = get_vla_action(cfg, vla, proc, obs(), instr, ah, pp, None, use_film=False, last_caches=None)
        return np.stack(a), lc

    captured = {}
    hook = vla.language_model.model.layers[pred.layer - 1].register_forward_hook(
        lambda m, i, o: captured.__setitem__("h", o[0].detach().clone()))
    a_full, lc = act(False)
    hook.remove()
    tm = lc["token_metadata"]
    vs, M, a0, a1 = tm["visual_token_start"], tm["num_visual_tokens"], tm["action_token_start"], tm["action_token_end"]
    rows = torch.cat([lc["attentions"][l][:, :, a0:a1, vs:vs + M].float().mean(dim=2) for l in range(pred.layer)], 1)
    ref = query_features(captured["h"], rows, vs, M, a0, a1)
    ok = []

    class Full:  # depth = L for all tokens, at the predictor's layer
        layer, M_ = pred.layer, M

        def __call__(self, h, a, img_start, s, e):
            return torch.full((M,), L, device=h.device)
    full = Full()
    full.M = M
    install_depth_forward(vla, full)
    a_depthL, _ = act(True)
    d = float(np.abs(a_depthL - a_full).max())
    ok.append(d == 0.0)
    print(f"[{'PASS' if ok[-1] else 'FAIL'}] depth forward with depth=L == vanilla OFT (max|action diff| {d:.2e})")

    seen = {}

    def spy(h, a, img_start, s, e):
        seen["f"] = query_features(h, a, img_start, M, s, e)
        return pred(h, a, img_start, s, e)
    spy.layer, spy.M = pred.layer, M
    install_depth_forward(vla, spy)
    a_pred, _ = act(True)
    diffs = [float((x - y).abs().max() / (y.abs().max() + 1e-12)) for x, y in zip(seen["f"], ref)]
    ok.append(max(diffs) < 1e-3)
    print(f"[{'PASS' if ok[-1] else 'FAIL'}] in-loop predictor inputs == labeling features (rel diff hv/hq/avis "
          f"{[f'{x:.1e}' for x in diffs]})")
    pi = vla.language_model.model.pruning_info
    counts = pi["layer_visual_counts"]
    ok.append(counts == pred.schedule["n_l"] and sum(counts) == pred.schedule["budget"] and np.isfinite(a_pred).all())
    print(f"[{'PASS' if ok[-1] else 'FAIL'}] counts == schedule, sum {sum(counts)} == budget {pred.schedule['budget']}; "
          f"action[0] {a_pred[0].round(3)} vs full {a_full[0].round(3)}")
    print(f"{sum(ok)}/{len(ok)} passed")


if __name__ == "__main__":
    main()
