"""
label_rlds.py -- teacher labels for the Oracle B depth predictor from the OpenVLA LIBERO RLDS training set.

For every demo frame: the image is preprocessed exactly like the LIBERO eval path (JPEG round trip + lanczos resize
to 224, then the 0.9-area center crop at query time), the frozen dense OpenVLA generates the D action tokens, and the
per-layer action->vision KV attention s[l, i] (L x M, `_action_vision_layer_scores`) is stored. Oracle depths are not
stored: they are a DP over s (any min_depth / budget) and are recomputed at training time.

One npz per episode in --out_dir (resumable):
    images      (T, 224, 224, 3) uint8   eval-path image before the center crop (what run_libero_eval passes on)
    s_layer     (T, L, M) float16        teacher label
    action_ids  (T, D) int64             dense teacher action tokens
    demo_action (T, D) float32           RLDS action (for the sanity check below)
    instruction str, episode int

Sanity check printed per episode: normalized L1 error between the teacher's unnormalized action and the demo action
(the checkpoint was finetuned on these frames, so it should be small if preprocessing matches).

Run from src/openvla:
    CUDA_VISIBLE_DEVICES=1 python oracle_scripts/label_rlds.py --suite libero_spatial
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # TF only decodes / resizes images here; keep it off the GPU

import tensorflow_datasets as tfds
import torch
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from experiments.robot.libero.libero_utils import resize_image
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import DEVICE, get_processor, get_vla, process_image
from experiments.robot.robot_utils import set_seed_everywhere


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--rlds_dir", default="/workspace/vla/data/modified_libero_rlds")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1, help="split demos across processes: ep_idx % num_shards")
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    out_dir = Path(args.out_dir or f"/workspace/vla/data/oracle_labels/{args.suite}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GenerateConfig(pretrained_checkpoint=f"checkpoints/openvla-7b-finetuned-{args.suite.replace('_', '-')}",
                         task_suite_name=args.suite, use_fastv=False)
    cfg.unnorm_key = args.suite
    set_seed_everywhere(cfg.seed)
    vla = get_vla(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f"{cfg.unnorm_key}_no_noops" in vla.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    processor = get_processor(cfg)
    D = vla.get_action_dim(cfg.unnorm_key)
    # Visual token span (set only on the use_fastv path in get_vla): inserted right after <BOS>
    vla.fastv_image_token_start_index, vla.fastv_image_token_length = 1, cfg.fastv_image_token_length
    stats = vla.get_action_stats(cfg.unnorm_key)
    scale = np.maximum(np.array(stats["q99"]) - np.array(stats["q01"]), 1e-6)

    ds = tfds.builder_from_directory(f"{args.rlds_dir}/{args.suite}_no_noops/1.0.0").as_dataset(split="train")
    t0, n_frames = time.time(), 0
    for ep_idx, episode in enumerate(ds):
        if args.max_episodes is not None and ep_idx >= args.max_episodes:
            break
        path = out_dir / f"ep{ep_idx:04d}.npz"
        if ep_idx % args.num_shards != args.shard or path.exists():
            continue
        images, s_layers, action_ids, demo_actions = [], [], [], []
        instruction = None
        for step in episode["steps"]:
            instruction = step["language_instruction"].numpy().decode()
            img = resize_image(step["observation"]["image"].numpy(), (224, 224))
            image = process_image(Image.fromarray(img).convert("RGB")) if cfg.center_crop else Image.fromarray(img)
            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
            inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
            input_ids = inputs["input_ids"]
            if not torch.all(input_ids[:, -1] == 29871):  # same suffix token as predict_action
                input_ids = torch.cat((input_ids, torch.tensor([[29871]], device=input_ids.device)), dim=1)
            with torch.no_grad():
                out = vla.generate(input_ids, pixel_values=inputs["pixel_values"], attention_mask=None,
                                   max_new_tokens=D, do_sample=False, return_dict_in_generate=True,
                                   output_attentions=True)
            assert out.attentions[0][0].shape[-1] == input_ids.shape[1] + vla.fastv_image_token_length
            s_layers.append(vla._action_vision_layer_scores(out.attentions).cpu().numpy().astype(np.float16))
            action_ids.append(out.sequences[0, -D:].cpu().numpy())
            images.append(img)
            demo_actions.append(step["action"].numpy().astype(np.float32))
            del out
        ids = np.stack(action_ids)
        # Unnormalize teacher tokens exactly like predict_action (gripper stays in [0, 1] like the RLDS action)
        bins = np.clip(vla.vocab_size - ids - 1, 0, vla.bin_centers.shape[0] - 1)
        norm = vla.bin_centers[bins]
        mask = np.array(stats.get("mask", np.ones(D, dtype=bool)))
        teacher = np.where(mask, 0.5 * (norm + 1) * scale + np.array(stats["q01"]), norm)
        demo = np.stack(demo_actions)
        err = np.abs(teacher - demo)[:, mask] / scale[mask]
        np.savez_compressed(path, images=np.stack(images), s_layer=np.stack(s_layers), action_ids=ids,
                            demo_action=demo, instruction=instruction, episode=ep_idx)
        n_frames += len(images)
        rate = n_frames / (time.time() - t0)
        print(f"ep {ep_idx:4d}  T={len(images):3d}  frames {n_frames}  {rate:.2f} fr/s  "
              f"teacher-vs-demo |a|/range median {np.median(err):.3f}  '{instruction[:50]}'", flush=True)


if __name__ == "__main__":
    main()
