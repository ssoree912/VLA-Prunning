"""
run_libero_eval.py
Runs a model in a LIBERO simulation environment.
"""

import json
import os
import sys
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import imageio
import draccus
import numpy as np
import tqdm
import wandb

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent
libero_path = project_root / "LIBERO"
if str(libero_path) not in sys.path:
    sys.path.insert(0, str(libero_path))

from libero.libero import benchmark

# Append current directory so that interpreter can find experiments.robot
sys.path.insert(0, str(project_root))
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # Use VLA-Pruner for faster inference
    # FastV Token Pruning Configuration
    use_fastv: bool = True              #enable fastvforward for token pruning
    fastv_k: int = 3                    # Layer to start pruning
    fastv_r: float = 0.50               # Pruning ratio (tokens to remove)
    fastv_image_token_start_index: int = 1  # Image token start index
    fastv_image_token_length: int = 256     # Image token length
    sparsevlm: bool = False         # enable sparsevlm
    #VLA-Pruner Settings
    use_temporal: bool = True      # Whether to use temporal attention guidance
    temporal_w: int = 3           # Temporal window size
    temporal_gamma: float = 0.80  # Temporal weight decay factor
    #use test-to-vison attention or prefill attention
    use_text_vision_selection: bool = False
    use_prefil_attention: bool = False
    # Oracle pruning from the current full-token action->vision KV attention (oracle_pruning/plan/)
    use_oracle: bool = False
    oracle_mode: str = "topk"     # "topk" = Oracle A (TopK_k kept from layer K on), "depth" = Oracle B (per-token depth)
    oracle_min_depth: int = 0     # Oracle B: no visual token is removed before this layer (depth_i >= m)
    oracle_schedule: Optional[str] = None  # Oracle B-fixed (oracle_mode "schedule"): json with per-layer counts "n_l"
    oracle_log_teacher: bool = False  # predictor mode: also run the dense teacher each query and log its scores
    oracle_label_mix: float = 0.0  # Oracle B-fixed: weight of instruction->vision attention in the ranking (0 = action only)
    oracle_predictor_ckpt: Optional[str] = None  # oracle_mode "predictor": trained ranker (oracle_scripts/train_predictor.py)
    oracle_layers: str = "post_k" # Oracle A score layers: "post_k" = K..31, "late" = 16..31 (Oracle-Act-TopK v0),
                                  # "pruner" = layer 15 only, or "a,b,c". Unused by Oracle B (all layers).
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task
    init_state_offset: int = 0                       # First LIBERO initial state index (pilot 0, confirmation 10)
    task_ids: Optional[str] = None                   # Comma-separated task ids to run (default: all)
    save_video: bool = True                          # Save rollout MP4s
    save_diagnostics: bool = True                    # Save per-query token selections / scores (npz per episode)
    save_observations: bool = False                  # Save raw 224x224 query images for offline K/label diagnosis
    obs_every: int = 5                               # Keep every n-th query image (queries 0, n, 2n, ...)
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def save_query_diagnostics(path, query_infos):
    """
    Stack per-query selection info; queries without a field (e.g. warm start) get NaN / -1 rows. 1-D fields are
    right-padded to the widest query; N-D fields (s_layer) must share one shape and keep their dtype.
    """
    out = {"env_step": np.array([q["env_step"] for q in query_infos]),
           "warm": np.array([q.get("warm", False) for q in query_infos])}
    keys = {k for q in query_infos for k, v in q.items() if isinstance(v, np.ndarray)}
    for key in sorted(keys):
        vals = [q.get(key) if isinstance(q.get(key), np.ndarray) else None for q in query_infos]
        first = next(v for v in vals if v is not None)
        is_float = np.issubdtype(first.dtype, np.floating)
        fill = np.nan if is_float else -1
        if first.ndim > 1:
            out[key] = np.stack([v if v is not None else np.full(first.shape, fill, dtype=first.dtype) for v in vals])
            continue
        width = max(len(v) for v in vals if v is not None)  # kept_visual: 256 on warm-start queries, k afterwards
        rows = []
        for v in vals:
            row = np.full(width, fill, dtype=np.float32 if is_float else np.int64)
            if v is not None:
                row[: len(v)] = v
            rows.append(row)
        out[key] = np.stack(rows)
    np.savez_compressed(path, **out)


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    # Set random seed
    set_seed_everywhere(cfg.seed)
    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    # Resolve Oracle-Act-TopK layer set (fixed before the run; see plan section 3)
    if cfg.use_oracle:
        assert cfg.oracle_mode in ("topk", "depth", "schedule", "predictor"), f"unknown oracle_mode {cfg.oracle_mode}"
        if cfg.oracle_mode == "schedule":
            cfg.oracle_schedule_n_l = json.load(open(cfg.oracle_schedule))["n_l"]
        if cfg.oracle_layers == "post_k":
            cfg.oracle_layer_ids = list(range(cfg.fastv_k, 32))
        elif cfg.oracle_layers == "late":
            cfg.oracle_layer_ids = list(range(16, 32))
        elif cfg.oracle_layers == "pruner":
            cfg.oracle_layer_ids = [15]
        else:
            cfg.oracle_layer_ids = [int(x) for x in cfg.oracle_layers.split(",")]
    # Load model
    model = get_model(cfg)
    # Layer-averaged visual-token retention sum_l n_l / (L*M), the budget shared by VLA-Pruner, Oracle A and Oracle B
    if cfg.use_fastv:
        n_layers, n_visual = len(model.language_model.model.layers), cfg.fastv_image_token_length
        k_keep = int(round(n_visual * (1 - cfg.fastv_r))) if not cfg.use_oracle else int(n_visual * (1 - cfg.fastv_r) + 1e-6)
        visual_keep_layer_avg = (cfg.fastv_k * n_visual + (n_layers - cfg.fastv_k) * k_keep) / (n_layers * n_visual)
    else:
        visual_keep_layer_avg = 1.0
    if cfg.use_oracle:
        n_layers = len(model.language_model.model.layers)
        assert n_layers == 32 and max(cfg.oracle_layer_ids) < n_layers, f"unexpected LLM depth {n_layers}"
    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    results_filepath = os.path.join(cfg.local_log_dir, run_id + ".jsonl")
    results_file = open(results_filepath, "w")
    diag_dir = os.path.join(cfg.local_log_dir, run_id + "_diag")
    if cfg.save_diagnostics:
        os.makedirs(diag_dir, exist_ok=True)
    print(f"Logging to local log file: {local_log_filepath}")
    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )
    log_file.write("=" * 80 + "\n")
    log_file.write("CONFIGURATION SUMMARY:\n")
    log_file.write("=" * 80 + "\n")
    for field_name, field_value in cfg.__dict__.items():
        log_file.write(f"  {field_name}: {field_value}\n")
    log_file.write("=" * 80 + "\n")
    log_file.write("\n")
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_id_list = range(num_tasks_in_suite) if cfg.task_ids is None else [int(x) for x in str(cfg.task_ids).split(",")]
    for task_id in tqdm.tqdm(task_id_list):
        # Get task
        task = task_suite.get_task(task_id)
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        # Start episodes
        task_episodes, task_successes = 0, 0
        assert cfg.init_state_offset + cfg.num_trials_per_task <= len(initial_states), (
            f"only {len(initial_states)} initial states for task {task_id}"
        )
        for episode_idx in tqdm.tqdm(range(cfg.init_state_offset, cfg.init_state_offset + cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            # Reset environment
            env.reset()
            model.reset_av_history()
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            # Setup
            t = 0
            replay_images = []
            replay_images_heatmap = []
            prev_img = None
            last_caches = None
            query_infos = []
            saved_obs, saved_obs_steps, saved_actions = [], [], []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue
                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    # Save previous image
                    if prev_img is None:
                        prev_img = img
                    else:
                        prev_img = replay_images[-1]
                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "prev_image": prev_img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    # Query model to get action
                    action, last_caches, result_image = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        last_caches=last_caches,
                    )
                    replay_images_heatmap.append(result_image)
                    query_infos.append(dict(model.last_query_info or {}, env_step=t))
                    if cfg.save_observations and (len(query_infos) - 1) % cfg.obs_every == 0:
                        saved_obs.append(img)
                        saved_obs_steps.append(t)
                        saved_actions.append(np.array(action, dtype=np.float32))
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)
                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)
                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
            task_episodes += 1
            total_episodes += 1

            if cfg.save_video:
                save_rollout_video(
                            replay_images_heatmap, total_episodes, success=done, task_description=task_description, log_file=log_file
                        )
            if cfg.save_observations and saved_obs:
                obs_dir = os.path.join(cfg.local_log_dir, run_id + "_obs")
                os.makedirs(obs_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(obs_dir, f"task{task_id:02d}_init{episode_idx:02d}.npz"),
                    images=np.stack(saved_obs), env_step=np.array(saved_obs_steps), raw_action=np.stack(saved_actions),
                    task_description=task_description, task_id=task_id, init_state=episode_idx, success=bool(done),
                )
            if cfg.save_diagnostics:
                save_query_diagnostics(os.path.join(diag_dir, f"task{task_id:02d}_init{episode_idx:02d}.npz"), query_infos)
            results_file.write(json.dumps({
                "task_id": task_id, "init_state": episode_idx, "task": task_description, "success": bool(done),
                "env_steps": t, "num_queries": len(query_infos), "seed": cfg.seed,
                "method": "oracle" if cfg.use_oracle else ("vla_pruner" if cfg.use_fastv else "full"),
                "oracle_mode": cfg.oracle_mode if cfg.use_oracle else None,
                "oracle_min_depth": cfg.oracle_min_depth if cfg.use_oracle and cfg.oracle_mode == "depth" else None,
                "oracle_schedule": cfg.oracle_schedule_n_l if cfg.use_oracle and cfg.oracle_mode == "schedule" else None,
                "oracle_label_mix": cfg.oracle_label_mix if cfg.use_oracle else None,
                "oracle_predictor_ckpt": cfg.oracle_predictor_ckpt if cfg.use_oracle and cfg.oracle_mode == "predictor" else None,
                "fastv_r": cfg.fastv_r if cfg.use_fastv else 0.0, "checkpoint": str(cfg.pretrained_checkpoint),
                "visual_keep_layer_avg": visual_keep_layer_avg,
            }) + "\n")
            results_file.flush()
            # Save a replay video of the episode
            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()
        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
    # Save local log file
    log_file.close()
    results_file.close()
    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

if __name__ == "__main__":
    eval_libero()
