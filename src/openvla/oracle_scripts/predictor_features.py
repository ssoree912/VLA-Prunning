"""
predictor_features.py -- cache the inputs of the Oracle B-fixed depth predictor (predictor sits at the layer-1 output,
i.e. right before layer 2, where the m=2 schedule first drops visual tokens).

For every labeled RLDS frame (oracle_scripts/label_rlds.py), run only the vision backbone, projector and LLM layers
0..1 of the frozen OpenVLA and store:
    hv     (T, M, P) float16  visual-token hidden states at the layer-1 output, standardized + PCA to P dims
    hlast  (T, 4096) float16  hidden state of the last prompt token (the only position that has seen image + text;
                              visual tokens precede the instruction, so causally they never see it)
    avis   (T, 2H, M) float16 attention of the last prompt token to each visual token in layers 0 and 1
    r      (T, M) float32     target: r_i = sum_{l >= 2} s[l, i]
    s_layer is left in the label files.

Run from src/openvla:
    CUDA_VISIBLE_DEVICES=1 python oracle_scripts/predictor_features.py --suite libero_spatial
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import DEVICE, get_processor, get_vla, process_image

M, N_PRE = 256, 2  # visual tokens; layers run before the predictor (0 and 1)


@torch.no_grad()
def prefix_features(vla, input_ids, pixel_values):
    """Hidden states after layer N_PRE-1 (B, T, 4096) and last-row attention to visual tokens (B, N_PRE*H, M)."""
    lm = vla.language_model.model
    proj = vla.projector(vla.vision_backbone(pixel_values))
    emb = vla.get_input_embeddings()(input_ids)
    h = torch.cat([emb[:, :1], proj, emb[:, 1:]], dim=1)
    pos = torch.arange(h.shape[1], device=h.device)
    mask = lm._update_causal_mask(None, h, 0)
    rows = []
    for layer in lm.layers[:N_PRE]:
        out = layer(h, attention_mask=mask, position_ids=pos[None].expand(h.shape[0], -1), output_attentions=True,
                    use_cache=False, cache_position=pos)
        h = out[0]
        rows.append(out[1][:, :, -1, 1:1 + M])
    return h, torch.cat(rows, dim=1)


def build_batch(vla, processor, images, instruction, center_crop):
    pil = [process_image(Image.fromarray(im).convert("RGB")) if center_crop else Image.fromarray(im) for im in images]
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor([prompt] * len(pil), pil).to(DEVICE, dtype=torch.bfloat16)
    ids = inputs["input_ids"]
    if not torch.all(ids[:, -1] == 29871):
        ids = torch.cat((ids, torch.full((ids.shape[0], 1), 29871, device=ids.device)), dim=1)
    return ids, inputs["pixel_values"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--pca_dim", type=int, default=512)
    parser.add_argument("--pca_episodes", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    label_dir = Path(f"/workspace/vla/data/oracle_labels/{args.suite}")
    out_dir = Path(f"/workspace/vla/data/oracle_features/{args.suite}")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(label_dir / "ep*.npz")))

    cfg = GenerateConfig(pretrained_checkpoint=f"checkpoints/openvla-7b-finetuned-{args.suite.replace('_', '-')}",
                         task_suite_name=args.suite, use_fastv=False)
    vla = get_vla(cfg)
    processor = get_processor(cfg)

    def episode_features(f):
        d = np.load(f)
        instr = str(d["instruction"])
        H, A = [], []
        for i in range(0, len(d["images"]), args.batch):
            ids, pv = build_batch(vla, processor, d["images"][i:i + args.batch], instr, cfg.center_crop)
            h, a = prefix_features(vla, ids, pv)
            H.append(h.float())
            A.append(a.float())
        return d, torch.cat(H), torch.cat(A)

    # Sanity: the partial forward equals the full model's hidden state after layer 1
    d = np.load(files[0])
    ids, pv = build_batch(vla, processor, d["images"][:1], str(d["instruction"]), cfg.center_crop)
    h, _ = prefix_features(vla, ids, pv)
    with torch.no_grad():
        ref = vla(input_ids=ids, pixel_values=pv, output_hidden_states=True).hidden_states[N_PRE]
    diff = (h.float() - ref.float()).abs()
    print(f"partial forward vs full hidden_states[{N_PRE}]: max|diff| {float(diff.max()):.3e}, "
          f"max|h| {float(ref.float().abs().max()):.1f}, rel {float(diff.max() / ref.float().abs().max()):.2e}", flush=True)

    # PCA basis on standardized visual hidden states from a subset of episodes
    pca_path = out_dir / "pca.npz"
    if not pca_path.exists():
        rng = np.random.default_rng(0)
        S1, S2, n_vec = 0.0, 0.0, 0  # running moments in float64 (the vectors themselves do not fit in memory)
        for f in rng.choice(files, min(args.pca_episodes, len(files)), replace=False):
            _, h, _ = episode_features(f)
            x = h[::3, 1:1 + M].reshape(-1, h.shape[-1]).double()
            S1, S2, n_vec = S1 + x.sum(0), S2 + x.T @ x, n_vec + len(x)
            del h, x
        mu = S1 / n_vec
        cov = S2 / n_vec - torch.outer(mu, mu)
        sd = cov.diagonal().clamp_min(0).sqrt() + 1e-6
        evals, evecs = torch.linalg.eigh(cov / torch.outer(sd, sd))  # PCA of standardized vectors
        comp = evecs[:, -args.pca_dim:].flip(1)
        explained = float(evals[-args.pca_dim:].sum() / evals.sum())
        np.savez(pca_path, mean=mu.float().cpu().numpy(), std=sd.float().cpu().numpy(),
                 components=comp.float().cpu().numpy(), explained=explained, n_vectors=n_vec)
        print(f"PCA {args.pca_dim} dims from {n_vec} vectors: explained variance {explained:.4f}", flush=True)
        del S1, S2, cov
    pca = np.load(pca_path)
    mu, sd, comp = (torch.tensor(pca[k], device=DEVICE) for k in ("mean", "std", "components"))

    t0, n = time.time(), 0
    for f in files:
        out = out_dir / Path(f).name
        if out.exists():
            continue
        d, h, a = episode_features(f)
        hv = ((h[:, 1:1 + M] - mu) / sd) @ comp
        s = d["s_layer"].astype(np.float32)
        np.savez(out, hv=hv.half().cpu().numpy(), hlast=h[:, -1].half().cpu().numpy(), avis=a.half().cpu().numpy(),
                 r=s[:, N_PRE:].sum(1), instruction=d["instruction"], episode=d["episode"])
        n += len(hv)
        print(f"{out.name}  T={len(hv)}  {n / (time.time() - t0):.1f} fr/s", flush=True)


if __name__ == "__main__":
    main()
