"""
label_instr.py -- instruction-side teacher labels + predictor features for the mixed label (base OpenVLA).

For every frame of the existing RLDS label files (oracle_labels/<suite>/epXXXX.npz: images, s_layer, instruction) run
the dense prefill once (vision + projector + all LLM layers, eager attention, batched) and store
    s_instr (T, L, M) f16  instruction tokens -> visual KV attention per layer (head + token mean) -- the instruction
                           term of the mixed label, same definition as the mixed oracle (Oracle B-fixed, label_mix)
    ains    (T, 2H, M) f16 instruction tokens -> visual attention in layers 0, 1 (head-wise)   } predictor inputs at
    hins    (T, 4096) f16  mean instruction-token hidden state at the layer-1 output         } the layer-2 entry
Instruction tokens: input_ids range [len(tok(prefix)), len(tok(prefix + " " + instruction))), sequence position M + j.

Run from src/openvla:
    CUDA_VISIBLE_DEVICES=1 python oracle_scripts/label_instr.py --suite libero_goal --num_shards 2 --shard 0
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import DEVICE, get_processor, get_vla, process_image

M, N_PRE = 256, 2
PREFIX = "In: What action should the robot take to"


@torch.no_grad()
def prefill_instr(vla, ids, pixel_values, q):
    """Full-depth prefill. Returns s_instr (B, L, M), ains (B, 2H, M), hins (B, 4096) for instruction rows q."""
    lm = vla.language_model.model
    proj = vla.projector(vla.vision_backbone(pixel_values))
    emb = vla.get_input_embeddings()(ids)
    h = torch.cat([emb[:, :1], proj, emb[:, 1:]], dim=1)
    pos = torch.arange(h.shape[1], device=h.device)
    mask = lm._update_causal_mask(None, h, 0)
    s, rows, hins = [], [], None
    for l, layer in enumerate(lm.layers):
        out = layer(h, attention_mask=mask, position_ids=pos[None].expand(h.shape[0], -1), output_attentions=True,
                    use_cache=False, cache_position=pos)
        h = out[0]
        a = out[1][:, :, q, 1:1 + M].float()  # (B, H, |q|, M)
        s.append(a.mean(dim=(1, 2)))
        if l < N_PRE:
            rows.append(a.mean(dim=2))  # (B, H, M)
        if l == N_PRE - 1:
            hins = h[:, q].float().mean(dim=1)
    return torch.stack(s, dim=1), torch.cat(rows, dim=1), hins


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    src = Path(f"/workspace/vla/data/oracle_labels/{args.suite}")
    out_dir = Path(f"/workspace/vla/data/oracle_instr_labels/{args.suite}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = GenerateConfig(pretrained_checkpoint=f"checkpoints/openvla-7b-finetuned-{args.suite.replace('_', '-')}",
                         task_suite_name=args.suite, use_fastv=False)
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    tok = processor.tokenizer
    files = sorted(glob.glob(str(src / "ep*.npz")))
    t0, n = time.time(), 0
    for i, f in enumerate(files):
        out = out_dir / Path(f).name
        if i % args.num_shards != args.shard or out.exists():
            continue
        d = np.load(f)
        instr = str(d["instruction"])
        a, b = len(tok(PREFIX).input_ids), len(tok(f"{PREFIX} {instr.lower()}").input_ids)
        q = torch.arange(M + a, M + b)
        prompt = f"{PREFIX} {instr.lower()}?\nOut:"
        S, A, Hh = [], [], []
        for j in range(0, len(d["images"]), args.batch):
            imgs = [process_image(Image.fromarray(x).convert("RGB")) for x in d["images"][j:j + args.batch]]
            inputs = processor([prompt] * len(imgs), imgs).to(DEVICE, dtype=torch.bfloat16)
            ids = inputs["input_ids"]
            if not torch.all(ids[:, -1] == 29871):
                ids = torch.cat((ids, torch.full((ids.shape[0], 1), 29871, device=ids.device)), dim=1)
            s, rows, hins = prefill_instr(vla, ids, inputs["pixel_values"], q.to(ids.device))
            S.append(s.half().cpu())
            A.append(rows.half().cpu())
            Hh.append(hins.half().cpu())
        np.savez(out, s_instr=torch.cat(S).numpy(), ains=torch.cat(A).numpy(), hins=torch.cat(Hh).numpy(),
                 instr_span=np.array([a, b]), instruction=instr, episode=d["episode"])
        n += len(d["images"])
        print(f"{out.name}  T={len(d['images'])}  span={a}:{b}  {n / (time.time() - t0):.1f} fr/s", flush=True)


if __name__ == "__main__":
    main()
