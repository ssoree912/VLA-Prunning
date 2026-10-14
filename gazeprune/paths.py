"""Filesystem roots, overridable by environment variables (defaults = the original server layout).

    GAZEPRUNE_ROOT   workspace root                      (default /workspace/vla)
    GAZEPRUNE_DATA   labels, predictors, RLDS data       (default $GAZEPRUNE_ROOT/data)
    GAZEPRUNE_MODELS per-method OFT checkpoint dirs      (default $GAZEPRUNE_ROOT/models)
    GAZEPRUNE_LOGS   evaluation outputs                  (default $GAZEPRUNE_ROOT/eval_logs)
    GAZEPRUNE_BASELINES  baseline repos (official OFT, SpecPrune, VLA-Cache, transformers forks)
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("GAZEPRUNE_ROOT", "/workspace/vla"))
DATA = Path(os.environ.get("GAZEPRUNE_DATA", ROOT / "data"))
MODELS = Path(os.environ.get("GAZEPRUNE_MODELS", ROOT / "models"))
LOGS = Path(os.environ.get("GAZEPRUNE_LOGS", ROOT / "eval_logs"))
BASELINES = Path(os.environ.get("GAZEPRUNE_BASELINES", ROOT / "baselines"))

LABEL_ROOT = DATA / "oft_labels"          # <suite>/ep*.npz (+ pca.npz)
PREDICTOR_ROOT = DATA / "oft_predictor"   # trained rankers: official/<suite>_twohead_no_avis_pca256/best.pt
SCHEDULE_DIR = PREDICTOR_ROOT / "schedules"
RLDS_ROOT = DATA / "modified_libero_rlds"


def oft_checkpoint(suite, method="ours"):
    """Per-method copy of the OFT checkpoint (weights symlinked; get_vla rewrites files inside it)."""
    return MODELS / f"oft-{method}" / f"openvla-7b-oft-finetuned-{suite.replace('_', '-')}-{method}"
