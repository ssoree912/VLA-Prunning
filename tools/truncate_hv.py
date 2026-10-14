"""truncate_hv.py -- keep only the first k PCA coordinates of hv in the label files (the PCA components are stored sorted
by explained variance, so hv[..., :k] is exactly the PCA-k projection). pca.npz keeps all 512 components (predictors
slice them by their pca_dim). Atomic per file (tmp_ + os.replace); files already at <= k dims are skipped.
Usage: python truncate_hv.py <label_dir> <k> [--compressed]"""
import glob
import os
import sys

import numpy as np

d, k, comp = sys.argv[1], int(sys.argv[2]), "--compressed" in sys.argv
n = saved = 0
for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
    z = np.load(f)
    if "hv" not in z.files or z["hv"].shape[-1] <= k:
        continue
    out = {key: (z[key][..., :k] if key == "hv" else z[key]) for key in z.files}
    tmp = os.path.join(d, "tmp_" + os.path.basename(f))
    before = os.path.getsize(f)
    (np.savez_compressed if comp else np.savez)(tmp, **out)
    os.replace(tmp, f)
    n += 1
    saved += before - os.path.getsize(f)
print(f"{d}: hv -> {k} dims in {n} files, saved {saved / 2**30:.1f} GB", flush=True)
