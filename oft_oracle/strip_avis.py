"""strip_avis.py -- remove the avis array (layer-0/1 action-query attention, only used by rankers WITH avis) from the
OFT label files to save disk (~10%). Atomic per file (tmp_ file + os.replace); files without avis are skipped.
Usage: python strip_avis.py <label_dir>"""
import glob
import os
import sys

import numpy as np

d = sys.argv[1]
n = saved = 0
for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
    z = np.load(f)
    if "avis" not in z.files:
        continue
    out = {k: z[k] for k in z.files if k != "avis"}
    tmp = os.path.join(d, "tmp_" + os.path.basename(f))
    before = os.path.getsize(f)
    np.savez(tmp, **out)
    os.replace(tmp, f)
    n += 1
    saved += before - os.path.getsize(f)
print(f"{d}: stripped avis from {n} files, saved {saved / 2**30:.1f} GB", flush=True)
