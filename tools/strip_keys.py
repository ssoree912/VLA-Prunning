"""strip_keys.py -- remove arrays from npz label/feature files (atomic per file: tmp_ file + os.replace; files that
lack the keys are skipped). Usage: python strip_keys.py <dir> <key> [<key> ...] [--compressed]"""
import glob
import os
import sys

import numpy as np

args = [a for a in sys.argv[1:] if a != "--compressed"]
d, keys, comp = args[0], set(args[1:]), "--compressed" in sys.argv
n = saved = 0
for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
    z = np.load(f)
    if not keys & set(z.files):
        continue
    out = {k: z[k] for k in z.files if k not in keys}
    tmp = os.path.join(d, "tmp_" + os.path.basename(f))
    before = os.path.getsize(f)
    (np.savez_compressed if comp else np.savez)(tmp, **out)
    os.replace(tmp, f)
    n += 1
    saved += before - os.path.getsize(f)
print(f"{d}: removed {sorted(keys)} from {n} files, saved {saved / 2**30:.1f} GB", flush=True)
