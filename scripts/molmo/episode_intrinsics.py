"""Per-episode camera intrinsics for a MolmoSpaces LeRobot dataset (they are not stored in LeRobot).

The LeRobot datasets made by molmospaces' mlspaces_multiview_to_lerobot.py with one exterior view per
trajectory have episode i = trajectory i in (sorted h5 file, sorted traj index) order, so the raw h5's
obs/sensor_param/<camera>/intrinsic_cv gives that episode's K at the raw render size (640x368).

    python scripts/molmo/episode_intrinsics.py <raw run dir> <camera> <out.json>   (needs h5py)
"""

import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np

run_dir, camera, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ks = []
for h5 in sorted(glob.glob(str(run_dir / "house_*" / "trajectories_batch_*.h5"))):
    with h5py.File(h5, "r") as f:
        for key in sorted((k for k in f if k.startswith("traj_")), key=lambda k: int(k[5:])):
            ks.append(np.asarray(f[key][f"obs/sensor_param/{camera}/intrinsic_cv"][0]).tolist())
w, h = 2 * ks[0][0][2], 2 * ks[0][1][2]  # principal point at the image centre
json.dump({"camera": camera, "width": w, "height": h, "intrinsics": ks}, open(out, "w"), indent=1)
fx = [k[0][0] for k in ks]
print(f"{len(ks)} episodes, {w:.0f}x{h:.0f}, fx {min(fx):.1f}..{max(fx):.1f} -> {out}")
