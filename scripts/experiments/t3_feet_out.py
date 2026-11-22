"""EXPERIMENT (Marty, 2026-11-23): do the feet leave the picture in task 3 jumps the finder misses?

Group B task 3, loaded and picked like scripts/make_clue_table.py. Files are split into: a jump
found (features.clues.timeline has an air event), no jump + Bad, no jump + Good/Perfect. Per group:
share of frames (neck and mid-hip seen) with an ankle missing, how many files lose an ankle right
after it was seen near the bottom edge, and the lowest ankle seen (share of picture height).
Writes no files.

Run:  .venv\\Scripts\\python.exe scripts/experiments/t3_feet_out.py
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import body25 as b
from common.pick import pick_child
from data.load_b import DEFAULT_ZIP, FPS, list_files, parse_name, read_file
from features.clues import timeline

H = 1080
EDGE = 0.9  # GUESS: "at the bottom edge" = last seen ankle below 90 % of the picture height
rows = []
with zipfile.ZipFile(DEFAULT_ZIP) as z:
    for name in list_files():
        m = parse_name(name)
        if m["task"] != 3:
            continue
        frames, label, _ = read_file(z, name)
        if len(frames) < 30:
            continue
        pts = (
            np.stack([fr[0] if len(fr) else np.zeros((25, 3), np.float32) for fr in frames])
            if max(len(fr) for fr in frames) <= 1
            else pick_child(frames)
        )
        t = np.arange(len(pts)) / FPS
        ev = timeline(3, pts, t)
        ank = pts[:, [b.L_ANKLE, b.R_ANKLE]]
        ok = ank[..., 2] > 0
        both_ok = ok.all(1)
        body = (pts[:, b.NECK, 2] > 0) & (pts[:, b.MID_HIP, 2] > 0)
        lost = body & ~both_ok  # body seen, an ankle not
        # ankle lost at the bottom edge: last seen y of that ankle before a lost run is near the bottom
        edge = 0
        for j in range(2):
            o = ok[:, j]
            for i in range(1, len(o)):
                if o[i - 1] and not o[i] and body[i] and ank[i - 1, j, 1] > EDGE * H:
                    edge += 1
        rows.append(
            dict(
                label=label,
                camera=m["camera"],
                jump=any(e["state"] == "air" for e in ev),
                lost_share=lost[body].mean() if body.any() else np.nan,
                edge_losses=edge,
                max_ank_y=np.nanmax(np.where(ok, ank[..., 1], np.nan)) / H if ok.any() else np.nan,
            )
        )
d = pd.DataFrame(rows)
d["grp"] = np.where(
    d.jump, "jump found", np.where(d.label > 0, "no jump, Good/Perfect", "no jump, Bad")
)
g = d.groupby("grp")
print(
    pd.DataFrame(
        {
            "files": g.size(),
            "median share of body frames with an ankle missing": g.lost_share.median().round(2),
            "files with an ankle lost at the bottom edge": g.apply(
                lambda x: int((x.edge_losses > 0).sum())
            ),
            "median lowest ankle (share of picture height)": g.max_ank_y.median().round(2),
        }
    ).to_string()
)
print("\nby camera, files with an ankle lost at the bottom edge / files:")
print(d.groupby("camera").apply(lambda x: f"{int((x.edge_losses > 0).sum())}/{len(x)}").to_string())
