"""EXPERIMENT (Marty, 2026-11-25): do "feet in the picture" clues raise task 3 macro-F1?

94 group B task 3 files have no jump found. The Bad ones keep both ankles in view; the
Good/Perfect ones lose an ankle (the child jumped out of the picture, card 2026-11-23 section 6).
Feet clues come from eval.dry_run_count.feet_in_picture (frame 1920x1080):
ankle_missing, lost_at_bottom, reached_edge (0/1 floats).

V0 = features_for(3) (17), V1 = V0 + ankle_missing, V2 = V1 + lost_at_bottom + reached_edge.
make_forest, GroupKFold(5) by child, seeds 0-4 = 25 runs. A variant counts only if the mean F1
gain is >= +0.02 and it beats V0 in >= 18/25 runs. Load and pick the child exactly like
scripts/make_clue_table.py. Writes no files.

Run:  .venv\\Scripts\\python.exe scripts/experiments/exp_t3_feet.py   (a few minutes)
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.pick import pick_child  # noqa: E402
from data.load_b import DEFAULT_ZIP, FPS, list_files, parse_name, read_file  # noqa: E402
from eval.dry_run_count import feet_in_picture  # noqa: E402
from features.clues import compute  # noqa: E402
from model.train_rf import _kappa, _macro_f1, features_for, make_forest  # noqa: E402

MIN_FRAMES = 30  # same as scripts/make_clue_table.py
FRAME_SIZE = (1920, 1080)  # dataset picture size
FEET = ["ankle_missing", "lost_at_bottom", "reached_edge"]
WORDS = {0: "Bad", 1: "Good", 2: "Perfect"}
GAIN_MIN, WINS_MIN = 0.02, 18
CUTOFFS = (0.05, 0.10, 0.20, 0.30)  # GUESSES: flag cutoffs for ankle_missing, compared below


def load() -> pd.DataFrame:
    rows, short, no_torso = [], 0, 0
    with zipfile.ZipFile(DEFAULT_ZIP) as z:
        for name in list_files():
            m = parse_name(name)
            if m["task"] != 3:
                continue
            frames, label, _ = read_file(z, name)
            if len(frames) < MIN_FRAMES:
                short += 1
                continue
            if max(len(fr) for fr in frames) <= 1:
                pts = np.stack(
                    [fr[0] if len(fr) else np.zeros((25, 3), np.float32) for fr in frames]
                )
            else:
                pts = pick_child(frames)
            t = np.arange(len(pts)) / FPS
            try:
                clues = compute(3, pts, t)
            except ValueError:  # no frame with both neck and mid-hip
                no_torso += 1
                continue
            feet = feet_in_picture(pts, FRAME_SIZE)
            rows.append(
                {
                    "file": name.split("/")[-1],
                    "child": m["child"],
                    "label": label,
                    **clues,
                    **{k: float(feet[k]) for k in FEET},
                }
            )
    d = pd.DataFrame(rows)
    print(
        f"task 3: {len(d)} rows | left out {short} under {MIN_FRAMES} frames, {no_torso} no torso"
    )
    return d


def cv(d: pd.DataFrame, cols: list[str]):
    x = d[cols].fillna(-9).to_numpy(dtype=np.float64)  # -9 fill: same as exp_t3_best.py
    y = d["label"].to_numpy(dtype=int)
    g = d["child"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(x, y, g))
    F, K, R = [], [], []
    oof = np.full(len(y), -1)
    for seed in range(5):
        for tr, te in folds:
            pr = make_forest(seed).fit(x[tr], y[tr]).predict(x[te])
            F.append(_macro_f1(y[te], pr))
            K.append(_kappa(y[te], pr))
            R.append(recall_score(y[te], pr, labels=[1], average="macro", zero_division=0))
            if seed == 0:
                oof[te] = pr
    return np.array(F), np.array(K), np.array(R), oof


def crosstab(y: np.ndarray, pred: np.ndarray) -> str:
    tab = pd.crosstab(
        pd.Series(y, name="true").map(WORDS),
        pd.Series(pred, name="pred").map(WORDS),
    ).reindex(index=WORDS.values(), columns=WORDS.values(), fill_value=0)
    return tab.to_string()


def main() -> None:
    d = load()
    nj = d["n_air"] == 0
    print(
        f"no jump found (n_air == 0): {int(nj.sum())} files = "
        f"{int((nj & (d.label == 0)).sum())} Bad + {int((nj & (d.label > 0)).sum())} Good/Perfect"
    )

    base = features_for(3)
    variants = {
        "V0": base,
        "V1 +ankle_missing": base + FEET[:1],
        "V2 +all 3 feet": base + FEET,
    }
    print(f"\nGroupKFold(5) by child, seeds 0-4 = 25 runs, {len(base)} base features")
    print(f"  {'variant':<19} F1     kappa  Good rec  F1 gain  F1 wins  kappa wins  counts")
    res, ref = {}, None
    for title, cols in variants.items():
        F, K, R, oof = cv(d, cols)
        res[title] = (F, oof)
        ref = ref or (F, K)
        gain = F.mean() - ref[0].mean()
        wf, wk = int((F > ref[0]).sum()), int((K > ref[1]).sum())
        ok = "-" if title == "V0" else ("yes" if gain >= GAIN_MIN and wf >= WINS_MIN else "no")
        print(
            f"  {title:<19} {F.mean():.3f}  {K.mean():.3f}  {R.mean():.2f}      "
            f"{gain:+.3f}   {wf:>2}/25    {wk:>2}/25       {ok}"
        )

    best = max((k for k in res if k != "V0"), key=lambda k: res[k][0].mean())
    y = d["label"].to_numpy(dtype=int)
    print(f"\nno-jump files only ({int(nj.sum())}), seed-0 out-of-fold predictions:")
    for title in ("V0", best):
        print(f"[{title}]\n{crosstab(y[nj.to_numpy()], res[title][1][nj.to_numpy()])}")

    print("\nankle_missing median (n) by label:")
    grp = np.where(nj, "no jump", "jump found")
    med = d.assign(grp=grp, lab=d.label.map(WORDS)).groupby(["lab", "grp"])["ankle_missing"]
    tab = med.agg(lambda s: f"{s.median():.2f} ({len(s)})").unstack("grp")
    print(tab.reindex(index=WORDS.values(), columns=["no jump", "jump found"]).to_string())

    # flag = ankle_missing > cutoff, no-jump files only (cutoffs are GUESSES to compare)
    nd = d[nj]
    bad, gp = nd[nd.label == 0], nd[nd.label > 0]
    print("\nno-jump files flagged by ankle_missing > cutoff:")
    print(f"  {'cutoff':<8}Bad flagged   Good+Perfect flagged")
    for c in CUTOFFS:
        nb, ng = int((bad.ankle_missing > c).sum()), int((gp.ankle_missing > c).sum())
        print(f"  {c:<8.2f}{nb:>3}/{len(bad):<10}{ng:>3}/{len(gp)}")


if __name__ == "__main__":
    main()
