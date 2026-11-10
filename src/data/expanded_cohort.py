"""Expanded geometry cohort inside the official AquaMonitor Classification fold0 splits (capping only, no new split).

Specimens (train): per class, all specimens if at most ``cap``; otherwise a source-aware allocation over the
``dataset`` groups (``allocate``): every group first gets ``min(available, floor)``, the remaining slots are shared in
proportion to each group's remaining availability (largest remainder, never above availability). Within a group,
specimens are taken in ``sha1(f"{seed}:{individual}")`` order. Val: every specimen. Specimens imaged by one camera only
are kept.

Frames: per specimen and camera, up to ``n_frames``: slots are dealt round-robin over the imaging runs (run-name order,
never more than a run holds), and within a run the chosen frames are evenly spaced in capture order (image names end
with the capture timestamp, so name order is capture order within a camera and run). Nothing is duplicated.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def stable_order(values: pd.Series, seed: int) -> pd.Series:
    return values.map(lambda v: hashlib.sha1(f"{seed}:{v}".encode()).hexdigest())


def allocate(available: dict[str, int], cap: int, floor: int) -> dict[str, int]:
    if sum(available.values()) <= cap:
        return dict(available)
    take = {g: min(n, floor) for g, n in available.items()}
    while (left := cap - sum(take.values())) > 0:
        room = {g: available[g] - take[g] for g in available if available[g] > take[g]}
        total = sum(room.values())
        share = {g: left * r / total for g, r in room.items()}
        add = {g: min(room[g], int(np.floor(s))) for g, s in share.items()}
        rest = left - sum(add.values())
        for g in sorted(room, key=lambda g: (-(share[g] - np.floor(share[g])), g)):
            if rest == 0:
                break
            if add[g] < room[g]:
                add[g] += 1
                rest -= 1
        for g in add:
            take[g] += add[g]
        if sum(add.values()) == 0:
            break
    return take


def select_specimens(rows: pd.DataFrame, cap: int | None, floor: int, seed: int,
                     source: str | None = "dataset") -> pd.DataFrame:
    """One row per selected specimen (taxon_group, individual and the ``source`` column) from the rows of one split.
    Without a source column (``source=None``) a capped class keeps its first ``cap`` specimens in stable order."""
    agg = {"taxon_group": ("taxon_group", "first")} | ({"dataset": (source, "first")} if source else {})
    spec = rows.groupby("individual").agg(**agg).reset_index()
    spec["key"] = stable_order(spec.individual, seed)
    out = []
    for _, g in spec.groupby("taxon_group"):
        if cap is None:
            out.append(g)
            continue
        if source is None:
            out.append(g.sort_values("key").head(cap))
            continue
        quota = allocate(g.dataset.value_counts().to_dict(), cap, floor)
        out += [s.sort_values("key").head(quota[d]) for d, s in g.groupby("dataset")]
    return pd.concat(out).drop(columns="key").sort_values(["taxon_group", "individual"]).reset_index(drop=True)


def spread(n_available: int, k: int) -> list[int]:
    """k distinct indices evenly spaced over 0..n_available-1 (all if k >= n_available)."""
    if k >= n_available:
        return list(range(n_available))
    return sorted({int(round(x)) for x in np.linspace(0, n_available - 1, k)})


def select_frames(images: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    """Up to ``n_frames`` images per (individual, camera), spread over imaging runs and capture time."""
    picked = []
    for _, g in images.groupby(["individual", "camera"], sort=False):
        runs = [r.sort_values("img") for _, r in g.groupby("imaging_run", sort=True)]
        quota = [0] * len(runs)
        left = min(n_frames, len(g))
        while left:
            for i, r in enumerate(runs):
                if left and quota[i] < len(r):
                    quota[i] += 1
                    left -= 1
        for r, q in zip(runs, quota):
            idx = spread(len(r), q)
            while len(idx) < q:  # rounding collision: add the nearest unused positions
                idx = sorted(set(idx) | {next(i for i in range(len(r)) if i not in idx)})
            picked.append(r.iloc[idx])
    return pd.concat(picked)
