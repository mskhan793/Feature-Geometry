"""Five-fold specimen checks and the fold-independence of the frame rule (software test, small fixture)."""

import pandas as pd
import pytest

from src.data.expanded_cohort import select_frames
from src.data.full_specimen import FOLDS, fold_checks


def rows(n_specimens=5, frames=3):
    out = []
    for s in range(n_specimens):
        folds = {f: "test" if k == s else ("val" if k == (s + 1) % 5 else "train") for k, f in enumerate(FOLDS)}
        for cam in (0, 1):
            for i in range(frames):
                out.append({"img": f"{cam}-s{s}.1.{i:03d}", "individual": f"s{s}", "taxon_group": "A" if s % 2 else "B",
                            "camera": cam, "imaging_run": f"s{s}.1", **folds})
    return pd.DataFrame(out)


def test_fold_checks_counts():
    counts = fold_checks(rows())
    assert counts["fold0"] == {"train": 3, "val": 1, "test": 1}
    assert all(sum(c.values()) == 5 for c in counts.values())


def test_fold_checks_rejects_partition_change_within_specimen():
    r = rows()
    r.loc[0, "fold0"] = "train"  # s0 is test in fold0; one of its images marked train
    with pytest.raises(RuntimeError, match="partition"):
        fold_checks(r)


def test_fold_checks_rejects_specimen_never_in_test():
    r = rows()
    r.loc[r.individual == "s0", "fold0"] = "train"
    with pytest.raises(RuntimeError, match="exactly one fold"):
        fold_checks(r)


def test_frame_rule_does_not_depend_on_other_specimens():
    r = rows(frames=15)
    full = set(select_frames(r, 10).img)
    alone = set(select_frames(r[r.individual == "s2"], 10).img)
    assert alone == {i for i in full if "-s2." in i} and len(alone) == 20
