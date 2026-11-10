"""Derived DF24 validation draw and FIN-Benthic2 five-split checks, on small synthetic metadata (unit tests only)."""

import pandas as pd
import pytest

from src.data.df24_validation import derive_validation, near_duplicate_groups
from src.data.finbenthic2_folds import FOLDS, fold_report


def df24_train():
    rows = []
    for cls, n_obs in ((0, 40), (1, 30), (2, 5)):  # class 2 is too rare for validation
        for i in range(n_obs):
            obs = cls * 1000 + i
            for img in range(1 + i % 3):  # 1-3 images per observation
                rows.append({"observationID": obs, "class_id": cls, "rightsHolder": f"r{i % 7}", "year": 2020,
                             "month": 5, "day": 1 + i % 4, "locality": f"L{i % 5}"})
    df = pd.DataFrame(rows)
    df.loc[(df.observationID == 1) & (df.index % 2 == 0), "rightsHolder"] = "other"  # second recorder on one obs
    return df


def test_near_duplicate_groups_link_shared_keys():
    base = {"class_id": 0, "year": 2020, "month": 5, "day": 1, "locality": "L"}
    train = pd.DataFrame([{"observationID": 1, "rightsHolder": "a", **base},
                          {"observationID": 2, "rightsHolder": "a", **base},   # same key as 1
                          {"observationID": 3, "rightsHolder": "b", **base},
                          {"observationID": 3, "rightsHolder": "c", **base},   # second recorder links 3 and 4
                          {"observationID": 4, "rightsHolder": "c", **base},
                          {"observationID": 5, "rightsHolder": "a", **base, "locality": None},  # incomplete key
                          {"observationID": 6, "rightsHolder": "a", **{**base, "class_id": 1}}])
    g = near_duplicate_groups(train)
    assert g[1] == g[2] and g[3] == g[4] and g[1] != g[3]
    assert len({g[5], g[6], g[1], g[3]}) == 4


def test_derived_validation_is_grouped_seeded_and_spares_rare_classes():
    train = df24_train()
    obs, support = derive_validation(train, seed=3)
    again, _ = derive_validation(train, seed=3)
    assert obs.equals(again)
    assert (obs.groupby("group").split.nunique() == 1).all()
    assert obs.observationID.is_unique and obs.images.sum() == len(train)
    s = support.set_index("class_id")
    assert s.loc[2, "status"] == "train_only_too_few" and s.loc[2, "val_observations"] == 0
    assert 0 < s.loc[0, "val_observations"] <= s.loc[0, "target_val"] == 4
    assert (obs[obs.class_id == 2].split == "train").all()


def fivefold():
    rows = []
    for k in range(10):  # 10 specimens, 2 classes, 3 images each; specimen k is test in fold k // 2
        for img in range(3):
            row = {"individual": f"s{k}", "taxon": f"t{k % 2}", "img": f"s{k}_{img}.png"}
            for f, fold in enumerate(FOLDS):
                row[fold] = "test" if k // 2 == f else "val" if (k // 2 + 1) % 5 == f else "train"
            rows.append(row)
    return pd.DataFrame(rows)


def test_fold_report_counts_and_disjoint_tests():
    r = fold_report(fivefold())
    assert r["per_fold"]["fold0"]["test"]["specimens"] == 2 and r["specimens_by_times_in_test"] == {1: 10}
    assert all(v == 0 for v in r["test_overlap_between_folds"].values())
    assert all(r["per_fold"][f]["specimens_in_more_than_one_partition"] == 0 for f in FOLDS)


def test_fold_report_detects_leak_and_bad_values():
    df = fivefold()
    df.loc[0, "fold0"] = "val"  # one image of a test specimen moved to val
    r = fold_report(df)
    assert r["per_fold"]["fold0"]["specimens_in_more_than_one_partition"] == 1
    assert r["per_fold"]["fold0"]["test_specimens_seen_in_train_or_val"] == 1
    df.loc[1, "fold1"] = "holdout"
    with pytest.raises(ValueError, match="fold1 has values"):
        fold_report(df)
