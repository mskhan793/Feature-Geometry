"""Exact-duplicate image groups of a standardised manifest (task.md d57a630, C0: CIFAR-100, Food-101), read-only,
metadata only (ids, labels, splits; no images or features are opened). A group = images sharing ``unit_id`` (the
content hash: CIFAR-100 pixel sha1, Food-101 file sha1). Flags per group: crosses official train/test (train or the
train-derived val vs test), crosses train/val (must be 0: derived validation keeps units whole), conflicting labels.

Writes the flagged image list (every image of a duplicate group, with its group's flags) and a summary. Nothing is
removed and the official partitions are not changed; candidate evaluation-safe rules are listed in the summary for
the author's decision:
  R1 report test metrics also on test minus images whose content occurs in train/val;
  R2 leave conflicting-label groups out of training statistics (all members), keeping them in the official test.

Usage: duplicate_groups.py --manifest DATASET.parquet --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RULES = {"R1": "report test metrics also on official test minus test images whose content occurs in train/val",
         "R2": "leave conflicting-label groups out of training statistics (all members); official test unchanged"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    m = pd.read_parquet(a.manifest, columns=["dataset", "source_id", "unit_id", "label", "split"])
    d = m[m.unit_id.duplicated(keep=False)].copy()
    g = d.groupby("unit_id")
    flags = pd.DataFrame({"group_size": g.size(), "splits": g.split.agg(lambda s: ",".join(sorted(set(s)))),
                          "labels": g.label.agg(lambda s: "|".join(sorted(set(map(str, s)))))})
    has = lambda s: g.split.agg(lambda x: s in set(x))  # noqa: E731
    flags["crosses_trainval_test"] = (has("train") | has("val")) & has("test")
    flags["crosses_train_val"] = has("train") & has("val")
    flags["conflicting_labels"] = g.label.nunique() > 1
    d = d.join(flags, on="unit_id").sort_values(["unit_id", "split", "source_id"])
    name = m.dataset.iloc[0]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out_dir / f"{name}_duplicate_images.csv", index=False)
    test_dup = d[(d.split == "test") & d.crosses_trainval_test]
    conflict = d[d.conflicting_labels]
    summary = {"dataset": name, "manifest": str(a.manifest), "images": int(len(m)), "groups": int(len(flags)),
               "images_in_groups": int(len(d)), "groups_crossing_trainval_test": int(flags.crosses_trainval_test.sum()),
               "groups_crossing_train_val": int(flags.crosses_train_val.sum()),
               "groups_conflicting_labels": int(flags.conflicting_labels.sum()),
               "R1_test_images_flagged": int(len(test_dup)), "R1_official_test_images": int((m.split == "test").sum()),
               "R2_trainval_images_flagged": int(conflict.split.isin(["train", "val"]).sum()),
               "R2_test_images_in_conflicting_groups": int((conflict.split == "test").sum()),
               "candidate_rules_not_applied": RULES}
    (a.out_dir / f"{name}_duplicate_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "candidate_rules_not_applied"}))
    assert summary["groups_crossing_train_val"] == 0, "derived validation split a duplicate group"


if __name__ == "__main__":
    main()
