"""src.data.benchmark_cohort on a small fixture manifest (software test only): schema, one unit per (unit, label) for a
content unit with conflicting labels (nothing removed), test rows and train/val-crossing units rejected, support rule."""

import pandas as pd
import pytest

from src.data.benchmark_cohort import SPLIT, cohort, support


def test_cohort_schema_conflicts_and_guards():
    e = pd.DataFrame({"img": ["a", "b", "c", "d", "e"], "unit_id": ["u1", "u1", "u2", "u3", "u3"],
                      "label": ["cat", "dog", "cat", "dog", "dog"], "split": ["train", "train", "val", "train", "train"]})
    c, info = cohort(e)
    assert list(c.columns) == ["img", "taxon_group", "individual", "camera", "imaging_run", SPLIT]
    assert c.individual.tolist() == ["u1|cat", "u1|dog", "u2", "u3", "u3"]
    assert info == {"images": 5, "units": 4, "conflicting_label_units": 1, "conflicting_label_images": 2}
    s = support(c, SPLIT, 2, 1).set_index("taxon_group")
    assert s.loc["cat"].tolist() == [1, 1, False] and s.loc["dog"].tolist() == [2, 0, False]
    with pytest.raises(RuntimeError, match="only train and val"):
        cohort(e.assign(split=["train", "train", "test", "train", "train"]))
    with pytest.raises(RuntimeError, match="both train and val"):
        cohort(e.assign(split=["train", "train", "val", "train", "val"]))
