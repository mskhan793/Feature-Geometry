import pandas as pd

from src.data.camera_cohort import select_cohort


def table():
    rows = []
    for c, n_spec in (("A", 6), ("B", 2)):
        for s in range(n_spec):
            for cam in (1, 2):
                if c == "A" and s == 0 and cam == 2:
                    continue  # specimen A0 has camera 1 only: never selected
                for f in range(5):
                    rows.append({"img": f"{c}{s}_{cam}_{f}", "individual": f"{c}{s}", "taxon_group": c, "camera": cam,
                                 "fold0": "val" if s == 1 else "train"})
    df = pd.DataFrame(rows)
    df["hdf5_row"] = range(len(df))[::-1]
    return df


def test_caps_pairing_and_split():
    out = select_cohort(table(), "fold0", "train", ["A", "B"], n_specimens=3, n_frames=2, seed=1)
    assert set(out.fold0) == {"train"}
    assert "A0" not in set(out.individual) and "A1" not in set(out.individual)
    per_class = out.groupby("taxon_group").individual.nunique()
    assert per_class["A"] == 3 and per_class["B"] == 1  # B has one paired train specimen: all used, none duplicated
    assert (out.groupby(["individual", "camera"]).size() == 2).all()
    assert (out.groupby("individual").camera.nunique() == 2).all()
    assert out.img.is_unique and out.hdf5_row.is_monotonic_increasing


def test_deterministic_and_seed_dependent():
    a = select_cohort(table(), "fold0", "train", ["A"], 3, 2, seed=1)
    b = select_cohort(table(), "fold0", "train", ["A"], 3, 2, seed=1)
    c = select_cohort(table(), "fold0", "train", ["A"], 3, 2, seed=2)
    assert a.equals(b)
    assert set(a.img) != set(c.img)
