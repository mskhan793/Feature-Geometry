import pandas as pd

from src.data.expanded_cohort import allocate, select_frames, select_specimens, spread


def test_allocate_floor_and_proportional_rest():
    assert allocate({"a": 5, "b": 10}, 200, 20) == {"a": 5, "b": 10}  # all if under cap
    q = allocate({"big": 1000, "mid": 300, "small": 8}, 200, 20)
    assert sum(q.values()) == 200 and q["small"] == 8  # small group kept whole
    assert q["big"] > q["mid"] >= 20
    q = allocate({"a": 150, "b": 150}, 200, 20)
    assert q == {"a": 100, "b": 100}


def test_select_specimens_cap_and_determinism():
    rows = pd.DataFrame({"individual": [f"s{i}" for i in range(50)], "taxon_group": "A",
                         "dataset": ["x"] * 45 + ["y"] * 5})
    a = select_specimens(rows, 10, 3, seed=1)
    assert len(a) == 10 and (a.dataset == "y").sum() >= 3 and a.individual.is_unique
    assert a.equals(select_specimens(rows, 10, 3, seed=1))
    assert len(select_specimens(rows, None, 3, seed=1)) == 50


def test_spread_is_even_and_distinct():
    assert spread(50, 10) == [0, 5, 11, 16, 22, 27, 33, 38, 44, 49]
    assert spread(4, 10) == [0, 1, 2, 3]


def test_select_frames_spans_runs_and_never_duplicates():
    rows = [{"individual": "s", "camera": 1, "imaging_run": f"r{r}", "img": f"1_s_r{r}_{t:03d}.jpg"}
            for r, n in ((1, 50), (2, 3)) for t in range(n)]
    rows += [{"individual": "s", "camera": 2, "imaging_run": "r1", "img": f"2_s_r1_{t:03d}.jpg"} for t in range(4)]
    out = select_frames(pd.DataFrame(rows), 10)
    c1 = out[out.camera == 1]
    assert len(c1) == 10 and (c1.imaging_run == "r2").sum() == 3  # short run used fully, rest from r1
    assert c1[c1.imaging_run == "r1"].img.tolist()[0] == "1_s_r1_000.jpg"
    assert len(out[out.camera == 2]) == 4 and out.img.is_unique
