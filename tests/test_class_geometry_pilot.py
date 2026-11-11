"""End-to-end run of scripts/class_geometry_pilot.py and scripts/component_metadata.py on a synthetic cohort and two
caches (software test only)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import chunk_store

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-3000:]


def write_cache(tmp_path, name, rows, feats):
    cohort = pd.DataFrame(rows)
    cohort.to_parquet(tmp_path / f"{name}.parquet", index=False)
    ident = {"start": 0, "end": len(feats), "names_sha1": chunk_store.names_sha1(cohort.img.tolist()), "encoder": "e",
             "model_id": "m", "transform": "t"}
    chunk_store.write_chunk(tmp_path / name, 0, np.array(feats, np.float32), ident, {"feature": "f", "tf32": False})
    return f"{tmp_path / name}.parquet={tmp_path / name}"


def test_geometry_pilot_runs_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    rows, feats = [], []
    for c, centre in (("A", 0.0), ("B", 3.0), ("C", 6.0), ("D", 9.0)):
        for s in range(70 if c != "D" else 12):  # D has too few train specimens
            split = "train" if s < 60 else "val"
            base = rng.normal(centre, 1.0, 24)
            odd = c == "C" and s % 2
            if odd:  # C has two groups, split orthogonal to the between-class direction, aligned with dataset
                base += 4.0 * np.repeat([1.0, -1.0], 12)
            for cam in ((1,) if (c == "A" and s == 0) else (1, 2)):  # A00 has one camera only
                for f in range(2):
                    rows.append({"img": f"{c}{s}_{cam}_{f}", "individual": f"{c}{s:02d}", "taxon_group": c, "fold0": split,
                                 "camera": cam, "hdf5_row": len(rows), "dataset": "dna" if odd else "bulk21",
                                 "year": "2021", "site": "x", "width": 40 if odd else 20, "height": 30})
                    feats.append(base + rng.normal(0, 0.1, 24))
    half = len(rows) // 2  # features split over two caches, looked up by name
    caches = [write_cache(tmp_path, "c1", rows[:half], feats[:half]), write_cache(tmp_path, "c2", rows[half:], feats[half:])]
    pd.DataFrame(rows).to_parquet(tmp_path / "cohort.parquet", index=False)
    out = tmp_path / "out"
    run(["scripts/class_geometry_pilot.py", "--cohort", str(tmp_path / "cohort.parquet"), "--caches", *caches,
         "--out", str(out), "--pcs", "8", "16", "--n-stab", "30", "--jobs", "2"])
    r = json.loads((out / "results.json").read_text())
    assert r["classes"] == 3 and r["specimens"] == {"train": 192, "val": 30}
    assert r["classes_insufficient_support"] == {"D": {"train": 12, "val": 0}}
    assert r["specimens_one_camera"] == {"train": 1, "val": 0}
    assert set(r["modality"]["by_pcs"]) == {"8", "16"} and r["modality"]["by_pcs"]["16"]["params"]["G1_full"] == 152
    mod = pd.read_csv(out / "per_class_modality.csv").set_index(["taxon_group", "pcs"])
    assert mod.loc[("C", 8), "verdict"] == mod.loc[("C", 16), "verdict"] == "credible multiple groups"
    assert mod.loc[("C", 8), "GD_tied-G1_full"] > 0  # the dataset model also explains C's split
    assert json.loads((out / "selection.json").read_text())["candidate_groups"][0] == "C"
    for f in ("eigenspectra.png", "class_internal_geometry.png", "class_pair_separation.png", "closest_pairs.png",
              "per_class_geometry.csv", "class_pairs.csv", "specimen_components.csv"):
        assert (out / f).exists()
    run(["scripts/component_metadata.py", "--cohort", str(tmp_path / "cohort.parquet"), "--components",
         str(out / "specimen_components.csv"), "--modality", str(out / "per_class_modality.csv"),
         "--geometry", str(out / "per_class_geometry.csv"), "--out", str(out)])
    meta = json.loads((out / "component_metadata.json").read_text())
    assert meta["8_pcs_train"]["dataset"]["classes_ari_at_least_0.5_list"] == ["C"]
    assert len(pd.read_csv(out / "per_class_summary.csv")) == 6  # 3 classes x 2 dimensions

    # a dataset without a source field (FIN-Benthic2): no GD_tied; numeric image-size check instead of ARIs
    plain = tmp_path / "plain"
    run(["scripts/class_geometry_pilot.py", "--cohort", str(tmp_path / "cohort.parquet"), "--caches", *caches,
         "--out", str(plain), "--pcs", "8", "--n-stab", "30", "--jobs", "2", "--source-column", "none"])
    mod = pd.read_csv(plain / "per_class_modality.csv").set_index("taxon_group")
    assert "GD_tied_status" not in mod and mod.loc["C", "verdict"] == "credible multiple groups"
    run(["scripts/component_metadata.py", "--cohort", str(tmp_path / "cohort.parquet"), "--components",
         str(plain / "specimen_components.csv"), "--modality", str(plain / "per_class_modality.csv"),
         "--geometry", str(plain / "per_class_geometry.csv"), "--out", str(plain), "--fields", "--numeric", "log_image_area"])
    size = json.loads((plain / "component_metadata.json").read_text())["8_pcs_train"]["log_image_area"]
    assert size["classes_auc_outside_0.2_0.8_list"] == ["C"]  # C's groups differ in image size

    # another fold column with test rows that have no features: they must be dropped before any feature lookup
    full = pd.DataFrame(rows).assign(fold3=lambda d: d.fold0)
    test_rows = full.head(6).assign(img=lambda d: "test_" + d.img, individual="T1", fold3="test")
    pd.concat([full, test_rows]).to_parquet(tmp_path / "five_fold.parquet", index=False)
    f3 = tmp_path / "f3"
    run(["scripts/class_geometry_pilot.py", "--cohort", str(tmp_path / "five_fold.parquet"), "--caches", *caches,
         "--out", str(f3), "--pcs", "8", "--n-stab", "30", "--jobs", "2", "--fold", "fold3"])
    r3 = json.loads((f3 / "results.json").read_text())
    assert r3["fold"] == "fold3" and r3["cohort_rows"]["test_not_loaded"] == 6
    assert r3["specimens"] == r["specimens"]
    assert "fold3" in pd.read_csv(f3 / "specimen_components.csv")
    run(["scripts/component_metadata.py", "--cohort", str(tmp_path / "five_fold.parquet"), "--components",
         str(f3 / "specimen_components.csv"), "--modality", str(f3 / "per_class_modality.csv"),
         "--geometry", str(f3 / "per_class_geometry.csv"), "--out", str(f3), "--fold", "fold3"])
    assert json.loads((f3 / "component_metadata.json").read_text())["8_pcs_train"]["dataset"][
        "classes_ari_at_least_0.5_list"] == ["C"]

    # descriptive (in-sample) output on the pooled train + val specimens of fold3; test rows are not read
    desc = tmp_path / "desc"
    run(["scripts/descriptive_geometry.py", "--cohort", str(tmp_path / "five_fold.parquet"), "--caches", *caches,
         "--out", str(desc), "--fold", "fold3", "--pcs", "8", "--jobs", "2", "--numeric", "log_image_area"])
    rd = json.loads((desc / "results.json").read_text())
    assert rd["cohort_rows"]["not_read"] == 6 and rd["specimens"] == 3 * 70 + 12 and rd["classes"] == 3
    assert rd["classes_below_min"] == {"D": 12}
    m8 = rd["mixture_in_sample_by_pcs"]["8"]
    assert m8["source_ari_at_least_0.5"] == ["C"] and m8["log_image_area_auc_outside_0.2_0.8"] == ["C"]
    assert len(pd.read_csv(desc / "class_pairs.csv")) == 3

    # five-fold summary: five identical copies of the fold3 run give zero spread across folds
    import shutil
    runs = tmp_path / "runs"
    for k in range(5):
        shutil.copytree(f3, runs / f"fold{k}")
    run(["scripts/summarise_five_folds.py", "--dataset", "Toy", str(runs), str(out), "dataset", "--out", str(tmp_path / "sum")])
    s = json.loads((tmp_path / "sum" / "Toy" / "summary.json").read_text())
    q = s["across_folds"]["pcs8_G2_tied-G1_full"]
    assert q["min"] == q["max"] and s["folds_ci_above_0"]["pcs8_G3_tied-G1_full"] in (0, 5)
    assert s["classes_credible_in_all_folds"]["8"] == ["C"] and s["pairs"]["in_all_folds"] == 3
    assert s["across_folds"]["pcs8_dataset_associated"]["mean"] == 1
    assert (tmp_path / "sum" / "comparison.md").exists()


def test_compare_caches_by_name(tmp_path):
    rng = np.random.default_rng(2)
    names = [f"i{k}" for k in range(40)]
    x = rng.normal(size=(40, 8)).astype(np.float32)
    one = write_cache(tmp_path, "one", {"img": names}, x)  # one cache, natural order
    order = rng.permutation(40)  # the other side: two caches, shuffled, tiny differences
    y = x[order] + np.float32(1e-6)
    two = [write_cache(tmp_path, "p", {"img": [names[i] for i in order[:25]]}, y[:25]),
           write_cache(tmp_path, "q", {"img": [names[i] for i in order[25:]]}, y[25:])]
    pd.DataFrame({"img": names[::2]}).to_parquet(tmp_path / "names.parquet", index=False)
    run(["scripts/compare_caches_by_name.py", "--names", str(tmp_path / "names.parquet"), "--a", one, "--b", *two,
         "--out", str(tmp_path / "cmp.json")])
    r = json.loads((tmp_path / "cmp.json").read_text())
    assert r["rows"] == 20 and 0 < r["max_abs_diff"] < 1e-5 and r["min_cosine"] > 0.999999


def test_reidentification_chunked_matches_dense():
    from src.statistics.specimen_geometry import reidentification
    rng = np.random.default_rng(1)
    a = rng.normal(size=(300, 16))
    b = a + rng.normal(scale=0.8, size=a.shape)
    dense = float(np.mean(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).argmin(1) == np.arange(len(a))))
    assert 0 < dense < 1 and reidentification(a, b, chunk=7) == dense
