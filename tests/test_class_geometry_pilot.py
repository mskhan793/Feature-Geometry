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
                                 "year": "2021", "site": "x"})
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
