"""End-to-end run of scripts/camera_gaussian_pilot.py on a synthetic cohort and cache (software test only)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import chunk_store

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_runs_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    rows, feats = [], []
    for c, centre in (("A", 0.0), ("B", 5.0)):
        for s in range(26):
            split = "train" if s < 20 else "val"
            base = rng.normal(centre, 1.0, 20)
            for cam in (1, 2):
                for f in range(2):
                    rows.append({"img": f"{c}{s}_{cam}_{f}", "individual": f"{c}{s}", "taxon_group": c, "fold0": split,
                                 "camera": cam, "imaging_run": f"run{c}{s}", "site": "x", "year": "2021", "lake": "l",
                                 "dataset": "bulk21"})
                    feats.append(base + (2.0 if cam == 2 else 0.0) + rng.normal(0, 0.1, 20))
    cohort = pd.DataFrame(rows)
    cohort.to_parquet(tmp_path / "cohort.parquet", index=False)
    x = np.array(feats, np.float32)
    names = cohort.img.tolist()
    ident = {"start": 0, "end": len(x), "names_sha1": chunk_store.names_sha1(names), "encoder": "e", "model_id": "m",
             "transform": "t"}
    chunk_store.write_chunk(tmp_path / "cache", 0, x, ident, {"feature": "f"})
    run = subprocess.run([sys.executable, "scripts/camera_gaussian_pilot.py", "--cohort", str(tmp_path / "cohort.parquet"),
                          "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "out")], cwd=ROOT,
                         capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert run.returncode == 0, run.stderr[-3000:]
    r = json.loads((tmp_path / "out" / "results.json").read_text())
    assert r["classes_fitted"] == 2 and r["observations"] == {"train": 80, "val": 24}
    assert r["camera_separation_native"]["train_classes_p_below_0.05"] == 2
    diag8 = r["settings"][0]["differences_vs_A_pooled"]
    assert diag8["B_camera_marginal-A_pooled"]["mean"] > 0  # a real camera shift helps the camera model
    assert "C_gmm3-A_pooled" not in diag8  # K=3 needs 30 train specimens
    for f in ("pca_all_classes.png", "pca_by_class.png", "density_differences_diag16.png", "per_class_density.csv"):
        assert (tmp_path / "out" / f).exists()
