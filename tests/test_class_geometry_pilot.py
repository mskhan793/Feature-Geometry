"""End-to-end run of scripts/class_geometry_pilot.py on a synthetic cohort and cache (software test only)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import chunk_store

ROOT = Path(__file__).resolve().parents[1]


def test_geometry_pilot_runs_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    rows, feats = [], []
    for c, centre in (("A", 0.0), ("B", 3.0), ("C", 6.0)):
        for s in range(38):
            split = "train" if s < 30 else "val"
            base = rng.normal(centre, 1.0, 24)
            if c == "C" and s % 2:  # C has two groups, split orthogonal to the between-class direction
                base += 3.0 * np.repeat([1.0, -1.0], 12)
            for cam in (1, 2):
                for f in range(2):
                    rows.append({"img": f"{c}{s}_{cam}_{f}", "individual": f"{c}{s:02d}", "taxon_group": c, "fold0": split,
                                 "camera": cam, "hdf5_row": len(rows)})
                    feats.append(base + rng.normal(0, 0.1, 24))
    cohort = pd.DataFrame(rows)
    cohort.to_parquet(tmp_path / "cohort.parquet", index=False)
    (tmp_path / "summary.json").write_text(json.dumps({"excluded_classes": ["D"]}))
    x = np.array(feats, np.float32)
    ident = {"start": 0, "end": len(x), "names_sha1": chunk_store.names_sha1(cohort.img.tolist()), "encoder": "e",
             "model_id": "m", "transform": "t"}
    chunk_store.write_chunk(tmp_path / "cache", 0, x, ident, {})
    run = subprocess.run([sys.executable, "scripts/class_geometry_pilot.py", "--cohort", str(tmp_path / "cohort.parquet"),
                          "--cohort-summary", str(tmp_path / "summary.json"), "--cache", str(tmp_path / "cache"),
                          "--out", str(tmp_path / "out")], cwd=ROOT, capture_output=True, text=True,
                         env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert run.returncode == 0, run.stderr[-3000:]
    r = json.loads((tmp_path / "out" / "results.json").read_text())
    assert r["classes"] == 3 and r["specimens"] == {"train": 90, "val": 24} and r["excluded_upstream"] == ["D"]
    assert r["modality"]["status_counts"]["G3_full"] == {"ineligible": 3}
    per_class = pd.read_csv(tmp_path / "out" / "per_class.csv").set_index("taxon_group")
    assert per_class.loc["C", "verdict"] == "credible multiple groups", per_class.loc["C"].filter(regex="status|ci95|split|usable|smallest").to_dict()
    assert per_class.loc["C", "bic_gain_best2_train"] > per_class.loc["A", "bic_gain_best2_train"]
    sel = json.loads((tmp_path / "out" / "selection.json").read_text())
    assert sel["candidate_groups"][0] == "C"
    for f in ("eigenspectra.png", "class_internal_geometry.png", "class_pair_separation.png", "closest_pairs.png"):
        assert (tmp_path / "out" / f).exists()
