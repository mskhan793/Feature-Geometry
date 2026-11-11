"""scripts/class_geometry_montage.py on a tiny file-based cohort with a single-camera specimen (software test)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_montage_with_single_camera_specimen(tmp_path):
    rows = []
    for ind, cams in (("s1", (0, 1)), ("s2", (1,)), ("s3", (0, 1))):  # s2 has no camera-0 image
        for cam in cams:
            path = f"{ind}/{cam}-{ind}.png"
            (tmp_path / "img" / ind).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 16), (40 * cam, 80, 120)).save(tmp_path / "img" / path)
            rows.append({"img": f"{cam}-{ind}.png", "path": path, "individual": ind, "camera": cam,
                         "fold2": "val" if ind == "s3" else "train"})
    pd.DataFrame(rows).to_parquet(tmp_path / "cohort.parquet", index=False)
    (tmp_path / "selection.json").write_text(json.dumps(
        {"montage_specimens": {"Taxon A": {"component_0": ["s1"], "component_1": ["s2"]}}}))
    r = subprocess.run([sys.executable, "scripts/class_geometry_montage.py", "--cohort", str(tmp_path / "cohort.parquet"),
                        "--image-root", str(tmp_path / "img"), "--selection", str(tmp_path / "selection.json"),
                        "--out", str(tmp_path), "--fold", "fold2"], cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "montage_Taxon_A.png").exists() and "Taxon A: 2 specimens" in r.stdout
